"""Base contract for AlphaZero-style trainers, plus the built-in reference
implementation that ships with Reinforcement Studio.

The built-in trainer follows nanoAlphaZero: batched self-play of many games
at once with the "1sh" Gumbel search (`rl_core/alphazero/gumbel.py`),
improved-policy targets, a win/draw/loss value head, a large replay buffer
where each position is trained on about `sample_reuse` times, and a
fixed-opponent Elo ladder instead of an arena gate
(`rl_core/alphazero/batched.py`).

Custom trainers (see rl_core/plugins/loader.py and the "AlphaZero" templates
on the Plugins page) subclass `AlphaZeroTrainer` and override either just
`build_network()` (swap the architecture, keep the built-in loop) or the
whole `run_iteration()` (replace everything). The shared
iterate/checkpoint/metrics-snapshot driver lives in rl_core/alphazero/loop.py
and works with any subclass.
"""
from __future__ import annotations

import math
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_core.alphazero.batched import EloLadder, SelfPlayPool, make_openings, play_match
from rl_core.alphazero.checkpoint import network_kwargs, save_checkpoint
from rl_core.alphazero.gumbel import NetEvaluator, SearchConfig
from rl_core.alphazero.network import AlphaZeroNet
from rl_core.alphazero.replay import ReplayBuffer
from rl_core.alphazero.selfplay_workers import WorkerSelfPlay
from rl_core.games.base import BoardGame
from rl_core.netbuilder import SpecAlphaZeroNet

DEFAULT_HYPERPARAMS = {
    # Built-in trainer.
    "parallel_games": 64,
    "selfplay_processes": 0,
    "plies_per_iteration": 16,
    "num_considered": 8,
    "num_survivors": 4,
    "root_temperature": 1.0,
    "exploration_moves": 8,
    "max_game_plies": 0,
    "batch_size": 256,
    "sample_reuse": 2.0,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "lr_warmup_steps": 200,
    "buffer_size": 100_000,
    "min_replay_positions": 2048,
    "eval_every": 5,
    "eval_games": 20,
    "eval_opening_plies": 1,
    "eval_max_plies": 0,
    "eval_promote_score": 0.75,
    "eval_promote_patience": 2,
    "channels": 64,
    "num_blocks": 4,
    "value_head": "wdl",
    "global_pool": True,
    "policy_channels": 8,
    # PUCT MCTS helpers (`self_play.py`, `arena.py`) used by custom plugins
    # and the Arena page.
    "num_simulations": 25,
    "games_per_iteration": 20,
    "epochs": 4,
    "eval_num_simulations": 30,
    "win_rate_threshold": 0.55,
    "c_puct": 1.5,
    "self_play_workers": 1,
}


class AlphaZeroTrainer(ABC):
    """Contract a custom AlphaZero implementation must satisfy.

    Required override: `run_iteration(iteration)` -> a dict of extra metrics
    to merge into that iteration's snapshot (`loss`, `eval_ladder_elo`,
    `board_history`, ... — whatever keys you want surfaced on the Training
    Monitor page). `board_history` (a list of board matrices, one per move)
    lets the Monitor replay a whole game move by move instead of a single
    freeze-frame — `play_self_play_game` already returns one per record.
    Include a `"_self_play_records"` key (a list of move-history dicts) if
    you want games viewable on the AlphaZero Arena replay page — the driver
    strips this key out before writing metrics.json.

    Optional override: `build_network()` if you only want to swap the
    network architecture and keep the built-in self-play/search/training
    loop — see `BuiltinAlphaZeroTrainer.run_iteration` for the reference
    loop you can copy and adapt for a full custom implementation.
    """

    def __init__(self, game_cls: type[BoardGame], hyperparams: dict[str, Any], device: str) -> None:
        self.game_cls = game_cls
        self.hp = {**DEFAULT_HYPERPARAMS, **hyperparams}
        self.device = device
        self.sample_game = game_cls()
        self.net = self.build_network()
        self.net.to(device)

    def build_network(self) -> nn.Module:
        """Default: the built-in dual-head ResNet, unless
        `hyperparams["network_spec"]` was set (a hand-designed architecture
        picked in the Designer, resolved by `rl_core.netbuilder_store` before
        the trainer is constructed) — then a `SpecAlphaZeroNet` built from
        that spec is used instead. Override to plug in a different
        architecture entirely — must expose `.rows`, `.cols`, `.action_size`
        attributes (for checkpointing), `forward(x) -> (policy_logits,
        value in [-1, 1])` and a `predict(encoded_state, device) ->
        (policy_probs, value)` method (used by the Arena's MCTS)."""
        g = self.sample_game
        in_planes = int(g.input_planes)
        network_spec = self.hp.get("network_spec")
        if network_spec:
            return SpecAlphaZeroNet(g.rows, g.cols, g.action_size, network_spec, in_planes=in_planes)
        return AlphaZeroNet(
            g.rows, g.cols, g.action_size, self.hp["channels"], self.hp["num_blocks"], in_planes=in_planes,
            **network_kwargs(self.hp),
        )

    @abstractmethod
    def run_iteration(self, iteration: int) -> dict[str, Any]: ...

    def close(self) -> None:
        """Releases background resources (worker processes, ...). Called by
        the training loop once the run ends."""

    def save(self, path: Path, meta: dict[str, Any]) -> None:
        save_checkpoint(self.net, path, meta)

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    @property
    def network_info(self) -> dict[str, Any]:
        g = self.sample_game
        info: dict[str, Any] = {
            "channels": self.hp.get("channels"),
            "num_blocks": self.hp.get("num_blocks"),
            "rows": g.rows,
            "cols": g.cols,
            "action_size": g.action_size,
            "input_planes": int(g.input_planes),
        }
        if getattr(self.net, "value_head_type", None):
            info["value_head"] = self.net.value_head_type
        if self.hp.get("network_spec"):
            info["network_spec"] = self.hp["network_spec"]
        return info


class BuiltinAlphaZeroTrainer(AlphaZeroTrainer):
    """Reference loop — exactly what Reinforcement Studio ships by default.

    One iteration = `plies_per_iteration` moves in each of the
    `parallel_games` concurrent self-play games, then enough gradient steps
    that every newly finished position is sampled about `sample_reuse`
    times, then (every `eval_every` iterations) a ladder evaluation.
    Kept as a normal subclassable class so custom trainers can reuse it and
    override only one piece (typically `build_network`)."""

    def __init__(self, game_cls: type[BoardGame], hyperparams: dict[str, Any], device: str) -> None:
        super().__init__(game_cls, hyperparams, device)
        hp = self.hp
        self.parallel_games = max(1, int(hp["parallel_games"]))
        hp["self_play_workers_requested"] = int(hp.get("self_play_workers", 1))
        hp["self_play_workers"] = self.parallel_games
        seed = hp.get("seed")
        self.rng = np.random.default_rng(None if seed is None else int(seed))
        self.optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=float(hp["learning_rate"]), weight_decay=float(hp["weight_decay"]),
        )
        g = self.sample_game
        self.replay = ReplayBuffer(int(hp["buffer_size"]), g.encode().shape, g.action_size)
        self.evaluator = NetEvaluator(self.net, device)
        self.ladder = EloLadder(float(hp["eval_promote_score"]), int(hp["eval_promote_patience"]))
        self.pool: SelfPlayPool | None = None
        self.workers: WorkerSelfPlay | None = None
        self.selfplay_processes = self._resolve_processes(int(hp["selfplay_processes"]))
        hp["selfplay_processes_resolved"] = self.selfplay_processes
        if str(device).startswith("cuda"):
            self.net.to(memory_format=torch.channels_last)
        self.train_steps = 0
        self.games_played = 0
        self._untrained_positions = 0.0
        self._last_eval: dict[str, Any] = {}
        self._openings: list[list[int]] | None = None
        self._uses_wdl = getattr(self.net, "value_head_type", None) == "wdl" and hasattr(self.net, "forward_heads")
        self._use_bf16 = str(device).startswith("cuda") and torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    def _resolve_processes(self, requested: int) -> int:
        """0 = auto: worker processes pay off once there are enough games
        to keep each busy; below that the in-process pool is faster."""
        if requested > 0:
            return min(requested, self.parallel_games)
        cpus = os.cpu_count() or 1
        if self.parallel_games < 64 or cpus < 4:
            return 1
        return max(1, min(cpus - 2, self.parallel_games // 8, 32))

    def close(self) -> None:
        if self.workers is not None:
            self.workers.close()
            self.workers = None

    def _play_selfplay(self, cfg: SearchConfig, plies: int) -> tuple[list[Any], list[Any]]:
        hp = self.hp
        if self.selfplay_processes > 1:
            if self.workers is None:
                self.workers = WorkerSelfPlay(
                    self.game_cls, self.parallel_games, self.selfplay_processes,
                    seed=int(self.rng.integers(1 << 31)),
                    exploration_moves=int(hp["exploration_moves"]), max_plies=int(hp["max_game_plies"]),
                    max_considered=cfg.num_considered,
                )
            return self.workers.play(self.evaluator, cfg, plies)
        if self.pool is None:
            self.pool = SelfPlayPool(
                self.game_cls, self.parallel_games, self.rng,
                exploration_moves=int(hp["exploration_moves"]), max_plies=int(hp["max_game_plies"]),
            )
        self.pool.forget_evaluations()
        samples: list[Any] = []
        finished: list[Any] = []
        for _ in range(plies):
            s, f = self.pool.play_ply(self.evaluator, cfg)
            samples += s
            finished += f
        return samples, finished

    def search_config(self, evaluation: bool = False) -> SearchConfig:
        hp = self.hp
        return SearchConfig(
            num_considered=max(1, int(hp["num_considered"])),
            num_survivors=max(1, int(hp["num_survivors"])),
            root_temperature=max(1e-3, float(hp["root_temperature"])),
            gumbel_scale=0.0 if evaluation else 1.0,
        )

    # ------------------------------------------------------------ training

    def _train_step(self) -> tuple[float, float]:
        net, device = self.net, self.device
        bs = int(self.hp["batch_size"])
        obs, idx, prob, outcome = self.replay.sample(bs, self.rng)
        x = torch.from_numpy(obs).to(device, non_blocking=True)
        if str(device).startswith("cuda"):
            x = x.contiguous(memory_format=torch.channels_last)
        idx_t = torch.from_numpy(idx).to(device, non_blocking=True)
        prob_t = torch.from_numpy(prob).to(device, non_blocking=True)
        outcome_t = torch.from_numpy(outcome).to(device, non_blocking=True)
        target = torch.zeros(bs, self.sample_game.action_size, device=device).scatter_add_(1, idx_t, prob_t)

        warmup = max(1, int(self.hp["lr_warmup_steps"]))
        lr = float(self.hp["learning_rate"]) * min(1.0, (self.train_steps + 1) / warmup)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self._use_bf16):
            if self._uses_wdl:
                logits, value_out = net.forward_heads(x)
            else:
                logits, value_out = net(x)
        policy_loss = -(target * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()
        if self._uses_wdl:
            value_loss = F.cross_entropy(value_out.float(), outcome_t)
        else:
            value_loss = F.mse_loss(value_out.float().reshape(-1), (1 - outcome_t).float())
        loss = policy_loss + value_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        self.optimizer.step()
        self.train_steps += 1
        return float(policy_loss.item()), float(value_loss.item())

    def _train(self) -> dict[str, float]:
        bs = int(self.hp["batch_size"])
        if len(self.replay) < max(bs, int(self.hp["min_replay_positions"])):
            return {"train_steps": 0}
        steps = int(self._untrained_positions * float(self.hp["sample_reuse"]) / bs)
        if steps <= 0:
            return {"train_steps": 0}
        self._untrained_positions -= steps * bs / float(self.hp["sample_reuse"])
        self.net.train()
        policy_losses, value_losses = [], []
        for _ in range(steps):
            p, v = self._train_step()
            policy_losses.append(p)
            value_losses.append(v)
        self.net.eval()
        policy_loss = float(np.mean(policy_losses))
        value_loss = float(np.mean(value_losses))
        return {
            "train_steps": steps,
            "loss": policy_loss + value_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
        }

    # ---------------------------------------------------------- evaluation

    def _evaluate(self) -> dict[str, Any]:
        hp = self.hp
        pairs = max(1, int(hp["eval_games"]) // 2)
        if self._openings is None:
            self._openings = make_openings(self.game_cls, pairs, int(hp["eval_opening_plies"]))
        cfg = self.search_config(evaluation=True)
        max_plies = int(hp["eval_max_plies"])
        rng = np.random.default_rng(int(self.rng.integers(1 << 31)))
        vs_random = play_match(self.game_cls, self.evaluator, None, self._openings, cfg, rng, max_plies)
        ladder = self.ladder
        if ladder.anchor_state is None:
            vs_anchor = vs_random
        else:
            anchor = self.build_network()
            anchor.load_state_dict(ladder.anchor_state)
            anchor.to(self.device).eval()
            vs_anchor = play_match(
                self.game_cls, self.evaluator, NetEvaluator(anchor, self.device), self._openings, cfg, rng, max_plies,
            )
        rung_before = ladder.rung
        promoted = ladder.update(vs_anchor["score"], vs_anchor["games"], self.net)
        return {
            "eval_score_vs_random": round(vs_random["score"], 3),
            "eval_random_wins": vs_random["wins"],
            "eval_random_draws": vs_random["draws"],
            "eval_random_losses": vs_random["losses"],
            "eval_score_vs_anchor": round(vs_anchor["score"], 3),
            "eval_anchor_wins": vs_anchor["wins"],
            "eval_anchor_draws": vs_anchor["draws"],
            "eval_anchor_losses": vs_anchor["losses"],
            "eval_anchor_rung": rung_before,
            "eval_ladder_elo": round(ladder.ladder_elo, 1),
            "eval_ladder_rung": ladder.rung,
            "eval_promoted": promoted,
            "eval_games": vs_anchor["games"],
        }

    # ------------------------------------------------------------ iteration

    def run_iteration(self, iteration: int) -> dict[str, Any]:
        hp = self.hp
        self.net.eval()
        cfg = self.search_config()
        t0 = time.perf_counter()
        plies = max(1, int(hp["plies_per_iteration"]))
        samples, finished = self._play_selfplay(cfg, plies)
        for sample in samples:
            self.replay.add(*sample)
        new_positions = len(samples)
        selfplay_seconds = time.perf_counter() - t0
        self._untrained_positions += new_positions

        t1 = time.perf_counter()
        train_metrics = self._train()
        train_seconds = time.perf_counter() - t1

        every = int(hp["eval_every"])
        evaluated = every > 0 and (iteration % every == 0 or iteration == 1)
        eval_seconds = 0.0
        if evaluated:
            t2 = time.perf_counter()
            self._last_eval = self._evaluate()
            eval_seconds = time.perf_counter() - t2
            self._last_eval["eval_iteration"] = iteration

        games = len(finished)
        self.games_played += games
        first = sum(1 for f in finished if f.winner == 1)
        second = sum(1 for f in finished if f.winner == -1)
        truncations = sum(1 for f in finished if f.truncated)
        draws = games - first - second
        records = [f.record for f in finished if f.record is not None][-3:]
        last_game = records[-1] if records else None

        metrics: dict[str, Any] = {
            "episodes_completed": self.games_played,
            "buffer_size": len(self.replay),
            "positions_seen": self.replay.total_added,
            "new_positions": new_positions,
            "train_steps_total": self.train_steps,
            "selfplay_plies_per_sec": round(plies * self.parallel_games / max(selfplay_seconds, 1e-9), 1),
            "selfplay_seconds": round(selfplay_seconds, 2),
            "train_seconds": round(train_seconds, 2),
            "eval_seconds": round(eval_seconds, 2),
            "learning_rate_current": self.optimizer.param_groups[0]["lr"],
            "self_play_workers": self.parallel_games,
            "selfplay_processes": self.selfplay_processes,
            "self_play_workers_requested": int(hp["self_play_workers_requested"]),
            "_self_play_records": records,
        }
        for key in ("loss", "policy_loss", "value_loss"):
            if key in train_metrics:
                metrics[key] = round(train_metrics[key], 4)
        metrics["train_steps"] = train_metrics["train_steps"]
        if games:
            metrics.update({
                "self_play_games": games,
                "self_play_first_player_wins": first,
                "self_play_second_player_wins": second,
                "self_play_draws": draws,
                "self_play_truncations": truncations,
                "self_play_decisive_games": first + second,
                "self_play_decisive_rate": round((first + second) / games, 3),
                "self_play_draw_rate": round(draws / games, 3),
                "self_play_avg_length": round(sum(f.plies for f in finished) / games, 1),
            })
        if last_game is not None:
            metrics["board_history"] = last_game["board_history"]
            metrics["board_winner"] = last_game["winner"]
        metrics.update(self._last_eval)
        if not math.isfinite(float(metrics.get("loss", 0.0))):
            raise FloatingPointError("AlphaZero loss became NaN/inf")
        return metrics
