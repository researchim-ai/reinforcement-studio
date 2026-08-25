"""Base contract for AlphaZero-style trainers, plus the built-in reference
implementation (self-play -> train -> arena-gate) that ships with
Reinforcement Studio.

Custom trainers (see rl_core/plugins/loader.py and the "AlphaZero" templates
on the Plugins page) subclass `AlphaZeroTrainer` and override either just
`build_network()` (swap the architecture, keep self-play/MCTS/arena-gating
as-is) or the whole `run_iteration()` (replace everything). The shared
iterate/checkpoint/metrics-snapshot driver lives in rl_core/alphazero/loop.py
and works with any subclass.
"""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_core.alphazero.arena import play_match
from rl_core.alphazero.checkpoint import save_checkpoint
from rl_core.alphazero.network import AlphaZeroNet
from rl_core.alphazero.self_play import play_self_play_game
from rl_core.games.base import BoardGame
from rl_core.netbuilder import SpecAlphaZeroNet

DEFAULT_HYPERPARAMS = {
    "num_simulations": 25,
    "games_per_iteration": 20,
    "epochs": 4,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "buffer_size": 20_000,
    "eval_games": 10,
    "eval_num_simulations": 30,
    "win_rate_threshold": 0.55,
    "c_puct": 1.5,
    "channels": 48,
    "num_blocks": 3,
}


class AlphaZeroTrainer(ABC):
    """Contract a custom AlphaZero implementation must satisfy.

    Required override: `run_iteration(iteration)` -> a dict of extra metrics
    to merge into that iteration's snapshot (`loss`, `win_rate_vs_prev`,
    `board_history`, ... — whatever keys you want surfaced on the Training
    Monitor page). `board_history` (a list of board matrices, one per move)
    lets the Monitor replay a whole game move by move instead of a single
    freeze-frame — `play_self_play_game` already returns one per record.
    Include a `"_self_play_records"` key (a list of move-history dicts) if
    you want games viewable on the AlphaZero Arena replay page — the driver
    strips this key out before writing metrics.json.

    Optional override: `build_network()` if you only want to swap the
    network architecture and keep self-play/MCTS/arena-gating as-is — see
    `BuiltinAlphaZeroTrainer.run_iteration` for the reference loop you can
    copy and adapt for a full custom implementation.
    """

    def __init__(self, game_cls: type[BoardGame], hyperparams: dict[str, Any], device: str) -> None:
        self.game_cls = game_cls
        self.hp = {**DEFAULT_HYPERPARAMS, **hyperparams}
        self.device = device
        self.sample_game = game_cls()
        self.net = self.build_network()
        self.net.to(device)

    def build_network(self) -> nn.Module:
        """Default: the built-in small dual-head conv net, unless
        `hyperparams["network_spec"]` was set (a hand-designed architecture
        picked in the Designer, resolved by `rl_core.netbuilder_store` before
        the trainer is constructed) — then a `SpecAlphaZeroNet` built from
        that spec is used instead. Override to plug in a different
        architecture entirely — must expose `.rows`, `.cols`, `.action_size`
        attributes (for checkpointing) and a
        `predict(encoded_state, device) -> (policy_probs, value)` method
        (used by MCTS)."""
        g = self.sample_game
        network_spec = self.hp.get("network_spec")
        if network_spec:
            return SpecAlphaZeroNet(g.rows, g.cols, g.action_size, network_spec)
        return AlphaZeroNet(g.rows, g.cols, g.action_size, self.hp["channels"], self.hp["num_blocks"])

    @abstractmethod
    def run_iteration(self, iteration: int) -> dict[str, Any]: ...

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
            "input_planes": 3,
        }
        if self.hp.get("network_spec"):
            info["network_spec"] = self.hp["network_spec"]
        return info


class BuiltinAlphaZeroTrainer(AlphaZeroTrainer):
    """Reference self-play -> train -> arena-gate loop — exactly what
    Reinforcement Studio ships by default. Kept as a normal (subclassable)
    class rather than inline procedural code so custom trainers can reuse
    self-play/MCTS/arena and override only one piece instead of starting
    from zero."""

    def __init__(self, game_cls: type[BoardGame], hyperparams: dict[str, Any], device: str) -> None:
        super().__init__(game_cls, hyperparams, device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hp["learning_rate"])
        self.replay_buffer: deque = deque(maxlen=self.hp["buffer_size"])
        self.best_state = copy.deepcopy(self.net.state_dict())

    def _train_epochs(self, examples: list[tuple[np.ndarray, np.ndarray, float]]) -> float:
        net, optimizer, hp, device = self.net, self.optimizer, self.hp, self.device
        net.train()
        states = torch.as_tensor(np.stack([e[0] for e in examples]), dtype=torch.float32, device=device)
        policies = torch.as_tensor(np.stack([e[1] for e in examples]), dtype=torch.float32, device=device)
        values = torch.as_tensor(np.array([e[2] for e in examples]), dtype=torch.float32, device=device)

        n = len(examples)
        total_loss = 0.0
        num_batches = 0
        for _ in range(hp["epochs"]):
            perm = torch.randperm(n)
            for start in range(0, n, hp["batch_size"]):
                idx = perm[start : start + hp["batch_size"]]
                if len(idx) < 2:
                    continue
                logits, value_pred = net(states[idx])
                policy_loss = -(policies[idx] * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                value_loss = F.mse_loss(value_pred, values[idx])
                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
        return total_loss / max(num_batches, 1)

    def run_iteration(self, iteration: int) -> dict[str, Any]:
        self.net.load_state_dict(self.best_state)
        self.net.eval()

        iteration_records = []
        for g in range(self.hp["games_per_iteration"]):
            examples, record = play_self_play_game(
                self.game_cls, self.net,
                num_simulations=self.hp["num_simulations"], c_puct=self.hp["c_puct"], device=self.device,
            )
            self.replay_buffer.extend(examples)
            if g < 3:
                iteration_records.append(record)

        loss = 0.0
        if len(self.replay_buffer) >= self.hp["batch_size"]:
            loss = self._train_epochs(list(self.replay_buffer))

        prev_net = self.build_network()
        prev_net.load_state_dict(self.best_state)
        prev_net.to(self.device)
        prev_net.eval()

        match = play_match(
            self.game_cls, self.net, prev_net,
            num_simulations=self.hp["eval_num_simulations"], num_games=self.hp["eval_games"], device=self.device,
        )
        decisive = match["wins_a"] + match["wins_b"]
        win_rate = match["wins_a"] / decisive if decisive > 0 else 0.5
        accepted = win_rate >= self.hp["win_rate_threshold"] or decisive == 0

        if accepted:
            self.best_state = copy.deepcopy(self.net.state_dict())
        else:
            self.net.load_state_dict(self.best_state)

        last_game = iteration_records[-1] if iteration_records else None
        return {
            "loss": round(loss, 4),
            "win_rate_vs_prev": round(win_rate, 3),
            "accepted": accepted,
            "arena": match,
            # Full move-by-move board history of one game from this
            # iteration (not just the final position) — the Training
            # Monitor replays it move by move instead of showing a single
            # disconnected freeze-frame per iteration.
            "board_history": last_game["board_history"] if last_game else None,
            "board_winner": last_game["winner"] if last_game else None,
            "buffer_size": len(self.replay_buffer),
            "_self_play_records": iteration_records,
        }
