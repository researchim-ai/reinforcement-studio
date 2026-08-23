"""AlphaZero self-play training loop: self-play -> train -> arena-gate.

Each iteration: play `games_per_iteration` self-play games with the current
best network, train a candidate on the replay buffer, then pit the
candidate against the previous best over `eval_games` games. The candidate
only becomes the new "best" (and gets checkpointed) if it wins clearly
enough — this arena-gating step is what keeps AlphaZero training stable.
"""
from __future__ import annotations

import copy
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from rl_core.alphazero.arena import play_match
from rl_core.alphazero.checkpoint import save_checkpoint
from rl_core.alphazero.network import AlphaZeroNet
from rl_core.alphazero.self_play import play_self_play_game
from rl_core.games import make_game

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


def _train_epochs(
    net: AlphaZeroNet,
    optimizer: torch.optim.Optimizer,
    examples: list[tuple[np.ndarray, np.ndarray, float]],
    epochs: int,
    batch_size: int,
    device: str,
) -> float:
    net.train()
    states = torch.as_tensor(np.stack([e[0] for e in examples]), dtype=torch.float32, device=device)
    policies = torch.as_tensor(np.stack([e[1] for e in examples]), dtype=torch.float32, device=device)
    values = torch.as_tensor(np.array([e[2] for e in examples]), dtype=torch.float32, device=device)

    n = len(examples)
    total_loss = 0.0
    num_batches = 0
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
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


def run(config: dict[str, Any], run_dir: Path) -> None:
    env_cfg = config.get("environment", {})
    algo_cfg = config.get("algorithm", {})
    training_cfg = config.get("training", {})

    game_id = env_cfg["id"]
    hp = {**DEFAULT_HYPERPARAMS, **(algo_cfg.get("hyperparams") or {})}
    num_iterations = int(training_cfg.get("num_iterations", 20))
    seed = training_cfg.get("seed")
    device = "cuda" if torch.cuda.is_available() and training_cfg.get("use_gpu", False) else "cpu"

    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

    game_cls = make_game(game_id).__class__
    sample_game = game_cls()

    net = AlphaZeroNet(sample_game.rows, sample_game.cols, sample_game.action_size, hp["channels"], hp["num_blocks"])
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=hp["learning_rate"])

    replay_buffer: deque = deque(maxlen=hp["buffer_size"])
    best_state = copy.deepcopy(net.state_dict())

    games_dir = run_dir / "games"
    games_dir.mkdir(exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    start_time = time.time()

    def stop_requested() -> bool:
        return (run_dir / "stop.flag").exists()

    def write_snapshot(iteration: int, status: str, extra: dict | None = None) -> None:
        snapshot = {
            "run_id": run_dir.name,
            "kind": "alphazero",
            "status": status,
            "algo": "alphazero",
            "env_id": game_id,
            "step": iteration,
            "total_timesteps": num_iterations,
            "elapsed_seconds": round(time.time() - start_time, 1),
            "buffer_size": len(replay_buffer),
        }
        if extra:
            snapshot.update(extra)
        (run_dir / "metrics.json").write_text(json.dumps(snapshot))

    write_snapshot(0, "running")
    final_status = "completed"

    for iteration in range(1, num_iterations + 1):
        if stop_requested():
            final_status = "stopped"
            break

        net.load_state_dict(best_state)
        net.eval()

        iteration_records = []
        for g in range(hp["games_per_iteration"]):
            examples, record = play_self_play_game(
                game_cls, net, num_simulations=hp["num_simulations"], c_puct=hp["c_puct"], device=device
            )
            replay_buffer.extend(examples)
            if g < 3:
                iteration_records.append(record)

        (games_dir / f"iteration_{iteration:04d}.json").write_text(json.dumps(iteration_records))

        loss = 0.0
        if len(replay_buffer) >= hp["batch_size"]:
            loss = _train_epochs(
                net, optimizer, list(replay_buffer), hp["epochs"], hp["batch_size"], device
            )

        prev_net = AlphaZeroNet(sample_game.rows, sample_game.cols, sample_game.action_size, hp["channels"], hp["num_blocks"])
        prev_net.load_state_dict(best_state)
        prev_net.to(device)
        prev_net.eval()

        match = play_match(
            game_cls, net, prev_net,
            num_simulations=hp["eval_num_simulations"],
            num_games=hp["eval_games"],
            device=device,
        )
        decisive = match["wins_a"] + match["wins_b"]
        win_rate = match["wins_a"] / decisive if decisive > 0 else 0.5
        accepted = win_rate >= hp["win_rate_threshold"] or decisive == 0

        if accepted:
            best_state = copy.deepcopy(net.state_dict())
            save_checkpoint(
                net, run_dir / "model.pt",
                {"game_id": game_id, "iteration": iteration, "win_rate": win_rate},
            )
        else:
            net.load_state_dict(best_state)

        write_snapshot(
            iteration,
            "running",
            {
                "loss": round(loss, 4),
                "win_rate_vs_prev": round(win_rate, 3),
                "accepted": accepted,
                "arena": match,
            },
        )

    net.load_state_dict(best_state)
    save_checkpoint(net, run_dir / "model.pt", {"game_id": game_id, "iteration": num_iterations})
    write_snapshot(num_iterations, final_status)
