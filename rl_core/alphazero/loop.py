"""Shared iterate/checkpoint/metrics-snapshot driver for AlphaZero trainers.

Used by both the built-in trainer (rl_core/alphazero/train.py) and custom
trainers loaded from rl_core/plugins (rl_core/alphazero/custom_train.py) —
this is the part of the training loop that has nothing to do with self-play/
MCTS specifically: metrics.json snapshots, stop.flag handling, and
checkpointing the network after each iteration.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from rl_core.alphazero.base import AlphaZeroTrainer


def run_training_loop(
    trainer: AlphaZeroTrainer,
    config: dict[str, Any],
    run_dir: Path,
    checkpoint_meta: dict[str, Any] | None = None,
) -> None:
    training_cfg = config.get("training", {})
    algo_id = config.get("algorithm", {}).get("id", "alphazero")
    game_id = config.get("environment", {}).get("id", "")
    num_iterations = int(training_cfg.get("num_iterations", 20))

    static_info = {
        "hyperparams": trainer.hp,
        "device": trainer.device,
        "total_params": trainer.total_params,
        "seed": training_cfg.get("seed"),
        "network": trainer.network_info,
    }

    games_dir = run_dir / "games"
    games_dir.mkdir(exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    start_time = time.time()

    def stop_requested() -> bool:
        return (run_dir / "stop.flag").exists()

    def write_snapshot(iteration: int, status: str, extra: dict[str, Any] | None = None) -> None:
        snapshot = {
            "run_id": run_dir.name,
            "kind": "alphazero",
            "status": status,
            "algo": algo_id,
            "env_id": game_id,
            "step": iteration,
            "total_timesteps": num_iterations,
            "elapsed_seconds": round(time.time() - start_time, 1),
        }
        snapshot.update(static_info)
        if extra:
            snapshot.update(extra)
        (run_dir / "metrics.json").write_text(json.dumps(snapshot))

    write_snapshot(0, "running")
    final_status = "completed"
    last_extra: dict[str, Any] = {}
    last_completed_iteration = 0

    for iteration in range(1, num_iterations + 1):
        if stop_requested():
            final_status = "stopped"
            break

        extra = dict(trainer.run_iteration(iteration) or {})
        records = extra.pop("_self_play_records", None)
        if records is not None:
            (games_dir / f"iteration_{iteration:04d}.json").write_text(json.dumps(records))

        last_extra = extra
        last_completed_iteration = iteration
        write_snapshot(iteration, "running", last_extra)

        meta = {"game_id": game_id, "iteration": iteration, **(checkpoint_meta or {})}
        if "win_rate_vs_prev" in extra:
            meta["win_rate"] = extra["win_rate_vs_prev"]
        trainer.save(run_dir / "model.pt", meta)

    # Carry the last iteration's extras (loss/win-rate/board/...) forward
    # into the final snapshot — otherwise this last write would blank them
    # out even though training genuinely produced them (metrics.json is the
    # source of truth for the run summary once the process has exited).
    final_meta = {"game_id": game_id, "iteration": last_completed_iteration, **(checkpoint_meta or {})}
    trainer.save(run_dir / "model.pt", final_meta)
    write_snapshot(last_completed_iteration, final_status, last_extra)
