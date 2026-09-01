"""Subprocess entrypoint for a single training run.

Usage: python -m rl_core.runner <config.json> <run_dir>

The backend spawns this as a subprocess per run (mirrors how the desktop
app's FastAPI process stays responsive while the actual RL loop, which can
take minutes, runs independently and just writes metrics.json / model
checkpoints to `run_dir` for the API/WebSocket layer to pick up).
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


def _limit_torch_threads() -> None:
    """PyTorch defaults its intra-op thread pool to one thread per CPU
    core — great for the large batched matmuls a vision model does, but
    actively harmful here: every algorithm in this app (native PPO/DQN/...,
    AlphaZero's MCTS, even SB3) spends most of its time doing *lots of
    tiny* forward/backward passes (a 64-unit MLP, a handful of board-game
    conv layers, ...) where the fixed cost of spinning up/synchronizing N
    worker threads dwarfs the few microseconds of actual matmul work being
    parallelized. On an 8+ core machine this alone can make training
    several times slower than single-threaded, and it gets worse still
    once `training.num_envs` also has several `AsyncVectorEnv` worker
    processes competing for the same cores. Set once, as early as
    possible in this subprocess — before any algorithm module (and thus
    torch) has done any real work — since `set_num_interop_threads` can
    only be called once, before torch starts any interop-parallel task."""
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already started parallel work somehow — not worth failing the run over


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python -m rl_core.runner <config.json> <run_dir>", file=sys.stderr)
        sys.exit(1)

    _limit_torch_threads()

    config_path = Path(sys.argv[1])
    run_dir = Path(sys.argv[2])
    run_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_path.read_text())
    kind = config.get("kind", "gym")
    algo_id = config.get("algorithm", {}).get("id", "")

    try:
        if isinstance(algo_id, str) and algo_id.startswith("custom:"):
            slug = algo_id.split(":", 1)[1]
            if kind == "alphazero":
                from rl_core.alphazero import custom_train

                custom_train.run(config, run_dir, slug)
            else:
                from rl_core.algorithms import custom_runner

                custom_runner.run(config, run_dir, slug)
        elif kind == "alphazero":
            from rl_core.alphazero import train

            train.run(config, run_dir)
        elif kind == "world_model":
            # Standalone World Model training (no RL agent at all) — see
            # rl_core/world_models/trainer.py's module docstring. Reuses
            # the exact same StartRunRequest shape as a Gym run:
            # `algorithm.id` is a WORLD_MODEL_TYPES entry instead of a real
            # algorithm, `algorithm.hyperparams` is that type's config.
            from rl_core.world_models import trainer as world_model_trainer

            world_model_trainer.run(config, run_dir)
        else:
            from rl_core.algorithms import native_runner

            native_runner.run(config, run_dir)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the run dir
        (run_dir / "metrics.json").write_text(
            json.dumps({"run_id": run_dir.name, "status": "failed", "error": str(exc)})
        )
        (run_dir / "error.log").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
