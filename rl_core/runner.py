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


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python -m rl_core.runner <config.json> <run_dir>", file=sys.stderr)
        sys.exit(1)

    config_path = Path(sys.argv[1])
    run_dir = Path(sys.argv[2])
    run_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_path.read_text())
    kind = config.get("kind", "gym")

    try:
        if kind == "alphazero":
            from rl_core.alphazero import train

            train.run(config, run_dir)
        else:
            from rl_core.algorithms import sb3_runner

            sb3_runner.run(config, run_dir)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the run dir
        (run_dir / "metrics.json").write_text(
            json.dumps({"run_id": run_dir.name, "status": "failed", "error": str(exc)})
        )
        (run_dir / "error.log").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
