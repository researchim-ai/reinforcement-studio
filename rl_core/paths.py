"""Shared path helpers for native and Docker backends."""
from __future__ import annotations

import os
from pathlib import Path


def get_root(default: Path) -> Path:
    """Return the data root (runs/checkpoints), overridable via RL_STUDIO_ROOT
    (used e.g. inside Docker to point at a mounted volume). This must stay
    independent from PACKAGE_DIR, which is always the on-disk location of the
    rl_core package and is what subprocesses need to import it as a module."""
    override = os.environ.get("RL_STUDIO_ROOT")
    return Path(override).resolve() if override else default.resolve()


PACKAGE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = PACKAGE_DIR.parent

ROOT = get_root(PACKAGE_DIR)
RUNS_DIR = ROOT / ".runs"
CHECKPOINTS_DIR = ROOT / "checkpoints"

RUNS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def run_dir(run_id: str) -> Path:
    d = RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d
