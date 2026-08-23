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

# User-authored plugins (custom algorithms/reward functions edited in-app via
# the Monaco editor on the Plugins page) — one .py file per script, named
# after its slug. Kept outside PACKAGE_DIR so they survive app updates/
# reinstalls, same rationale as RUNS_DIR/CHECKPOINTS_DIR.
CUSTOM_GYM_ALGOS_DIR = ROOT / "custom_algorithms" / "gym"
CUSTOM_ALPHAZERO_ALGOS_DIR = ROOT / "custom_algorithms" / "alphazero"
CUSTOM_REWARDS_DIR = ROOT / "custom_rewards"

for _d in (RUNS_DIR, CHECKPOINTS_DIR, CUSTOM_GYM_ALGOS_DIR, CUSTOM_ALPHAZERO_ALGOS_DIR, CUSTOM_REWARDS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def run_dir(run_id: str) -> Path:
    d = RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d
