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
# Saved architectures from the visual Network Architecture Builder
# (/network-builder) — plain JSON (a layer spec), not code, since the actual
# nn.Module only gets built at train time by rl_core/netbuilder.py.
CUSTOM_NETWORKS_DIR = ROOT / "custom_networks"
# User-designed 3D RL scenes from the Scene Builder page (/scene-builder) —
# JSON specs compiled into a `SceneMultiAgentEnv` at train time by
# rl_core/envs/scene_env.py (not code plugins).
CUSTOM_SCENES_DIR = ROOT / "custom_scenes"
# Saved World Model specs from the World Model Builder page
# (/world-models) — a JSON spec (type + hyperparams) plus, once trained
# (standalone via `kind: "world_model"` runs, or attached from a finished
# run's checkpoint), a `<slug>/model.pt` weights file living right next to
# it. See rl_core/world_models/store.py. Named `custom_world_models`
# (not `world_models`, unlike every other `CUSTOM_*_DIR` here using its
# bare feature name) specifically to avoid colliding with the
# `rl_core/world_models/` *code* package sitting right next to `paths.py`
# — `ROOT` already equals `PACKAGE_DIR` by default, so `ROOT / "world_models"`
# would otherwise resolve to that package's own on-disk directory and start
# writing run data straight into it.
CUSTOM_WORLD_MODELS_DIR = ROOT / "custom_world_models"

for _d in (
    RUNS_DIR, CHECKPOINTS_DIR, CUSTOM_GYM_ALGOS_DIR, CUSTOM_ALPHAZERO_ALGOS_DIR, CUSTOM_REWARDS_DIR,
    CUSTOM_NETWORKS_DIR, CUSTOM_SCENES_DIR, CUSTOM_WORLD_MODELS_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)


def run_dir(run_id: str) -> Path:
    d = RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d
