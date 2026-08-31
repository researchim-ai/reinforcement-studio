"""Tests for the gymnasium-robotics maze addition (see
rl_core/envs/robotics_envs.py, registry.py). Skipped whole-file if
`gymnasium_robotics` isn't installed — matches test_new_environments.py's
skip-if-extra-missing convention."""
from __future__ import annotations

import gymnasium as gym
import pytest

from rl_core.envs import registry


def _has(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has("gymnasium_robotics"), reason="gymnasium_robotics not installed")


def test_maze_envs_listed_and_available():
    envs = {e["id"]: e for e in registry.list_environments()}
    for eid in ["PointMaze-UMaze-Flat-v0", "PointMaze-Medium-Flat-v0", "AntMaze-UMaze-Flat-v0", "AntMaze-Medium-Flat-v0"]:
        assert envs[eid]["available"] is True
        assert envs[eid]["action_kind"] == "continuous"
        assert envs[eid]["category"] == "robotics"


def test_point_maze_flat_obs_has_no_dict_left():
    env = gym.make("PointMaze-UMaze-Flat-v0")
    assert isinstance(env.observation_space, gym.spaces.Box)
    obs, _ = env.reset(seed=0)
    assert obs.ndim == 1
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == env.observation_space.shape
    env.close()


def test_ant_maze_flat_obs_has_no_dict_left():
    env = gym.make("AntMaze-UMaze-Flat-v0")
    assert isinstance(env.observation_space, gym.spaces.Box)
    obs, _ = env.reset(seed=0)
    assert obs.ndim == 1
    env.close()


def test_ppo_smoke_on_point_maze():
    from rl_core.algorithms.native_runner import run
    import json
    import tempfile
    from pathlib import Path

    config = {
        "kind": "gym",
        "environment": {"id": "PointMaze-UMaze-Flat-v0", "wrappers": []},
        "algorithm": {"id": "ppo", "hyperparams": {"n_steps": 32, "batch_size": 16, "n_epochs": 1}},
        "training": {"total_timesteps": 96, "seed": 0},
    }
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        run(config, run_dir)
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["status"] == "completed"
        assert metrics["step"] >= 96
