"""Tests for the MiniGrid / Highway-env additions to the Environments
gallery (see rl_core/envs/minigrid_envs.py, highway_envs.py, registry.py)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import gymnasium as gym
import pytest

from rl_core.envs import registry


def _has(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not (_has("minigrid") and _has("highway_env")),
    reason="minigrid/highway-env not installed",
)


def test_new_envs_listed_and_available():
    envs = {e["id"]: e for e in registry.list_environments()}
    assert envs["MiniGrid-Empty-8x8-Img-v0"]["available"] is True
    assert envs["MiniGrid-Empty-8x8-Img-v0"]["action_kind"] == "discrete"
    assert envs["highway-v0"]["available"] is True
    assert envs["parking-Flat-v0"]["action_kind"] == "continuous"


def test_minigrid_wrapped_obs_is_plain_box_image():
    env = gym.make("MiniGrid-Empty-8x8-Img-v0")
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == (7, 7, 3)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (7, 7, 3)
    env.close()


def test_highway_kinematics_obs_is_plain_box_vector():
    env = gym.make("highway-v0")
    assert isinstance(env.observation_space, gym.spaces.Box)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    env.close()


def test_parking_flat_obs_has_no_dict_left():
    env = gym.make("parking-Flat-v0")
    assert isinstance(env.observation_space, gym.spaces.Box)
    obs, _ = env.reset(seed=0)
    assert obs.ndim == 1
    env.close()


def test_ppo_smoke_on_minigrid():
    from rl_core.algorithms.native_runner import run

    config = {
        "kind": "gym",
        "environment": {"id": "MiniGrid-Empty-8x8-Img-v0", "wrappers": []},
        "algorithm": {"id": "ppo", "hyperparams": {"n_steps": 32, "batch_size": 16, "n_epochs": 1}},
        "training": {"total_timesteps": 96, "seed": 0},
    }
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        run(config, run_dir)
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["status"] == "completed"
        assert metrics["step"] >= 96
        assert metrics["policy"] == "CnnPolicy"


def test_dqn_smoke_on_highway():
    from rl_core.algorithms.native_runner import run

    config = {
        "kind": "gym",
        "environment": {"id": "highway-v0", "wrappers": []},
        "algorithm": {"id": "dqn", "hyperparams": {"learning_rate": 1e-3, "buffer_size": 500}},
        "training": {"total_timesteps": 96, "seed": 0},
    }
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        run(config, run_dir)
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["status"] == "completed"
        assert metrics["step"] >= 96
