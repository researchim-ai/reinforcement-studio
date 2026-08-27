"""Tests for Scene Builder environments."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from rl_core.algorithms.vec_env import num_envs_of, vec_reset, vec_step
from rl_core.envs.scene_env import SceneMultiAgentEnv, SceneRenderEnv
from rl_core import scene_store


def test_default_spec_roundtrip():
    spec = scene_store.default_spec()
    slug = "pytest_scene"
    scene_store.save(slug, spec)
    loaded = scene_store.load(slug)
    assert loaded["name"] == spec["name"]
    assert scene_store.agent_count(loaded) == 2
    scene_store.delete(slug)


def test_vector_env_step_and_render():
    spec = scene_store.default_spec()
    env = SceneMultiAgentEnv(spec, render_mode="rgb_array")
    assert num_envs_of(env) == 2
    obs, _ = env.reset(seed=0)
    assert obs.shape == (2, env.single_observation_space.shape[0])
    obs2, rewards, term, trunc, _ = env.step(np.array([1, 2], dtype=np.int64))
    assert obs2.shape == obs.shape
    assert rewards.shape == (2,)
    frame = env.render()
    assert frame is not None
    assert frame.shape == (256, 256, 3)
    env.close()


def test_render_env_single_agent_gif_path():
    spec = scene_store.default_spec()
    env = SceneRenderEnv(spec)
    obs, _ = env.reset(seed=1)
    assert obs.ndim == 1
    obs, r, term, trunc, _ = env.step(1)
    assert isinstance(r, float)
    frame = env.render()
    assert frame is not None
    env.close()


def test_ppo_smoke_on_scene():
    from rl_core.algorithms.native_runner import run

    spec = scene_store.default_spec()
    slug = "pytest_train"
    scene_store.save(slug, spec)
    config = {
        "kind": "gym",
        "name": "pytest",
        "environment": {"id": scene_store.scene_env_id(slug), "wrappers": []},
        "algorithm": {"id": "dqn", "hyperparams": {"learning_rate": 1e-3, "buffer_size": 500}},
        "training": {"total_timesteps": 128, "seed": 0},
    }
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        run(config, run_dir)
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["status"] == "completed"
        assert metrics["step"] >= 128
    scene_store.delete(slug)
