"""Tests for Scene Builder environments."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

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


# --------------------------------------------------------------------------
# MARL: multi-team scenes (predator/prey, team battle) + Independent PPO.
# --------------------------------------------------------------------------


def test_team_count_and_templates():
    assert scene_store.team_count(scene_store.default_spec()) == 1
    predator_prey = scene_store.default_predator_prey_spec()
    assert scene_store.team_count(predator_prey) == 2
    assert scene_store.agent_count(predator_prey) == 4
    team_battle = scene_store.default_team_battle_spec()
    assert scene_store.team_count(team_battle) == 2
    assert scene_store.agent_count(team_battle) == 4


def test_multi_team_env_assigns_team_ids_and_roles_per_lane():
    spec = scene_store.default_predator_prey_spec()
    env = SceneMultiAgentEnv(spec)
    assert env.num_envs == 4
    # Predator group ("count": 1) occupies lane 0; prey group ("count": 3)
    # occupies the rest, in the same order `spec["agents"]` lists them.
    assert env.team_ids == ["predator", "prey", "prey", "prey"]
    assert env.roles == ["predator", "prey", "prey", "prey"]
    env.close()


def test_sensor_readout_distinguishes_teammate_from_opponent():
    spec = scene_store.default_team_battle_spec()
    env = SceneMultiAgentEnv(spec)
    env.reset(seed=0)
    # Force known positions: lane 0/1 = red team, lane 2/3 = blue team.
    env._positions[:] = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    entities = env._entities_for_sensors(0)
    kinds = {round(x, 1): kind for x, _z, kind in entities if kind in (3.0, 4.0)}
    assert kinds[1.0] == 3.0  # lane 1 is a teammate (same team "red")
    assert kinds[2.0] == 4.0  # lane 2 is an opponent (team "blue")
    assert kinds[3.0] == 4.0
    env.close()


def test_tag_rule_rewards_predator_and_penalizes_prey_on_catch():
    spec = scene_store.default_predator_prey_spec()
    env = SceneMultiAgentEnv(spec)
    env.reset(seed=0)
    # Put the predator (lane 0) right on top of one prey (lane 1), and the
    # other two prey (lanes 2/3) far away, so the very next
    # `_apply_tag_rule()` call catches exactly one prey.
    env._positions[0] = (0.0, 0.0)
    env._positions[1] = (0.05, 0.0)
    env._positions[2] = (-10.0, -10.0)
    env._positions[3] = (10.0, 10.0)
    env._rewards[:] = 0.0
    env._terminations[:] = False
    env._apply_tag_rule()
    assert env._rewards[0] == 2.0  # predator_reward from the template
    assert env._rewards[1] == -2.0  # prey_reward
    env.close()


def test_team_shared_reward_pools_within_team():
    spec = scene_store.default_team_battle_spec()
    env = SceneMultiAgentEnv(spec)
    env.reset(seed=0)
    env._rewards[:] = [1.0, 0.0, 0.0, 2.0]  # lanes 0/1 = red, 2/3 = blue
    env._apply_team_shared_reward()
    assert list(env._rewards) == [1.0, 1.0, 2.0, 2.0]
    env.close()


def test_restrict_team_item_only_pays_out_for_its_team():
    spec = scene_store.default_predator_prey_spec()
    env = SceneMultiAgentEnv(spec)
    env.reset(seed=0)
    # All coins in this template are `restrict_team: "prey"` — the
    # predator (lane 0, team "predator") standing on one should not
    # collect it.
    coin = spec["items"][0]
    env._positions[0] = (coin["position"][0], coin["position"][2])
    env._step_agent(0, 0)
    assert env._item_active[0]  # still active — predator can't collect it
    assert env._rewards[0] == 0.0

    env._positions[1] = (coin["position"][0], coin["position"][2])
    env._step_agent(1, 0)
    assert not env._item_active[0]  # prey (lane 1) collects it fine
    assert env._rewards[1] > 0.0
    env.close()


def test_multi_agent_ppo_requires_multi_team_env():
    from rl_core.algorithms.native.marl_ppo import DEFAULT_HYPERPARAMS, MultiAgentPPO

    spec = scene_store.default_spec()  # single team
    env = SceneMultiAgentEnv(spec)
    try:
        with pytest.raises(ValueError):
            MultiAgentPPO(env, DEFAULT_HYPERPARAMS, seed=0, device="cpu")
    finally:
        env.close()


def test_multi_agent_ppo_learns_and_saves_roundtrip():
    from rl_core.algorithms.base import TrainingCallback
    from rl_core.algorithms.native.marl_ppo import DEFAULT_HYPERPARAMS, MultiAgentPPO

    spec = scene_store.default_predator_prey_spec()
    env = SceneMultiAgentEnv(spec)
    hp = {**DEFAULT_HYPERPARAMS, "n_steps": 32, "batch_size": 16, "n_epochs": 2}
    algo = MultiAgentPPO(env, hp, seed=0, device="cpu")
    assert set(algo.teams) == {"predator", "prey"}

    steps: list[int] = []

    def writer(num_timesteps, episode_reward=None, episode_length=None, metrics=None):
        steps.append(num_timesteps)
        return num_timesteps < 200

    algo.learn(total_timesteps=200, callback=TrainingCallback(writer))
    assert steps[-1] >= 200

    obs, _ = env.reset(seed=1)
    action, _ = algo.predict(obs[0], deterministic=True)
    assert env.single_action_space.contains(action)

    with tempfile.TemporaryDirectory() as d:
        model_path = Path(d) / "model.zip"
        algo.save(model_path)
        loaded = MultiAgentPPO.load(model_path, env, device="cpu")
        assert set(loaded.teams) == {"predator", "prey"}
    env.close()


def test_ippo_end_to_end_via_native_runner_on_team_battle():
    from rl_core.algorithms.native_runner import run

    spec = scene_store.default_team_battle_spec()
    slug = "pytest_marl_train"
    scene_store.save(slug, spec)
    config = {
        "kind": "gym",
        "name": "pytest-ippo",
        "environment": {"id": scene_store.scene_env_id(slug), "wrappers": []},
        "algorithm": {"id": "ippo", "hyperparams": {"n_steps": 32, "batch_size": 16, "n_epochs": 1}},
        "training": {"total_timesteps": 128, "seed": 0},
    }
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        run(config, run_dir)
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["status"] == "completed"
        assert metrics["step"] >= 128
    scene_store.delete(slug)


def test_scene_registry_offers_ippo_only_for_multi_team_scenes():
    from rl_core.envs import registry

    single = scene_store.default_spec()
    multi = scene_store.default_predator_prey_spec()
    scene_store.save("pytest_single_team", single)
    scene_store.save("pytest_multi_team", multi)
    try:
        envs = {e["id"]: e for e in registry.list_environments()}
        single_entry = envs[scene_store.scene_env_id("pytest_single_team")]
        multi_entry = envs[scene_store.scene_env_id("pytest_multi_team")]
        assert "ippo" not in single_entry["compatible_algorithms"]
        assert "ippo" in multi_entry["compatible_algorithms"]
        assert single_entry["scene_team_count"] == 1
        assert multi_entry["scene_team_count"] == 2
    finally:
        scene_store.delete("pytest_single_team")
        scene_store.delete("pytest_multi_team")
