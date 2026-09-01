"""Tests for the built-in PettingZoo MARL benchmarks (`petting:{slug}` env
ids — see `rl_core/envs/pettingzoo_envs.py`'s module docstring for why these
4 specifically, instead of e.g. Dota/StarCraft II)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rl_core.algorithms.vec_env import num_envs_of, vec_reset, vec_step
from rl_core.envs.factory import is_shared_world_env_id, make_inspect_env, make_training_env
from rl_core.envs.pettingzoo_envs import (
    PettingZooRenderEnv,
    PettingZooVectorEnv,
    _SPECS,
    is_pettingzoo_env_id,
    list_meta,
    pettingzoo_env_id,
    pettingzoo_slug_from_env_id,
)


def test_env_id_helpers():
    assert is_pettingzoo_env_id("petting:simple_spread")
    assert not is_pettingzoo_env_id("scene:foo")
    assert not is_pettingzoo_env_id("CartPole-v1")
    assert pettingzoo_slug_from_env_id("petting:pursuit") == "pursuit"
    assert pettingzoo_env_id("pursuit") == "petting:pursuit"
    assert is_shared_world_env_id("petting:simple_spread")
    assert is_shared_world_env_id("scene:whatever")
    assert not is_shared_world_env_id("CartPole-v1")


def test_list_meta_has_all_five_envs():
    slugs = {m["slug"] for m in list_meta()}
    assert slugs == {"simple_spread", "simple_adversary", "simple_tag", "pursuit", "knights_archers_zombies"}
    for m in list_meta():
        assert m["available"], f"{m['slug']} should be importable/available in the test env"
        assert m["agent_count"] > 0


@pytest.mark.parametrize("slug", list(_SPECS.keys()))
def test_vector_env_step_and_reset(slug):
    spec = _SPECS[slug]
    env = PettingZooVectorEnv(spec)
    n = env.num_envs
    assert n == len(env.team_ids)
    obs, _info = env.reset(seed=0)
    assert obs.shape[0] == n
    assert obs.shape[1:] == env.single_observation_space.shape

    actions = np.array([env.single_action_space.sample() for _ in range(n)])
    obs2, rewards, term, trunc, _info = env.step(actions)
    assert obs2.shape == obs.shape
    assert rewards.shape == (n,)
    assert term.shape == (n,)
    assert trunc.shape == (n,)
    env.close()


def test_simple_adversary_has_two_teams_and_padded_obs():
    spec = _SPECS["simple_adversary"]
    env = PettingZooVectorEnv(spec)
    assert env.teams == ["adversary", "agent"] or set(env.teams) == {"adversary", "agent"}
    assert len(set(env.team_ids)) == 2
    # Heterogeneous per-role obs sizes get zero-padded to the widest one —
    # exercises the non-homogeneous branch of `_fill_obs`.
    assert env._homogeneous_shape is None
    assert env.single_observation_space.shape == (env._max_obs_dim,)
    env.close()


def test_pursuit_keeps_image_observation_shape():
    """`pursuit`'s per-agent obs is already homogeneous across agents — the
    adapter must keep its native (7, 7, 3) shape rather than flattening it,
    so it still routes through the CNN feature extractor."""
    spec = _SPECS["pursuit"]
    env = PettingZooVectorEnv(spec)
    assert len(env.single_observation_space.shape) == 3
    obs, _ = env.reset(seed=0)
    assert obs.shape == (env.num_envs, *env.single_observation_space.shape)
    env.close()


def test_kaz_has_two_role_teams_and_vector_obs():
    """Knights Archers Zombies: `obs_method="vector"` (the default) gives
    every agent — knight or archer — the same flat (27, 5) egocentric
    ray-cast, not pixels, so this stays on the homogeneous-shape branch
    of `_fill_obs` (like `pursuit`) while still being 2 "teams" by role
    (`archer`/`knight`), unlike `pursuit`'s single team."""
    spec = _SPECS["knights_archers_zombies"]
    env = PettingZooVectorEnv(spec)
    assert set(env.team_ids) == {"archer", "knight"}
    assert env.teams == ["archer", "knight"]
    assert env._homogeneous_shape is not None
    obs, _ = env.reset(seed=0)
    assert obs.shape == (env.num_envs, *env.single_observation_space.shape)
    env.close()


def test_vec_env_helpers_recognize_pettingzoo_vector_env():
    env = PettingZooVectorEnv(_SPECS["simple_spread"])
    assert num_envs_of(env) == env.num_envs
    obs_list = vec_reset(env, seed=0)
    assert len(obs_list) == env.num_envs
    actions = [env.single_action_space.sample() for _ in range(env.num_envs)]
    obs_list2, rewards, term, trunc, _infos = vec_step(env, actions)
    assert len(obs_list2) == len(obs_list)
    assert rewards.shape == (env.num_envs,)
    env.close()


def test_render_env_controls_lane_zero():
    env = PettingZooRenderEnv(_SPECS["simple_tag"], render_mode="rgb_array")
    obs, _info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    obs2, reward, term, trunc, _info = env.step(env.action_space.sample())
    assert obs2.shape == obs.shape
    assert isinstance(reward, float)
    frame = env.render()
    assert frame is not None and frame.ndim == 3
    env.close()


def test_make_training_env_and_make_inspect_env_route_petting_ids():
    env = make_training_env("petting:simple_spread")
    assert isinstance(env, PettingZooVectorEnv)
    env.close()

    render_env = make_training_env("petting:simple_spread", render=True)
    assert isinstance(render_env, PettingZooRenderEnv)
    render_env.close()

    inspect_env = make_inspect_env("petting:simple_spread")
    assert isinstance(inspect_env, PettingZooVectorEnv)
    inspect_env.close()


def test_registry_lists_pettingzoo_envs_with_marl_category_and_algo_gating():
    from rl_core.envs.registry import list_environments

    envs = {e["id"]: e for e in list_environments()}
    assert "petting:simple_spread" in envs
    assert "petting:simple_adversary" in envs
    assert "petting:simple_tag" in envs
    assert "petting:pursuit" in envs
    assert "petting:knights_archers_zombies" in envs

    spread = envs["petting:simple_spread"]
    assert spread["category"] == "marl"
    assert "ippo" not in spread["compatible_algorithms"]  # single team
    assert "qmix" in spread["compatible_algorithms"]  # 1 team, 3 agents — QMIX's own classic case

    tag = envs["petting:simple_tag"]
    assert "ippo" in tag["compatible_algorithms"]
    assert "qmix" in tag["compatible_algorithms"]  # 2 teams, discrete

    kaz = envs["petting:knights_archers_zombies"]
    assert "ippo" in kaz["compatible_algorithms"]  # 2 role-teams
    assert "qmix" in kaz["compatible_algorithms"]


def test_ippo_end_to_end_on_simple_tag():
    from rl_core.algorithms.native_runner import run

    config = {
        "kind": "gym",
        "name": "pytest-ippo-petting",
        "environment": {"id": "petting:simple_tag", "wrappers": []},
        "algorithm": {"id": "ippo", "hyperparams": {"n_steps": 16, "batch_size": 8}},
        "training": {"total_timesteps": 64, "seed": 0},
    }
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        run(config, run_dir)
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["status"] == "completed"
        assert metrics["step"] >= 64


def test_qmix_end_to_end_on_simple_adversary():
    from rl_core.algorithms.native_runner import run

    config = {
        "kind": "gym",
        "name": "pytest-qmix-petting",
        "environment": {"id": "petting:simple_adversary", "wrappers": []},
        "algorithm": {
            "id": "qmix",
            "hyperparams": {"buffer_size": 200, "batch_size": 8, "learning_starts": 16, "train_freq": 4},
        },
        "training": {"total_timesteps": 64, "seed": 0},
    }
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        run(config, run_dir)
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["status"] == "completed"
        assert metrics["step"] >= 64


def test_ippo_and_qmix_end_to_end_on_kaz():
    from rl_core.algorithms.native_runner import run

    for algo, hp in (
        ("ippo", {"n_steps": 16, "batch_size": 8}),
        ("qmix", {"buffer_size": 200, "batch_size": 8, "learning_starts": 16, "train_freq": 4}),
    ):
        config = {
            "kind": "gym",
            "name": f"pytest-{algo}-kaz",
            "environment": {"id": "petting:knights_archers_zombies", "wrappers": []},
            "algorithm": {"id": algo, "hyperparams": hp},
            "training": {"total_timesteps": 48, "seed": 0},
        }
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d)
            run(config, run_dir)
            metrics = json.loads((run_dir / "metrics.json").read_text())
            assert metrics["status"] == "completed", f"{algo} on KAZ: {metrics}"


def test_native_ppo_on_pursuit_image_obs():
    """Sanity check for the one env with an image (not vector) observation
    going through the plain single-policy path (native PPO's CNN branch)."""
    from rl_core.algorithms.native_runner import run

    config = {
        "kind": "gym",
        "name": "pytest-ppo-pursuit",
        "environment": {"id": "petting:pursuit", "wrappers": []},
        "algorithm": {"id": "ppo", "hyperparams": {"n_steps": 16, "batch_size": 8, "n_epochs": 1}},
        "training": {"total_timesteps": 48, "seed": 0},
    }
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        run(config, run_dir)
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["status"] == "completed"
