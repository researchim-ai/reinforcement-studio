"""Tests for the built-in Tuple-space MARL benchmarks — RWARE, LBForaging,
and SMAClite (`marlgym:{slug}` env ids — see
`rl_core/envs/tuple_marl_envs.py`'s module docstring for why these three)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rl_core.algorithms.vec_env import num_envs_of, vec_reset, vec_step
from rl_core.envs.factory import is_shared_world_env_id, make_inspect_env, make_training_env
from rl_core.envs.tuple_marl_envs import (
    TupleMarlRenderEnv,
    TupleMarlVectorEnv,
    _SPECS,
    available,
    is_tuple_marl_env_id,
    list_meta,
    tuple_marl_env_id,
    tuple_marl_slug_from_env_id,
)

# smaclite is an optional extra (`extra_requirement="smaclite"` — a
# 2-step `pip install` given its transitive `Rtree==1.0.0` pin has no
# cp311+ wheel, see `EXTRA_HINT` in `src/pages/Environments.tsx`), unlike
# rware/lbforaging which this app's `requirements.txt` always installs —
# so, unlike every other slug here, `smac_*` tests must degrade to a skip
# rather than a hard failure when it isn't present in a given dev/CI env.
_SMAC_SLUGS = [s for s in _SPECS if s.startswith("smac_")]
_NON_SMAC_SLUGS = [s for s in _SPECS if not s.startswith("smac_")]
_smaclite_installed = all(available(s) for s in _SMAC_SLUGS)
requires_smaclite = pytest.mark.skipif(not _smaclite_installed, reason="smaclite not installed (optional extra)")


def test_env_id_helpers():
    assert is_tuple_marl_env_id("marlgym:rware_tiny_2ag")
    assert not is_tuple_marl_env_id("petting:pursuit")
    assert not is_tuple_marl_env_id("scene:foo")
    assert not is_tuple_marl_env_id("CartPole-v1")
    assert tuple_marl_slug_from_env_id("marlgym:lbforaging_8x8_2p") == "lbforaging_8x8_2p"
    assert tuple_marl_env_id("lbforaging_8x8_2p") == "marlgym:lbforaging_8x8_2p"
    assert is_shared_world_env_id("marlgym:rware_tiny_2ag")
    assert not is_shared_world_env_id("CartPole-v1")


def test_list_meta_has_all_seven_envs():
    slugs = {m["slug"] for m in list_meta()}
    assert slugs == {
        "rware_tiny_2ag", "rware_small_4ag", "lbforaging_8x8_2p", "lbforaging_10x10_4p",
        "smac_2s3z", "smac_3s_vs_5z", "smac_mmm2",
    }
    for m in list_meta():
        if m["slug"] in _SMAC_SLUGS:
            assert m["mask_aware"]
            assert m["extra_requirement"] == "smaclite"
            if not m["available"]:
                continue  # optional extra — see `requires_smaclite` above
        else:
            assert not m["mask_aware"]
            assert m["extra_requirement"] == "rware/lbforaging"
            assert m["available"], f"{m['slug']} should be importable/available in the test env"
        assert m["agent_count"] >= 2
        assert m["team_count"] == 1  # every benchmark here is single-team by design


@pytest.mark.parametrize("slug", _NON_SMAC_SLUGS)
def test_vector_env_step_and_reset(slug):
    spec = _SPECS[slug]
    env = TupleMarlVectorEnv(spec)
    n = env.num_envs
    assert n == len(env.team_ids)
    assert len(set(env.team_ids)) == 1  # single cooperative team
    obs, _info = env.reset(seed=0)
    assert obs.shape[0] == n
    assert obs.shape[1:] == env.single_observation_space.shape

    actions = np.array([env.single_action_space.sample() for _ in range(n)])
    obs2, rewards, term, trunc, _info = env.step(actions)
    assert obs2.shape == obs.shape
    assert rewards.shape == (n,)
    assert term.shape == (n,)
    assert trunc.shape == (n,)
    # Episode end is shared team-wide (not per-agent) — every lane must
    # agree on terminated/truncated.
    assert len(set(term.tolist())) == 1
    assert len(set(trunc.tolist())) == 1
    env.close()


@requires_smaclite
@pytest.mark.parametrize("slug", _SMAC_SLUGS)
def test_smac_vector_env_exposes_and_enforces_action_mask(slug):
    """SMAClite-specific: `get_avail_actions()` must be present and must
    actually gate what `step()` accepts — passing every lane's action as
    `0` (near-guaranteed illegal for at least one lane, since `0` is
    "no-op", never valid the instant any teammate is still alive/acting)
    must not raise, unlike calling the raw `smaclite` env directly with an
    illegal action (see module docstring)."""
    spec = _SPECS[slug]
    env = TupleMarlVectorEnv(spec)
    n = env.num_envs
    obs, _info = env.reset(seed=0)
    mask = env.get_avail_actions()
    assert mask is not None
    assert mask.shape == (n, env.single_action_space.n)
    assert mask.dtype == np.bool_
    assert np.all(mask.sum(axis=-1) >= 1)  # every lane always has >=1 legal action

    # Deliberately feed an all-zeros action vector — must be silently
    # clamped to a legal action per lane, never raise `ValueError` the way
    # calling `smaclite`'s own `.step()` directly would (see module
    # docstring's "situational action space" section).
    bogus_actions = np.zeros(n, dtype=np.int64)
    obs2, rewards, term, trunc, _info = env.step(bogus_actions)
    assert obs2.shape == obs.shape
    env.close()


def test_non_smac_tuple_marl_envs_have_no_action_mask():
    """RWARE/LBForaging's action spaces are always fully valid — `get_
    avail_actions()` must return `None`, not e.g. an all-True array (the
    absence is itself the signal `qmix` checks — see its module
    docstring's "Action-masking note")."""
    env = TupleMarlVectorEnv(_SPECS["rware_tiny_2ag"])
    env.reset(seed=0)
    assert env.get_avail_actions() is None
    env.close()


def test_vec_env_helpers_recognize_tuple_marl_vector_env():
    env = TupleMarlVectorEnv(_SPECS["rware_tiny_2ag"])
    assert num_envs_of(env) == env.num_envs
    obs_list = vec_reset(env, seed=0)
    assert len(obs_list) == env.num_envs
    actions = [env.single_action_space.sample() for _ in range(env.num_envs)]
    obs_list2, rewards, term, trunc, _infos = vec_step(env, actions)
    assert len(obs_list2) == len(obs_list)
    assert rewards.shape == (env.num_envs,)
    env.close()


def test_render_env_controls_lane_zero_and_never_crashes_headless():
    """Rendering RWARE/LBForaging needs a real display (pyglet GL
    context) — headless test environments must not crash, only silently
    skip the preview (see `TupleMarlVectorEnv.render`)."""
    env = TupleMarlRenderEnv(_SPECS["lbforaging_8x8_2p"], render_mode="rgb_array")
    obs, _info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    obs2, reward, term, trunc, _info = env.step(env.action_space.sample())
    assert obs2.shape == obs.shape
    assert isinstance(reward, float)
    frame = env.render()  # None (no display) or an (H, W, 3) array — both fine
    assert frame is None or frame.ndim == 3
    env.close()


def test_make_training_env_and_make_inspect_env_route_marlgym_ids():
    env = make_training_env("marlgym:rware_tiny_2ag")
    assert isinstance(env, TupleMarlVectorEnv)
    env.close()

    render_env = make_training_env("marlgym:rware_tiny_2ag", render=True)
    assert isinstance(render_env, TupleMarlRenderEnv)
    render_env.close()

    inspect_env = make_inspect_env("marlgym:rware_tiny_2ag")
    assert isinstance(inspect_env, TupleMarlVectorEnv)
    inspect_env.close()


def test_registry_lists_tuple_marl_envs_with_marl_category_and_qmix_only_gating():
    from rl_core.envs.registry import list_environments

    envs = {e["id"]: e for e in list_environments()}
    assert "marlgym:rware_tiny_2ag" in envs
    assert "marlgym:rware_small_4ag" in envs
    assert "marlgym:lbforaging_8x8_2p" in envs
    assert "marlgym:lbforaging_10x10_4p" in envs

    rware = envs["marlgym:rware_tiny_2ag"]
    assert rware["category"] == "marl"
    assert "ippo" not in rware["compatible_algorithms"]  # single team
    assert "qmix" in rware["compatible_algorithms"]  # 2 agents, one team
    assert "dqn" in rware["compatible_algorithms"]
    assert "ppo" in rware["compatible_algorithms"]

    assert "marlgym:smac_2s3z" in envs
    assert "marlgym:smac_3s_vs_5z" in envs
    assert "marlgym:smac_mmm2" in envs
    for smac_id in ("marlgym:smac_2s3z", "marlgym:smac_3s_vs_5z", "marlgym:smac_mmm2"):
        smac = envs[smac_id]
        assert smac["category"] == "marl"
        # Situational (masked) action space — only `qmix` reads the mask,
        # so it's the *only* compatible algorithm, unlike RWARE/LBForaging
        # above (see `rl_core/algorithms/native/qmix.py`'s module
        # docstring's "Action-masking note").
        assert smac["compatible_algorithms"] == ["qmix"]
        if not smac["available"]:
            assert smac["extra_requirement"] == "smaclite"


def test_qmix_end_to_end_on_rware():
    from rl_core.algorithms.native_runner import run

    config = {
        "kind": "gym",
        "name": "pytest-qmix-rware",
        "environment": {"id": "marlgym:rware_tiny_2ag", "wrappers": []},
        "algorithm": {
            "id": "qmix",
            "hyperparams": {"buffer_size": 200, "batch_size": 8, "learning_starts": 16, "train_freq": 4},
        },
        "training": {"total_timesteps": 48, "seed": 0},
    }
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        run(config, run_dir)
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["status"] == "completed"
        assert metrics["step"] >= 48


@requires_smaclite
def test_qmix_end_to_end_on_smac_masks_without_crashing():
    """The whole point of the masking machinery: a full `qmix` run on a
    `smac_*` scenario must complete without ever hitting `SMACliteEnv.
    step`'s `ValueError` on an illegal action — exercises action selection
    (masked ε-greedy + masked greedy argmax), the `JointReplayBuffer`'s
    `next_action_mask` storage, and `_train_step`'s masked Double-DQN
    target end to end."""
    from rl_core.algorithms.native_runner import run

    config = {
        "kind": "gym",
        "name": "pytest-qmix-smac",
        "environment": {"id": "marlgym:smac_2s3z", "wrappers": []},
        "algorithm": {
            "id": "qmix",
            "hyperparams": {"buffer_size": 200, "batch_size": 8, "learning_starts": 16, "train_freq": 4},
        },
        "training": {"total_timesteps": 48, "seed": 0},
    }
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        run(config, run_dir)
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["status"] == "completed", metrics
        assert metrics["step"] >= 48


def test_dqn_and_ppo_end_to_end_on_lbforaging():
    from rl_core.algorithms.native_runner import run

    for algo, hp in (
        ("dqn", {"buffer_size": 200, "batch_size": 8, "learning_starts": 16, "train_freq": 4}),
        ("ppo", {"n_steps": 16, "batch_size": 8}),
    ):
        config = {
            "kind": "gym",
            "name": f"pytest-{algo}-lbforaging",
            "environment": {"id": "marlgym:lbforaging_8x8_2p", "wrappers": []},
            "algorithm": {"id": algo, "hyperparams": hp},
            "training": {"total_timesteps": 48, "seed": 0},
        }
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d)
            run(config, run_dir)
            metrics = json.loads((run_dir / "metrics.json").read_text())
            assert metrics["status"] == "completed", f"{algo} on LBForaging: {metrics}"
