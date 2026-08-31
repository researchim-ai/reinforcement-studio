"""Tests for the NetHack/MiniHack addition to the Environments gallery (see
rl_core/envs/nethack_envs.py, registry.py). Skipped whole-file if neither
`nle` nor `minihack` is installed — matches test_new_environments.py's
skip-if-extra-missing convention, since these packages only ship prebuilt
wheels for Linux and aren't in the default requirements.txt."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import gymnasium as gym
import pytest

from rl_core.envs import registry
from rl_core.envs.nethack_envs import _MINIHACK_BASE_IDS, _NETHACK_BASE_IDS


def _has(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not (_has("nle") or _has("minihack")),
    reason="nle/minihack not installed",
)


def test_new_envs_listed():
    envs = {e["id"]: e for e in registry.list_environments()}
    assert "NetHackScore-Img-v0" in envs
    assert "MiniHack-Room-5x5-Img-v0" in envs
    assert envs["MiniHack-Room-5x5-Img-v0"]["action_kind"] == "discrete"
    if _has("nle"):
        assert envs["NetHackScore-Img-v0"]["available"] is True
    if _has("minihack"):
        assert envs["MiniHack-Room-5x5-Img-v0"]["available"] is True


@pytest.mark.skipif(not _has("nle"), reason="nle not installed")
def test_nethack_wrapped_obs_is_plain_box_image():
    env = gym.make("NetHackScore-Img-v0")
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == (21, 79, 1)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (21, 79, 1)
    assert obs.dtype == env.observation_space.dtype
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (21, 79, 1)
    env.close()


@pytest.mark.skipif(not _has("minihack"), reason="minihack not installed")
def test_minihack_wrapped_obs_uses_egocentric_crop():
    env = gym.make("MiniHack-Room-5x5-Img-v0")
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == (9, 9, 1)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (9, 9, 1)
    env.close()


@pytest.mark.skipif(not _has("nle"), reason="nle not installed")
@pytest.mark.parametrize("img_id", [img_id for _base_id, img_id, _key in _NETHACK_BASE_IDS])
def test_every_registered_nethack_id_is_constructible(img_id):
    # Regression test: MiniHack-KeyRoom-5x5-Img-v0 silently pointed at a
    # non-existent base id (`MiniHack-KeyRoom-5x5-v0` instead of the real
    # `MiniHack-KeyRoom-S5-v0`) for a while — `gym.make()` raised
    # `NameNotFound` on *every* use (training, live GIF preview, gallery
    # preview), but nothing exercised that specific id, so it went
    # unnoticed. This parametrizes over the actual `_NETHACK_BASE_IDS`
    # table so a typo'd base id anywhere in it fails loudly instead.
    env = gym.make(img_id)
    env.reset(seed=0)
    env.step(env.action_space.sample())
    env.close()


@pytest.mark.skipif(not _has("minihack"), reason="minihack not installed")
@pytest.mark.parametrize("img_id", [img_id for _base_id, img_id, _key in _MINIHACK_BASE_IDS])
def test_every_registered_minihack_id_is_constructible(img_id):
    # Same rationale as test_every_registered_nethack_id_is_constructible
    # above — this is the test that would have (and now does) catch the
    # MiniHack-KeyRoom-5x5-Img-v0 base-id typo.
    env = gym.make(img_id)
    env.reset(seed=0)
    env.step(env.action_space.sample())
    env.close()


@pytest.mark.skipif(not _has("minihack"), reason="minihack not installed")
def test_ppo_smoke_on_minihack():
    from rl_core.algorithms.native_runner import run

    config = {
        "kind": "gym",
        "environment": {"id": "MiniHack-Room-5x5-Img-v0", "wrappers": []},
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
