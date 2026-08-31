"""Tests for the Environments gallery preview pipeline (see
rl_core/envs/previews.py), specifically the "tier 3" local-rollout GIF
fallback used by every env this app registers itself (MiniGrid/Highway-env
wrappers, JobShop/BinPacking/Trading/FinRL-*, ...) that has neither an
official Farama Foundation asset nor a hand-drawn bundled SVG."""
from __future__ import annotations

import pytest

from rl_core.envs import previews


@pytest.fixture(autouse=True)
def _isolated_preview_cache(tmp_path, monkeypatch):
    """Redirects the on-disk preview cache to a throwaway tmp dir and
    resets the negative-result cache — without this, tests would both
    pollute (and be polluted by) the real `rl_core/previews/` cache dir
    other manual runs/the app itself write to."""
    monkeypatch.setattr(previews, "PREVIEW_CACHE_DIR", tmp_path)
    previews._local_render_unsupported.clear()
    yield
    previews._local_render_unsupported.clear()


class TestHasPreview:
    def test_true_for_farama_hosted_env(self):
        assert previews.has_preview("CartPole-v1") is True

    def test_true_for_bundled_svg_env(self):
        assert previews.has_preview("tic_tac_toe") is True

    def test_true_for_custom_env_with_no_dedicated_asset(self):
        # None of this app's own from-scratch/wrapped envs have a Farama or
        # bundled entry — tier 3 (local rollout) is assumed available for
        # them (see module docstring) until proven otherwise.
        for eid in ["JobShop-6x6-v0", "BinPacking-v0", "Trading-Discrete-v0", "FinRL-StockTrading-v0"]:
            assert previews.has_preview(eid) is True, eid

    def test_false_after_a_failed_local_render_attempt(self):
        fake_id = "NotARealEnv-v0"
        assert previews.has_preview(fake_id) is True  # optimistic before ever trying
        assert previews.resolve_preview_file(fake_id) is None  # gym.make() raises -> tier 3 fails
        assert previews.has_preview(fake_id) is False  # now negatively cached


class TestResolvePreviewFileLocalRollout:
    def test_generates_a_real_multi_frame_gif_for_a_fast_custom_env(self):
        # BinPacking-v0 is one of the fastest envs to both step and render
        # in the gallery — a good pick for a real (not mocked) rollout in a
        # unit test.
        path = previews.resolve_preview_file("BinPacking-v0")
        assert path is not None
        assert path.is_file()
        assert path.suffix == ".gif"

        from PIL import Image

        with Image.open(path) as img:
            assert getattr(img, "n_frames", 1) > 1

    def test_second_call_reuses_the_cached_file_instead_of_re_rendering(self):
        first = previews.resolve_preview_file("BinPacking-v0")
        mtime_before = first.stat().st_mtime_ns
        second = previews.resolve_preview_file("BinPacking-v0")
        assert second == first
        assert second.stat().st_mtime_ns == mtime_before

    def test_thumb_is_a_valid_jpeg_distinct_from_the_full_gif(self):
        gif_path = previews.resolve_preview_file("BinPacking-v0")
        thumb_path = previews.resolve_preview_file("BinPacking-v0", thumb=True)
        assert thumb_path is not None
        assert thumb_path != gif_path
        assert thumb_path.suffix == ".jpg"

        from PIL import Image

        with Image.open(thumb_path) as img:
            img.verify()

    def test_short_episode_env_still_produces_a_multi_frame_gif(self):
        # JobShop instances often finish (terminate) well under
        # `_LOCAL_ROLLOUT_MAX_STEPS` on a random policy — the rollout must
        # restart the episode rather than stop dead after the first one.
        path = previews.resolve_preview_file("JobShop-6x6-v0")
        assert path is not None

        from PIL import Image

        with Image.open(path) as img:
            assert getattr(img, "n_frames", 1) >= 10

    def test_unresolvable_env_id_returns_none_without_raising(self):
        assert previews.resolve_preview_file("NotARealEnv-v0") is None
        assert previews.resolve_preview_file("NotARealEnv-v0", thumb=True) is None


class TestSubprocessIsolation:
    def test_a_crashing_render_backend_does_not_take_down_the_caller(self):
        """`PointMaze-UMaze-Flat-v0` (gymnasium_robotics + MuJoCo, wrapped
        in rl_core/envs/robotics_envs.py) is the concrete case that
        motivated running rollouts in a subprocess (see
        `_render_rollout_gif_in_subprocess`'s docstring): on a machine with
        no display for MuJoCo's default GLFW backend to open a window on,
        rendering it segfaults the interpreter rather than raising a
        catchable exception. Whatever the outcome here (`None` if this
        sandbox truly can't render it, a real file if it can), the mere
        fact this test process is still alive to make the assertion is the
        actual regression check — a segfault taking the caller down with
        it would abort the whole pytest run instead of failing one test."""
        pytest.importorskip("gymnasium_robotics")
        result = previews.resolve_preview_file("PointMaze-UMaze-Flat-v0")
        assert result is None or result.is_file()
