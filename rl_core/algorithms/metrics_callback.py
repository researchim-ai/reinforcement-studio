"""SB3 callback that snapshots training progress to metrics.json.

The FastAPI backend tails this file (see backend/ws.py) and forwards each
new snapshot to the frontend over a WebSocket — same pattern used for the
AlphaZero trainer, so the Training Monitor page can treat both the same way.
"""
from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from rl_core.metrics_history import append_history, json_safe

# SB3 algorithms log per-update training stats under these dotted keys (not
# every algorithm reports every key — DQN has no "entropy_loss", PPO has no
# "loss", etc.) via `model.logger.name_to_value`. Surfacing them generically
# here (rather than hand-picking per-algorithm) means any SB3-based custom
# plugin automatically gets a loss chart in the Training Monitor too.
_SB3_TRAIN_METRIC_KEYS = {
    "train/policy_loss": "policy_loss",
    "train/policy_gradient_loss": "policy_loss",
    "train/value_loss": "value_loss",
    "train/entropy_loss": "entropy_loss",
    "train/loss": "loss",
    "train/approx_kl": "approx_kl",
    "train/clip_fraction": "clip_fraction",
    "train/std": "action_std",
}

# Hard caps on the recorded episode preview, independent of how long the
# real episode/env actually runs — without these, a slow-to-terminate env
# (BipedalWalker, MuJoCo, ...) could turn one "occasional" render into a
# many-MB GIF and noticeably stall training on the render_every_steps tick.
_MAX_EPISODE_STEPS = 500
_MAX_GIF_FRAMES = 120
_GIF_MAX_SIDE = 320
# Belt-and-suspenders byte cap: visually noisy renders (Atari, MuJoCo, ...)
# compress far worse than the cartoon-flat classic-control envs this was
# tuned against, so if the first attempt still comes out large, retry once
# with a smaller/shorter GIF instead of shipping a multi-MB blob through
# metrics.json and the metrics WebSocket on every render tick.
_MAX_GIF_BYTES = 3_000_000
# See the identical constant/reasoning in `runner_utils.py` — `_on_step()`
# below runs once per env step, so checking `stop.flag` unconditionally
# every time is a filesystem `stat()` per step that adds up over a long
# run for no real benefit (a few dozen steps of staleness is still a
# fraction of a second for any env in this app).
_STOP_CHECK_EVERY_STEPS = 50


def _subsample(frames: list[np.ndarray], max_frames: int) -> list[np.ndarray]:
    """Evenly thins a long frame list down to at most `max_frames`,
    preserving temporal order (start/middle/end all stay represented) rather
    than truncating — dropping only how *smooth* the playback looks, never
    how much of the episode is shown."""
    if len(frames) <= max_frames:
        return frames
    idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
    return [frames[i] for i in idx]


def _build_gif_bytes(frames: list[np.ndarray], fps: float, max_frames: int, max_side: int) -> bytes:
    from PIL import Image

    frames = _subsample(frames, max_frames)
    images = [Image.fromarray(f.astype(np.uint8)).convert("RGB") for f in frames]
    for img in images:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    frame_ms = max(20, round(1000 / max(fps, 1e-6)))
    # Linger on the final frame. The GIF intentionally has no `loop`
    # extension: each newly recorded episode plays exactly once and then
    # remains on its final frame until a newer recording replaces it.
    durations = [frame_ms] * (len(images) - 1) + [frame_ms * 4]
    buf = io.BytesIO()
    images[0].save(
        buf, format="GIF", save_all=True, append_images=images[1:],
        duration=durations, optimize=True,
    )
    return buf.getvalue()


def persist_episode_gif(run_dir: Path, gif_b64: str, step: int) -> str:
    """Write the live-preview GIF to disk — always `episode_preview.gif`
    (latest) plus a step-stamped copy under `previews/` for history."""
    data = base64.b64decode(gif_b64)
    run_dir.mkdir(parents=True, exist_ok=True)
    rel_latest = "episode_preview.gif"
    (run_dir / rel_latest).write_bytes(data)
    previews = run_dir / "previews"
    previews.mkdir(exist_ok=True)
    (previews / f"episode_{step:08d}.gif").write_bytes(data)
    return rel_latest


def _encode_episode_gif(frames: list[np.ndarray], fps: float) -> str:
    """Packs a full episode's frames into a single-play animated GIF — a
    single self-contained blob the frontend can drop straight into an
    `<img>` tag. It stops on the final frame instead of replaying stale
    motion forever after training has ended."""
    data = _build_gif_bytes(frames, fps, _MAX_GIF_FRAMES, _GIF_MAX_SIDE)
    if len(data) > _MAX_GIF_BYTES:
        # Visually busy env — halve both knobs and try once more rather
        # than shipping the oversized first attempt.
        data = _build_gif_bytes(frames, fps, _MAX_GIF_FRAMES // 2, _GIF_MAX_SIDE // 2)
    return base64.b64encode(data).decode("ascii")


def render_episode(
    make_render_env: Callable[[], Any] | None,
    predict: Callable[[Any, bool], tuple[Any, Any]],
    max_steps: int = _MAX_EPISODE_STEPS,
) -> str | None:
    """Rolls out `predict` in a fresh render-mode env for one full episode —
    reset all the way to terminated/truncated (capped at `max_steps` as a
    safety net for envs that can run "forever") — and returns it as a single
    base64-encoded single-play GIF.

    Deliberately a whole coherent playthrough rather than one freeze-frame:
    grabbing just the *last* frame of a fresh rollout every
    `render_every_steps` (the previous behavior) showed a new, unrelated,
    arbitrary moment each time it refreshed — confusing to watch, and not
    actually representative of "how is the agent doing" the way watching it
    play a full episode is.

    `predict` is called as `predict(obs, episode_start)` — `episode_start`
    is `True` only on the very first step right after `env.reset()`, so a
    memory-enabled (LSTM/GRU) algorithm knows to start this rollout from a
    fresh hidden state instead of carrying one over from wherever a
    *previous* preview render (or, worse, actual training) happened to
    leave it. Callers that don't care just ignore the second argument."""
    if make_render_env is None:
        return None
    try:
        env = make_render_env()
        obs, _ = env.reset()
        first_frame = env.render()
        frames = [np.asarray(first_frame)] if first_frame is not None else []
        episode_start = True
        for _ in range(max_steps):
            action, _ = predict(obs, episode_start)
            episode_start = False
            obs, _, terminated, truncated, _ = env.step(action)
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame))
            if terminated or truncated:
                break
        fps = float(getattr(env, "metadata", {}).get("render_fps", 20) or 20)
        env.close()
        if not frames:
            return None
        return _encode_episode_gif(frames, fps=fps)
    except Exception:
        return None


class MetricsCallback(BaseCallback):
    def __init__(
        self,
        run_dir: Path,
        env_id: str,
        algo_id: str,
        total_timesteps: int,
        make_render_env: Callable[[], Any] | None = None,
        write_every_steps: int = 500,
        render_every_steps: int = 2000,
        static_info: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.env_id = env_id
        self.algo_id = algo_id
        self.total_timesteps = total_timesteps
        self.make_render_env = make_render_env
        self.write_every_steps = write_every_steps
        self.render_every_steps = render_every_steps
        # Static per-run facts (effective hyperparams, network/policy shape, ...)
        # that don't change step to step — merged into every snapshot so the
        # Training Monitor always has them, without a separate round-trip.
        self.static_info = static_info or {}
        self.start_time = time.time()
        self._last_write = 0
        self._last_render = 0
        self._last_stop_check = 0
        self._stop_requested_cached = False
        self._last_gif: str | None = None
        self._last_gif_step = 0
        self._stopped_early = False
        # Cumulative episode counter — SB3 only exposes a *trailing* window
        # via `model.ep_info_buffer` (a `deque(maxlen=100)`, see
        # `_episode_stats` below), so a running total needs its own
        # bookkeeping here, summed off `dones` every step (`_on_step`
        # below) - SB3 puts `env.step()`'s own `dones` array into
        # `self.locals` via `callback.update_locals(locals())` right before
        # calling `on_step()`, for both on-policy (`collect_rollouts` in
        # `on_policy_algorithm.py`) and off-policy (`off_policy_algorithm.py`)
        # algorithms alike.
        self._episodes_completed = 0

    def _stop_requested(self) -> bool:
        if not self._stop_requested_cached and self.num_timesteps - self._last_stop_check >= _STOP_CHECK_EVERY_STEPS:
            self._last_stop_check = self.num_timesteps
            self._stop_requested_cached = (self.run_dir / "stop.flag").exists()
        return self._stop_requested_cached

    def _episode_stats(self) -> tuple[float | None, float | None]:
        buf = getattr(self.model, "ep_info_buffer", None)
        if not buf:
            return None, None
        rewards = [ep["r"] for ep in buf]
        lengths = [ep["l"] for ep in buf]
        return float(np.mean(rewards)), float(np.mean(lengths))

    def _train_metrics(self) -> dict[str, float]:
        """Best-effort pull of whatever SB3's own logger recorded during the
        most recent `train()` call — see `_SB3_TRAIN_METRIC_KEYS` above."""
        name_to_value = getattr(getattr(self.model, "logger", None), "name_to_value", None)
        if not name_to_value:
            return {}
        out: dict[str, float] = {}
        for sb3_key, our_key in _SB3_TRAIN_METRIC_KEYS.items():
            value = name_to_value.get(sb3_key)
            if value is not None:
                try:
                    out[our_key] = float(value)
                except (TypeError, ValueError):
                    pass
        return out

    def _maybe_render(self) -> bool:
        if not self.make_render_env or self.num_timesteps - self._last_render < self.render_every_steps:
            return False
        self._last_render = self.num_timesteps
        # SB3 policies handle their own recurrent state internally when the
        # policy needs it (e.g. sb3-contrib's RecurrentPPO) — this app
        # doesn't use those, so `episode_start` is simply unused here.
        gif = render_episode(self.make_render_env, lambda obs, episode_start: self.model.predict(obs, deterministic=True))
        if not gif:
            # A transient failure (env recreation hiccup, ...) keeps
            # showing the previous episode instead of blanking the preview.
            return False
        self._last_gif = gif
        self._last_gif_step = self.num_timesteps
        persist_episode_gif(self.run_dir, gif, self.num_timesteps)
        return True

    def _write_snapshot(self, status: str, extra: dict | None = None) -> None:
        mean_reward, mean_length = self._episode_stats()
        elapsed = time.time() - self.start_time
        snapshot = {
            "run_id": self.run_dir.name,
            "kind": "gym",
            "status": status,
            "algo": self.algo_id,
            "env_id": self.env_id,
            "step": int(self.num_timesteps),
            "total_timesteps": int(self.total_timesteps),
            "episode_reward_mean": mean_reward,
            "episode_length_mean": mean_length,
            "episodes_completed": self._episodes_completed,
            "fps": round(self.num_timesteps / elapsed, 1) if elapsed > 0 else 0,
            "elapsed_seconds": round(elapsed, 1),
        }
        snapshot.update(self.static_info)
        snapshot.update(self._train_metrics())
        just_rendered = self._maybe_render()
        if self._last_gif:
            # `episode_gif_file`/`episode_gif_step` are cheap to re-attach
            # to every snapshot — the Training Monitor fetches/cache-busts
            # `GET .../preview.gif` off them whenever the (up to a few MB)
            # base64 blob itself isn't present. That blob is only included
            # on the exact write that just captured it (or the run's very
            # last write, belt-and-suspenders in case serving the file back
            # ever fails) — see the identical comment in `runner_utils.py`.
            snapshot["episode_gif_file"] = "episode_preview.gif"
            snapshot["episode_gif_step"] = self._last_gif_step
            if just_rendered or status != "running":
                snapshot["episode_gif_base64"] = self._last_gif
        if extra:
            snapshot.update(extra)
        snapshot = json_safe(snapshot)
        append_history(self.run_dir, snapshot)
        (self.run_dir / "metrics.json").write_text(json.dumps(snapshot))

    def _on_training_start(self) -> None:
        self._write_snapshot("running")

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is not None:
            self._episodes_completed += int(np.sum(dones))
        if self.num_timesteps - self._last_write >= self.write_every_steps:
            self._last_write = self.num_timesteps
            self._write_snapshot("running")
        if self._stop_requested():
            self._stopped_early = True
            return False
        return True

    def _on_training_end(self) -> None:
        status = "stopped" if self._stopped_early else "completed"
        self._write_snapshot(status)
