"""Evaluation mode: run N deterministic episodes of an already-trained model
with no further learning, for either a still-around run or a promoted Model
Zoo checkpoint — separate from the periodic "live preview" GIF recorded
*during* training (see `render_episode` in `metrics_callback.py`), which
never reports episode statistics anywhere.

Scoped to the Gym track only (native algorithms + custom Gym plugins,
whether `CustomAlgorithm`- or SB3 `BaseAlgorithm`-based) — AlphaZero already
has its own evaluation story (self-play arena gating during training) that
doesn't fit the same "run N episodes, report reward stats" shape.
"""
from __future__ import annotations

import inspect
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor

from rl_core.algorithms.base import CustomAlgorithm
from rl_core.algorithms.metrics_callback import _build_gif_bytes
from rl_core.algorithms.sb3_runner import _make_env
from rl_core.device import resolve_device
from rl_core.plugins.loader import load_gym_algorithm

_MAX_EPISODE_STEPS = 2000
_MAX_GIF_FRAMES = 150
_GIF_MAX_SIDE = 360


class EvaluationError(RuntimeError):
    """Raised for anything that keeps evaluation from running at all
    (unknown algorithm, env mismatch, ...) — surfaced verbatim to the UI."""


def resolve_algo_class(algo_id: str) -> tuple[type, bool]:
    """Returns `(cls, is_sb3)` — mirrors the dispatch in
    `rl_core/runner.py`/`native_runner.py`/`custom_runner.py` so evaluation
    loads exactly the same class training would have used."""
    if algo_id.startswith("custom:"):
        slug = algo_id.split(":", 1)[1]
        cls, _ = load_gym_algorithm(slug)
        return cls, issubclass(cls, BaseAlgorithm)

    from rl_core.algorithms.native_runner import ALGO_CLASSES

    if algo_id not in ALGO_CLASSES:
        raise EvaluationError(f"Оценка недоступна для алгоритма «{algo_id}»")
    return ALGO_CLASSES[algo_id], False


def _encode_gif_b64(frames: list[np.ndarray], fps: float) -> str | None:
    import base64

    if not frames:
        return None
    try:
        data = _build_gif_bytes(frames, fps, _MAX_GIF_FRAMES, _GIF_MAX_SIDE)
        return base64.b64encode(data).decode("ascii")
    except Exception:
        return None


def _run_episodes_custom_algorithm(
    algo: CustomAlgorithm, env_id: str, wrapper_specs: list[dict], episodes: int,
    seed: int | None, record_gif: bool, deterministic: bool,
) -> dict[str, Any]:
    accepts_episode_start = "episode_start" in inspect.signature(algo.predict).parameters
    rewards: list[float] = []
    lengths: list[int] = []
    gif_frames: list[np.ndarray] = []
    fps = 20.0

    for ep in range(episodes):
        want_frames = record_gif and ep == 0
        env = _make_env(env_id, wrapper_specs, render=want_frames)
        obs, _ = env.reset(seed=(seed + ep) if seed is not None else None)
        if want_frames:
            fps = float(getattr(env, "metadata", {}).get("render_fps", 20) or 20)
            frame = env.render()
            if frame is not None:
                gif_frames.append(np.asarray(frame))
        episode_start = True
        total_reward = 0.0
        length = 0
        for _ in range(_MAX_EPISODE_STEPS):
            if accepts_episode_start:
                action, _ = algo.predict(obs, deterministic=deterministic, episode_start=episode_start)
            else:
                action, _ = algo.predict(obs, deterministic=deterministic)
            episode_start = False
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += float(reward)
            length += 1
            if want_frames:
                frame = env.render()
                if frame is not None:
                    gif_frames.append(np.asarray(frame))
            if terminated or truncated:
                break
        env.close()
        rewards.append(total_reward)
        lengths.append(length)

    return _summarize(rewards, lengths, gif_frames, fps)


def _run_episodes_sb3(
    model: BaseAlgorithm, env_id: str, wrapper_specs: list[dict], episodes: int,
    seed: int | None, record_gif: bool, deterministic: bool,
) -> dict[str, Any]:
    rewards: list[float] = []
    lengths: list[int] = []
    gif_frames: list[np.ndarray] = []
    fps = 20.0

    for ep in range(episodes):
        want_frames = record_gif and ep == 0
        env = Monitor(_make_env(env_id, wrapper_specs, render=want_frames))
        obs, _ = env.reset(seed=(seed + ep) if seed is not None else None)
        if want_frames:
            fps = float(getattr(env, "metadata", {}).get("render_fps", 20) or 20)
            frame = env.render()
            if frame is not None:
                gif_frames.append(np.asarray(frame))
        total_reward = 0.0
        length = 0
        for _ in range(_MAX_EPISODE_STEPS):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += float(reward)
            length += 1
            if want_frames:
                frame = env.render()
                if frame is not None:
                    gif_frames.append(np.asarray(frame))
            if terminated or truncated:
                break
        env.close()
        rewards.append(total_reward)
        lengths.append(length)

    return _summarize(rewards, lengths, gif_frames, fps)


def _summarize(rewards: list[float], lengths: list[int], gif_frames: list[np.ndarray], fps: float) -> dict[str, Any]:
    result: dict[str, Any] = {
        "episodes": len(rewards),
        "rewards": rewards,
        "lengths": lengths,
        "reward_mean": statistics.fmean(rewards) if rewards else None,
        "reward_std": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        "reward_min": min(rewards) if rewards else None,
        "reward_max": max(rewards) if rewards else None,
        "length_mean": statistics.fmean(lengths) if lengths else None,
    }
    gif = _encode_gif_b64(gif_frames, fps)
    if gif:
        result["episode_gif_base64"] = gif
    return result


def evaluate_gym_run(
    config: dict[str, Any], model_path: Path, episodes: int = 10,
    record_gif: bool = True, seed: int | None = None, deterministic: bool = True,
) -> dict[str, Any]:
    """Loads the trained model described by `config` (a full `config.json` —
    environment + wrappers + algorithm id) from `model_path` and rolls out
    `episodes` fresh episodes with no further learning. Runs on CPU
    regardless of what the original run trained on — evaluation is cheap
    (no backward passes) and this avoids fighting an active training run for
    the GPU."""
    env_cfg = config.get("environment", {})
    algo_cfg = config.get("algorithm", {})
    env_id = env_cfg.get("id")
    wrapper_specs = env_cfg.get("wrappers", [])
    algo_id = algo_cfg.get("id", "ppo")
    if not env_id:
        raise EvaluationError("В конфиге отсутствует environment.id")

    device = resolve_device({"use_gpu": False})
    cls, is_sb3 = resolve_algo_class(algo_id)
    episodes = max(1, int(episodes))

    if is_sb3:
        model = cls.load(str(model_path), device=device)
        return _run_episodes_sb3(model, env_id, wrapper_specs, episodes, seed, record_gif, deterministic)

    assert issubclass(cls, CustomAlgorithm)
    probe_env = Monitor(_make_env(env_id, wrapper_specs))
    load_sig = inspect.signature(cls.load)
    algo = cls.load(model_path, probe_env, device=device) if "device" in load_sig.parameters else cls.load(model_path, probe_env)
    probe_env.close()
    return _run_episodes_custom_algorithm(algo, env_id, wrapper_specs, episodes, seed, record_gif, deterministic)
