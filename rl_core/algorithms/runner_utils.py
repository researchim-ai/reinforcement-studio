"""Shared driver for any `CustomAlgorithm` (from-scratch, no SB3) run —
used both by `native_runner.py` (our own default PPO/DQN/A2C) and by
`custom_runner.py` (user-authored plugins that subclass `CustomAlgorithm`
instead of an SB3 `BaseAlgorithm`). Mirrors the metrics.json / stop.flag /
live-frame contract that `MetricsCallback` implements for SB3 runs, so the
Training Monitor page treats every kind of run identically.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Union

from stable_baselines3.common.monitor import Monitor

from rl_core.algorithms.base import CustomAlgorithm, TrainingCallback
from rl_core.algorithms.metrics_callback import render_frame
from rl_core.algorithms.sb3_runner import _make_env

_WRITE_EVERY_STEPS = 500
_RENDER_EVERY_STEPS = 2000


def run_custom_algorithm(
    cls: type[CustomAlgorithm],
    hyperparams: dict[str, Any],
    config: dict[str, Any],
    run_dir: Path,
    algo_label: str,
    policy_label: Union[str, Callable[[Any], str]] = "custom",
) -> None:
    """Drives any `CustomAlgorithm` subclass through the standard
    Monitor + metrics.json pipeline that the rest of the app expects.
    `policy_label` may be a plain string or a `(observation_space) -> str`
    callable (used by native_runner.py to report "MlpPolicy"/"CnnPolicy"
    without instantiating a second throwaway env)."""
    env_cfg = config.get("environment", {})
    training_cfg = config.get("training", {})
    env_id = env_cfg["id"]
    wrapper_specs = env_cfg.get("wrappers", [])
    total_timesteps = int(training_cfg.get("total_timesteps", 50_000))
    seed = training_cfg.get("seed")
    device = "cpu"

    train_env = Monitor(_make_env(env_id, wrapper_specs))
    if seed is not None:
        train_env.reset(seed=int(seed))

    algo = cls(train_env, hyperparams, seed, device)

    resolved_policy_label = policy_label(train_env.observation_space) if callable(policy_label) else policy_label
    static_info = {
        "policy": resolved_policy_label,
        "device": device,
        "hyperparams": hyperparams,
        "wrappers": wrapper_specs,
        "seed": seed,
    }

    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    start_time = time.time()
    state = {"last_write": 0, "last_render": 0, "step": 0}
    recent_rewards: list[float] = []
    recent_lengths: list[float] = []

    def write_snapshot(step: int, status: str) -> None:
        elapsed = time.time() - start_time
        frame = None
        if step - state["last_render"] >= _RENDER_EVERY_STEPS:
            state["last_render"] = step
            frame = render_frame(
                lambda: _make_env(env_id, wrapper_specs, render=True),
                lambda obs: algo.predict(obs, deterministic=True),
            )
        snapshot = {
            "run_id": run_dir.name,
            "kind": "gym",
            "status": status,
            "algo": algo_label,
            "env_id": env_id,
            "step": int(step),
            "total_timesteps": int(total_timesteps),
            "episode_reward_mean": (sum(recent_rewards[-100:]) / len(recent_rewards[-100:])) if recent_rewards else None,
            "episode_length_mean": (sum(recent_lengths[-100:]) / len(recent_lengths[-100:])) if recent_lengths else None,
            "fps": round(step / elapsed, 1) if elapsed > 0 else 0,
            "elapsed_seconds": round(elapsed, 1),
        }
        snapshot.update(static_info)
        if frame:
            snapshot["frame_base64"] = frame
        (run_dir / "metrics.json").write_text(json.dumps(snapshot))

    def writer(num_timesteps: int, episode_reward: float | None, episode_length: float | None) -> bool:
        state["step"] = num_timesteps
        if episode_reward is not None:
            recent_rewards.append(episode_reward)
        if episode_length is not None:
            recent_lengths.append(episode_length)
        if num_timesteps - state["last_write"] >= _WRITE_EVERY_STEPS:
            state["last_write"] = num_timesteps
            write_snapshot(num_timesteps, "running")
        return not (run_dir / "stop.flag").exists()

    write_snapshot(0, "running")
    status = "completed"
    try:
        algo.learn(total_timesteps=total_timesteps, callback=TrainingCallback(writer))
        if (run_dir / "stop.flag").exists():
            status = "stopped"
    finally:
        algo.save(run_dir / "model.zip")
        train_env.close()

    write_snapshot(state["step"], status)
