"""Shared driver for any `CustomAlgorithm` (from-scratch, no SB3) run —
used both by `native_runner.py` (our own default PPO/DQN/A2C) and by
`custom_runner.py` (user-authored plugins that subclass `CustomAlgorithm`
instead of an SB3 `BaseAlgorithm`). Mirrors the metrics.json / stop.flag /
live-frame contract that `MetricsCallback` implements for SB3 runs, so the
Training Monitor page treats every kind of run identically.
"""
from __future__ import annotations

import inspect
import json
import time
from pathlib import Path
from typing import Any, Callable, Union

from stable_baselines3.common.monitor import Monitor

from rl_core.algorithms.base import CustomAlgorithm, TrainingCallback
from rl_core.algorithms.metrics_callback import render_episode
from rl_core.algorithms.sb3_runner import _make_env, _space_info
from rl_core.device import resolve_device
from rl_core.inspect import _count_params, _describe_layers, _find_torch_module
from rl_core.netbuilder_store import resolve_network_spec

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
    device = resolve_device(training_cfg)

    train_env = Monitor(_make_env(env_id, wrapper_specs))
    if seed is not None:
        train_env.reset(seed=int(seed))

    # `network_spec` (a hand-designed architecture from the Network
    # Builder) is merged into a *copy* of hyperparams used only for
    # constructing the algorithm — kept out of `hyperparams` itself so it
    # never leaks into the "Гиперпараметры алгоритма" display (which
    # renders every key as a plain value) or gets clamped by the numeric
    # hyperparam UI. NativePPO/NativeA2C/NativeDQN check for this key
    # themselves (see rl_core/algorithms/native/{ppo,a2c,dqn}.py); a plugin
    # CustomAlgorithm subclass that doesn't know about it just ignores it.
    network_spec = resolve_network_spec(config)
    construct_hyperparams = {**hyperparams, "network_spec": network_spec} if network_spec else hyperparams
    algo = cls(train_env, construct_hyperparams, seed, device)

    # Memory-enabled (LSTM/GRU) algorithms need to know when a live-preview
    # rollout starts over so they reset hidden state instead of carrying
    # one over from the last preview (see `render_episode`'s docstring).
    # `cls` here can be a user-authored plugin predating this parameter, so
    # only pass it through if the algorithm's own `predict()` actually
    # declares it — anything else keeps working exactly as before.
    _predict_accepts_episode_start = "episode_start" in inspect.signature(algo.predict).parameters

    def _predict(obs: Any, episode_start: bool) -> tuple[Any, Any]:
        if _predict_accepts_episode_start:
            return algo.predict(obs, deterministic=True, episode_start=episode_start)
        return algo.predict(obs, deterministic=True)

    resolved_policy_label = policy_label(train_env.observation_space) if callable(policy_label) else policy_label
    static_info = {
        "policy": resolved_policy_label,
        "device": device,
        "hyperparams": hyperparams,
        "wrappers": wrapper_specs,
        "seed": seed,
        "observation_space": _space_info(train_env.observation_space),
        "action_space": _space_info(train_env.action_space),
    }
    # Best-effort network introspection so the Training Monitor's "Схема
    # алгоритма" card can draw the actual layer shapes for native PPO/DQN/A2C
    # and from-scratch plugins too, not just the SB3 path (_run_sb3 already
    # has this) — reuses the exact same helpers the pre-run Designer
    # inspector uses (rl_core/inspect.py), just pointed at the real trained
    # module instead of a throwaway one.
    module = _find_torch_module(algo)
    if module is not None:
        total_params, _ = _count_params(module)
        static_info["total_params"] = total_params
        static_info["layers"] = _describe_layers(module)

    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    start_time = time.time()
    state: dict[str, Any] = {
        "last_write": 0, "last_render": 0, "step": 0, "last_gif": None,
        "exploration_metrics": {},
    }
    recent_rewards: list[float] = []
    recent_extrinsic_rewards: list[float] = []
    recent_intrinsic_rewards: list[float] = []
    recent_lengths: list[float] = []

    def write_snapshot(step: int, status: str) -> None:
        elapsed = time.time() - start_time
        if step - state["last_render"] >= _RENDER_EVERY_STEPS:
            state["last_render"] = step
            # Only overwrite `last_gif` on a *successful* render — a
            # transient failure (env recreation hiccup, ...) should keep
            # showing the previous episode rather than blanking the preview.
            gif = render_episode(lambda: _make_env(env_id, wrapper_specs, render=True), _predict)
            if gif:
                state["last_gif"] = gif
        snapshot = {
            "run_id": run_dir.name,
            "kind": "gym",
            "status": status,
            "algo": algo_label,
            "env_id": env_id,
            "step": int(step),
            "total_timesteps": int(total_timesteps),
            "episode_reward_mean": (sum(recent_rewards[-100:]) / len(recent_rewards[-100:])) if recent_rewards else None,
            "episode_extrinsic_reward_mean": (
                sum(recent_extrinsic_rewards[-100:]) / len(recent_extrinsic_rewards[-100:])
            ) if recent_extrinsic_rewards else None,
            "episode_intrinsic_reward_mean": (
                sum(recent_intrinsic_rewards[-100:]) / len(recent_intrinsic_rewards[-100:])
            ) if recent_intrinsic_rewards else None,
            "episode_length_mean": (sum(recent_lengths[-100:]) / len(recent_lengths[-100:])) if recent_lengths else None,
            "fps": round(step / elapsed, 1) if elapsed > 0 else 0,
            "elapsed_seconds": round(elapsed, 1),
        }
        snapshot.update(static_info)
        snapshot.update(state["exploration_metrics"])
        # Keep re-attaching the last successfully recorded episode to every
        # snapshot (not just the one that just rendered it) — so a run that
        # finishes between two render ticks, or is reopened later without a
        # live WebSocket to have carried it forward client-side, still shows
        # its most recent episode instead of nothing at all.
        if state["last_gif"]:
            snapshot["episode_gif_base64"] = state["last_gif"]
        (run_dir / "metrics.json").write_text(json.dumps(snapshot))

    def writer(
        num_timesteps: int,
        episode_reward: float | None,
        episode_length: float | None,
        metrics: dict[str, float] | None = None,
    ) -> bool:
        state["step"] = num_timesteps
        if metrics:
            state["exploration_metrics"].update({
                key: value for key, value in metrics.items()
                if key not in {"episode_extrinsic_reward", "episode_intrinsic_reward"}
            })
            if "episode_extrinsic_reward" in metrics:
                recent_extrinsic_rewards.append(float(metrics["episode_extrinsic_reward"]))
            if "episode_intrinsic_reward" in metrics:
                recent_intrinsic_rewards.append(float(metrics["episode_intrinsic_reward"]))
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
