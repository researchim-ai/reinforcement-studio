"""Runs a Gymnasium experiment (PPO/DQN/A2C via Stable-Baselines3)."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.monitor import Monitor

from rl_core.algorithms.metrics_callback import MetricsCallback
from rl_core.envs.wrappers import apply_wrappers

try:
    import ale_py

    gym.register_envs(ale_py)
except ImportError:
    pass

ALGO_CLASSES = {"ppo": PPO, "dqn": DQN, "a2c": A2C}

DEFAULT_HYPERPARAMS = {
    "ppo": {"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64, "gamma": 0.99, "ent_coef": 0.0},
    "dqn": {"learning_rate": 1e-3, "buffer_size": 50_000, "batch_size": 64, "gamma": 0.99, "exploration_fraction": 0.2},
    "a2c": {"learning_rate": 7e-4, "n_steps": 5, "gamma": 0.99, "ent_coef": 0.01},
}


def _filter_kwargs(cls: type, hyperparams: dict[str, Any]) -> dict[str, Any]:
    """Drops keys the SB3 constructor doesn't accept so a stale hyperparam
    from the UI never crashes a run."""
    accepted = set(inspect.signature(cls.__init__).parameters)
    return {k: v for k, v in hyperparams.items() if k in accepted}


def _make_env(env_id: str, wrapper_specs: list[dict], render: bool = False) -> gym.Env:
    env = gym.make(env_id, render_mode="rgb_array" if render else None)
    env = apply_wrappers(env, wrapper_specs)
    if isinstance(env.observation_space, (gym.spaces.Tuple, gym.spaces.Dict)):
        # SB3 policies don't support Tuple/Dict observation spaces at all
        # (e.g. Blackjack-v1 is Tuple(Discrete(32), Discrete(11), Discrete(2))).
        # Flatten into a single Box (one-hot for Discrete components) so any
        # such env "just works" with MlpPolicy, instead of failing at
        # model construction with "observation space is not supported".
        env = gym.wrappers.FlattenObservation(env)
    return env


def _space_info(space: gym.Space) -> dict[str, Any]:
    info: dict[str, Any] = {
        "type": space.__class__.__name__,
        "shape": list(space.shape) if getattr(space, "shape", None) is not None else None,
    }
    if hasattr(space, "n"):
        info["n"] = int(space.n)
    return info


def _policy_for_env(env: gym.Env) -> str:
    """MlpPolicy for vector obs, CnnPolicy for image / Atari / CarRacing frames."""
    space = env.observation_space
    if isinstance(space, gym.spaces.Box) and len(space.shape) >= 2:
        # (H, W), (H, W, C) or (C, H, W) — treat as an image.
        if len(space.shape) == 2 or (len(space.shape) == 3 and min(space.shape) <= 4):
            return "CnnPolicy"
        if len(space.shape) == 3 and np.prod(space.shape) > 256:
            return "CnnPolicy"
    return "MlpPolicy"


def _run_sb3(algo_cls: type, algo_label: str, hyperparams: dict[str, Any], config: dict[str, Any], run_dir: Path) -> None:
    """Drives any SB3-compatible `BaseAlgorithm` subclass through the
    standard Monitor + MetricsCallback pipeline. Shared by the built-in
    `run()` below and by `rl_core/algorithms/custom_runner.py` for custom
    algorithms that subclass an existing SB3 algorithm (e.g. PPO)."""
    env_cfg = config.get("environment", {})
    training_cfg = config.get("training", {})

    env_id = env_cfg["id"]
    wrapper_specs = env_cfg.get("wrappers", [])
    total_timesteps = int(training_cfg.get("total_timesteps", 50_000))
    seed = training_cfg.get("seed")

    train_env = Monitor(_make_env(env_id, wrapper_specs))
    if seed is not None:
        train_env.reset(seed=int(seed))

    model_kwargs = _filter_kwargs(algo_cls, hyperparams)
    policy = _policy_for_env(train_env)
    model = algo_cls(policy, train_env, seed=seed, verbose=0, **model_kwargs)

    total_params = sum(p.numel() for p in model.policy.parameters())
    static_info = {
        "policy": policy,
        "device": str(model.device),
        "hyperparams": {**hyperparams, **model_kwargs},
        "total_params": int(total_params),
        "observation_space": _space_info(train_env.observation_space),
        "action_space": _space_info(train_env.action_space),
        "wrappers": wrapper_specs,
        "seed": seed,
    }

    callback = MetricsCallback(
        run_dir=run_dir,
        env_id=env_id,
        algo_id=algo_label,
        total_timesteps=total_timesteps,
        make_render_env=lambda: _make_env(env_id, wrapper_specs, render=True),
        static_info=static_info,
    )

    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    try:
        model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    finally:
        model.save(str(run_dir / "model.zip"))
        train_env.close()


def run(config: dict[str, Any], run_dir: Path) -> None:
    algo_cfg = config.get("algorithm", {})
    algo_id = algo_cfg.get("id", "ppo").lower()
    if algo_id not in ALGO_CLASSES:
        raise ValueError(f"Unknown algorithm: {algo_id}")
    hyperparams = {**DEFAULT_HYPERPARAMS.get(algo_id, {}), **(algo_cfg.get("hyperparams") or {})}
    _run_sb3(ALGO_CLASSES[algo_id], algo_id, hyperparams, config, run_dir)
