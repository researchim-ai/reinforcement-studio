"""Optional wrapper nodes the Experiment Designer graph can attach to an env."""
from __future__ import annotations

from typing import Any

import gymnasium as gym


def apply_wrappers(env: gym.Env, wrapper_specs: list[dict[str, Any]]) -> gym.Env:
    """Applies a list of {"type": ..., "params": {...}} wrapper specs in order."""
    for spec in wrapper_specs or []:
        wtype = spec.get("type")
        params = spec.get("params", {}) or {}
        if wtype == "time_limit":
            env = gym.wrappers.TimeLimit(env, max_episode_steps=int(params.get("max_episode_steps", 500)))
        elif wtype == "normalize_observation":
            env = gym.wrappers.NormalizeObservation(env)
        elif wtype == "normalize_reward":
            env = gym.wrappers.NormalizeReward(env, gamma=float(params.get("gamma", 0.99)))
        elif wtype == "clip_action":
            env = gym.wrappers.ClipAction(env)
        elif wtype == "clip_reward":
            env = gym.wrappers.TransformReward(
                env,
                lambda r, lo=params.get("min", -10), hi=params.get("max", 10): max(lo, min(hi, r)),
            )
        elif wtype == "frame_stack":
            env = gym.wrappers.FrameStackObservation(env, stack_size=int(params.get("num_stack", 4)))
        # Unknown wrapper types are ignored rather than raising, so a stale
        # graph node never blocks a run.
    return env


WRAPPER_CATALOG = [
    {"type": "time_limit", "label": "Time Limit", "params": {"max_episode_steps": 500}},
    {"type": "normalize_observation", "label": "Normalize Observation", "params": {}},
    {"type": "normalize_reward", "label": "Normalize Reward", "params": {"gamma": 0.99}},
    {"type": "clip_action", "label": "Clip Action", "params": {}},
    {"type": "clip_reward", "label": "Clip Reward", "params": {"min": -10, "max": 10}},
    {"type": "frame_stack", "label": "Frame Stack", "params": {"num_stack": 4}},
]
