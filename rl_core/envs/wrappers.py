"""Optional wrapper nodes the Experiment Designer graph can attach to an env."""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import gymnasium as gym


class FrameSkip(gym.Wrapper):
    """Repeats the same action for `skip` raw env steps and returns only the
    *last* frame, reward summed across the skipped steps.

    Deliberately **not** gymnasium's own `MaxAndSkipObservation`, which
    additionally takes the pixel-wise max across the last two frames — a
    trick from the original Atari DQN paper to cancel out sprites that
    Atari's rendering only draws every other frame. CarRacing doesn't
    flicker like that; maxing two frames of a fast-moving car instead
    creates visible ghosting/double edges right where precise steering
    needs a clean track boundary, which can cap how well the policy can
    ever drive regardless of how long it trains. Matches
    `rl_zoo3.wrappers.FrameSkip`, the one CarRacing-v3's tuned PPO config
    actually uses (DLR-RM/rl-baselines3-zoo)."""

    def __init__(self, env: gym.Env, skip: int = 2) -> None:
        super().__init__(env)
        self.skip = max(1, int(skip))

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class ChannelStackObservation(gym.ObservationWrapper):
    """Stacks the last `num_stack` observations along the *last* (channel)
    axis — `(H, W, C)` -> `(H, W, C*num_stack)` — instead of gymnasium's own
    `FrameStackObservation`, which stacks on a brand new leading axis
    (`(H, W, C)` -> `(num_stack, H, W, C)`).

    That leading-axis shape is 4-D, which fails every "is this an image?"
    heuristic in this codebase (`rl_core/algorithms/native/preprocessing.py`
    and SB3's own `is_image_space`, both of which require exactly 3 dims) —
    so attaching Frame Stack to a pixel env silently degrades to flattening
    the whole stack through an `MLPExtractor`/`MlpPolicy` instead of a CNN,
    exactly the "learns nothing from raw pixels" outcome Frame Stack exists
    to avoid. Channel-concatenation is also the standard Atari-DQN-style
    convention (4 grayscale frames -> a 4-channel "image"), so this keeps
    any stacked pixel observation a proper `(H, W, C)` CNN input, and for
    non-image vector observations it degenerates to "concatenate the last N
    feature vectors", which is equally valid history-stacking there."""

    def __init__(self, env: gym.Env, num_stack: int) -> None:
        super().__init__(env)
        self.num_stack = max(1, int(num_stack))
        space = env.observation_space
        assert isinstance(space, gym.spaces.Box), "ChannelStackObservation only supports Box observations"
        self.observation_space = gym.spaces.Box(
            low=np.repeat(space.low, self.num_stack, axis=-1),
            high=np.repeat(space.high, self.num_stack, axis=-1),
            dtype=space.dtype,
        )
        self._frames: deque = deque(maxlen=self.num_stack)

    def _stack(self) -> np.ndarray:
        return np.concatenate(list(self._frames), axis=-1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self._frames.append(obs)
        return self._stack(), info

    def observation(self, observation):
        self._frames.append(observation)
        return self._stack()


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
            env = ChannelStackObservation(env, num_stack=int(params.get("num_stack", 4)))
        elif wtype == "frame_skip":
            # Cuts the number of policy forward/backward passes needed to
            # cover the same amount of "game time" roughly in half for
            # skip=2, and is applied *before* Resize/Grayscale below (i.e.
            # wraps the raw base env) so those only ever run once per
            # skipped block instead of on every raw frame. See `FrameSkip`
            # above for why this doesn't max-pool frames like Atari's.
            env = FrameSkip(env, skip=int(params.get("skip", 2)))
        elif wtype == "resize_observation":
            shape = (int(params.get("height", 64)), int(params.get("width", 64)))
            env = gym.wrappers.ResizeObservation(env, shape=shape)
        elif wtype == "grayscale_observation":
            env = gym.wrappers.GrayscaleObservation(env, keep_dim=bool(params.get("keep_dim", True)))
        elif wtype and wtype.startswith("custom_reward:"):
            # Type encodes the plugin slug directly (custom_reward:<slug>)
            # rather than relying on params, so multiple custom reward
            # scripts each get a distinct wrapper-catalog entry/type — see
            # backend/routes/environments.py::_custom_wrapper_entries.
            from rl_core.envs.reward_fn import CustomRewardWrapper
            from rl_core.plugins.loader import load_reward_fn

            slug = wtype.split(":", 1)[1]
            env = CustomRewardWrapper(env, load_reward_fn(slug))
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
    # Pixel-env speed trio, in the order they're meant to be chained (see
    # CarRacing-v3's `recommended_wrappers` in rl_core/envs/registry.py):
    # Frame Skip closest to the base env (cheapest — skips whole raw steps),
    # then Resize/Grayscale to shrink what the CNN actually has to process.
    {"type": "frame_skip", "label": "Frame Skip", "params": {"skip": 2}},
    {"type": "resize_observation", "label": "Resize Observation", "params": {"height": 64, "width": 64}},
    {"type": "grayscale_observation", "label": "Grayscale Observation", "params": {"keep_dim": True}},
]
