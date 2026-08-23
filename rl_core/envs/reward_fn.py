"""Contract for custom reward-shaping functions (Gym track only — AlphaZero
board games use fixed +1/0/-1 win/draw/loss rewards intrinsic to the game,
so there's nothing to shape there).
"""
from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym


def shape_reward(
    obs: Any,
    action: Any,
    reward: float,
    next_obs: Any,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
) -> float:
    """Reference signature every custom reward script must expose, either as
    a module-level `shape_reward` (this name) or as `REWARD_FN` pointing at
    a differently-named function — the loader accepts either. Return the
    (possibly reshaped) reward for this step."""
    return reward


class CustomRewardWrapper(gym.Wrapper):
    """Applies a user-supplied `shape_reward`-shaped function to every
    step's reward. Wired in as wrapper type `"custom_reward:<slug>"` (see
    `rl_core/envs/wrappers.py::apply_wrappers`).
    """

    def __init__(self, env: gym.Env, reward_fn: Callable[..., float]) -> None:
        super().__init__(env)
        self._reward_fn = reward_fn
        self._last_obs: Any = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(self._reward_fn(self._last_obs, action, reward, obs, terminated, truncated, info))
        self._last_obs = obs
        return obs, reward, terminated, truncated, info
