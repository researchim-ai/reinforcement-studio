"""Exploration building blocks shared by native value-based algorithms."""

from rl_core.algorithms.native.exploration.noisy import (
    NoisyLinear,
    reset_noise,
    set_noise_enabled,
)
from rl_core.algorithms.native.exploration.rnd import RNDModule
from rl_core.algorithms.native.exploration.schedules import linear_schedule

ACTION_EPSILON = 0
ACTION_NOISY_NET = 1
INTRINSIC_NONE = 0
INTRINSIC_RND = 1


def uses_noisy_net(hyperparams: dict) -> bool:
    return int(hyperparams.get("action_exploration", ACTION_EPSILON) or 0) == ACTION_NOISY_NET


def uses_rnd(hyperparams: dict) -> bool:
    return int(hyperparams.get("intrinsic_exploration", INTRINSIC_NONE) or 0) == INTRINSIC_RND


__all__ = [
    "ACTION_EPSILON",
    "ACTION_NOISY_NET",
    "INTRINSIC_NONE",
    "INTRINSIC_RND",
    "NoisyLinear",
    "RNDModule",
    "linear_schedule",
    "reset_noise",
    "set_noise_enabled",
    "uses_noisy_net",
    "uses_rnd",
]
