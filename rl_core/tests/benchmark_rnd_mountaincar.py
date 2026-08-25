"""Repeatable sparse-reward A/B benchmark (not part of the fast test suite).

Run:
    python -m rl_core.tests.benchmark_rnd_mountaincar
"""
from __future__ import annotations

import json

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor

from rl_core.algorithms.base import TrainingCallback
from rl_core.algorithms.native.dqn import DEFAULT_HYPERPARAMS, NativeDQN


class CoverageWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.visited_bins: set[tuple[int, int]] = set()
        self.successes = 0

    def _record(self, obs) -> None:
        position = int(np.clip((obs[0] + 1.2) / 1.8 * 59, 0, 59))
        velocity = int(np.clip((obs[1] + 0.07) / 0.14 * 39, 0, 39))
        self.visited_bins.add((position, velocity))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._record(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._record(obs)
        self.successes += int(terminated)
        return obs, reward, terminated, truncated, info


def run(use_rnd: bool, steps: int = 10_000, seed: int = 42) -> dict[str, int]:
    tracked_env = CoverageWrapper(gym.make("MountainCar-v0"))
    env = Monitor(tracked_env)
    hyperparams = {
        **DEFAULT_HYPERPARAMS,
        "intrinsic_exploration": int(use_rnd),
        "rnd_bonus_coef": 0.2,
        "rnd_feature_dim": 32,
        "rnd_hidden_dim": 64,
        "learning_starts": 500,
        "buffer_size": 10_000,
        "batch_size": 64,
        "train_freq": 4,
        "target_update_interval": 500,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.05,
    }
    NativeDQN(env, hyperparams, seed, "cpu").learn(
        steps, TrainingCallback(lambda *_args: True),
    )
    result = {
        "coverage_bins": len(tracked_env.visited_bins),
        "successes": tracked_env.successes,
    }
    env.close()
    return result


if __name__ == "__main__":
    output = {
        "steps": 10_000,
        "seed": 42,
        "baseline": run(False),
        "rnd": run(True),
    }
    print(json.dumps(output, indent=2))
