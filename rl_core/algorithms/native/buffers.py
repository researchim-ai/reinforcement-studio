"""Rollout storage for the native algorithms. Plain numpy ring/append
buffers — no vectorized envs, since every run here drives exactly one Gym
env instance (matches how the rest of the app already works)."""
from __future__ import annotations

from typing import Iterator

import numpy as np


class RolloutBuffer:
    """On-policy buffer for PPO/A2C: collects `capacity` transitions, then
    computes GAE(lambda) advantages/returns once the rollout is full."""

    def __init__(self, capacity: int, obs_shape: tuple[int, ...], action_dim: int, discrete: bool) -> None:
        self.capacity = capacity
        self.discrete = discrete
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64) if discrete else np.zeros((capacity, action_dim), dtype=np.float32)
        self.log_probs = np.zeros((capacity,), dtype=np.float32)
        self.values = np.zeros((capacity,), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.advantages = np.zeros((capacity,), dtype=np.float32)
        self.returns = np.zeros((capacity,), dtype=np.float32)
        self._ptr = 0

    def add(self, obs, action, log_prob: float, value: float, reward: float, done: bool) -> None:
        i = self._ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.values[i] = value
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self._ptr += 1

    def full(self) -> bool:
        return self._ptr >= self.capacity

    def compute_returns_and_advantage(self, last_value: float, last_done: bool, gamma: float, gae_lambda: float) -> None:
        last_gae = 0.0
        next_value = last_value
        next_non_terminal = 1.0 - float(last_done)
        for t in reversed(range(self._ptr)):
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
            next_value = self.values[t]
            next_non_terminal = 1.0 - self.dones[t]
        self.returns[: self._ptr] = self.advantages[: self._ptr] + self.values[: self._ptr]

    def minibatches(self, batch_size: int) -> Iterator[dict[str, np.ndarray]]:
        n = self._ptr
        indices = np.random.permutation(n)
        adv = self.advantages[:n]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            yield {
                "obs": self.obs[idx],
                "actions": self.actions[idx],
                "log_probs": self.log_probs[idx],
                "values": self.values[idx],
                "advantages": adv[idx],
                "returns": self.returns[idx],
            }

    def all(self) -> dict[str, np.ndarray]:
        n = self._ptr
        adv = self.advantages[:n]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return {
            "obs": self.obs[:n],
            "actions": self.actions[:n],
            "log_probs": self.log_probs[:n],
            "values": self.values[:n],
            "advantages": adv,
            "returns": self.returns[:n],
        }

    def reset(self) -> None:
        self._ptr = 0


class ReplayBuffer:
    """Fixed-size circular experience replay buffer for DQN."""

    def __init__(self, capacity: int, obs_shape: tuple[int, ...]) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def add(self, obs, action: int, reward: float, next_obs, done: bool) -> None:
        i = self._ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def __len__(self) -> int:
        return self._size

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        idx = np.random.randint(0, self._size, size=batch_size)
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
        }
