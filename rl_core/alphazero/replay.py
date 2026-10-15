"""Preallocated ring buffer of self-play positions.

Positions are stored compactly — observation as float16, policy target as
the top-K (action, probability) pairs over the legal moves, value target as
a single win/draw/loss class — so a chess buffer of a few hundred thousand
positions stays in the low hundreds of MB instead of storing a dense
4674-wide policy per position.
"""
from __future__ import annotations

import numpy as np

WIN, DRAW, LOSS = 0, 1, 2


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple[int, ...], action_size: int, max_policy_entries: int = 256) -> None:
        self.capacity = max(1, int(capacity))
        self.obs_shape = tuple(int(x) for x in obs_shape)
        self.action_size = int(action_size)
        self.k = max(1, min(self.action_size, int(max_policy_entries)))
        self._obs: np.ndarray | None = None
        self._idx: np.ndarray | None = None
        self._prob: np.ndarray | None = None
        self._outcome: np.ndarray | None = None
        self.size = 0
        self.cursor = 0
        self.total_added = 0

    def __len__(self) -> int:
        return self.size

    def _allocate(self) -> None:
        idx_dtype = np.int16 if self.action_size <= np.iinfo(np.int16).max else np.int32
        self._obs = np.zeros((self.capacity, *self.obs_shape), dtype=np.float16)
        self._idx = np.zeros((self.capacity, self.k), dtype=idx_dtype)
        self._prob = np.zeros((self.capacity, self.k), dtype=np.float16)
        self._outcome = np.zeros(self.capacity, dtype=np.int8)

    def add(self, obs: np.ndarray, legal: np.ndarray, policy: np.ndarray, outcome: int) -> None:
        if self._obs is None:
            self._allocate()
        if len(legal) > self.k:
            keep = np.argpartition(-policy, self.k - 1)[: self.k]
            legal, policy = legal[keep], policy[keep]
        policy = policy / max(float(policy.sum()), 1e-12)
        pos = self.cursor
        self._obs[pos] = obs
        self._idx[pos] = 0
        self._prob[pos] = 0
        self._idx[pos, : len(legal)] = legal
        self._prob[pos, : len(legal)] = policy
        self._outcome[pos] = outcome
        self.cursor = (pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_added += 1

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (obs float32, policy indices int64, policy probs float32,
        outcome int64) — the policy as (B, K) pairs to scatter into a dense
        target on the training device."""
        rows = rng.integers(0, self.size, size=int(batch_size))
        return (
            self._obs[rows].astype(np.float32),
            self._idx[rows].astype(np.int64),
            self._prob[rows].astype(np.float32),
            self._outcome[rows].astype(np.int64),
        )
