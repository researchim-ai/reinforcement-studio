"""Rollout storage for the native algorithms. Plain numpy ring/append
buffers — no vectorized envs, since every run here drives exactly one Gym
env instance (matches how the rest of the app already works)."""
from __future__ import annotations

from collections import deque
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
        # 1.0 at index t means a *new* episode began exactly at t (i.e. the
        # env was reset right before this observation) — only used by
        # recurrent training (see `sequences()`) to know where the LSTM/GRU
        # hidden state must be zeroed instead of carried over from t-1.
        self.episode_starts = np.zeros((capacity,), dtype=np.float32)
        self.advantages = np.zeros((capacity,), dtype=np.float32)
        self.returns = np.zeros((capacity,), dtype=np.float32)
        self._ptr = 0

    def add(self, obs, action, log_prob: float, value: float, reward: float, done: bool, episode_start: bool = False) -> None:
        i = self._ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.values[i] = value
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.episode_starts[i] = float(episode_start)
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

    def sequences(self, seq_len: int) -> Iterator[dict[str, np.ndarray]]:
        """Like `minibatches()`, but for recurrent training: this app never
        vectorizes envs, so one rollout *is* one continuous sequence — there
        are no other sequences to shuffle against, only contiguous chunks
        of this one to bound how far truncated BPTT looks back. Chunks are
        therefore yielded strictly in time order (never shuffled), each
        carrying its own `episode_starts` slice so the caller knows where
        within the chunk to reset hidden state instead of threading it in
        from the previous timestep."""
        n = self._ptr
        adv = self.advantages[:n]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for start in range(0, n, seq_len):
            end = min(start + seq_len, n)
            yield {
                "obs": self.obs[start:end],
                "actions": self.actions[start:end],
                "log_probs": self.log_probs[start:end],
                "values": self.values[start:end],
                "advantages": adv[start:end],
                "returns": self.returns[start:end],
                "episode_starts": self.episode_starts[start:end],
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


class ContinuousReplayBuffer:
    """Fixed-size circular experience replay buffer for continuous-action
    off-policy algorithms (SAC) — same idea as `ReplayBuffer` above, but
    actions are float vectors instead of a single discrete index."""

    def __init__(self, capacity: int, obs_shape: tuple[int, ...], action_dim: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def add(self, obs, action, reward: float, next_obs, done: bool) -> None:
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


class EpisodeSequenceReplayBuffer:
    """Episode-aware replay buffer for recurrent (DRQN-style) off-policy
    training — the plain `ReplayBuffer` above stores independently shuffled
    transitions, which throws away exactly the temporal order an LSTM/GRU
    needs to be useful. This stores whole episodes instead, and `sample()`
    draws fixed-length *contiguous* windows from them for truncated BPTT.

    Windows start from a zero hidden state (Hausknecht & Stone 2015's
    original DRQN) rather than R2D2's "burn-in" warm-start — simpler, and
    the recurrent core has `min(seq_len, episode_length)` real steps to
    settle into a useful hidden state before the loss is computed on any
    given window, which is plenty at this app's episode lengths.

    One consequence of storing whole episodes worth noting: a transition
    only becomes sampleable once its *entire* episode has finished, so
    training draws exclusively from *completed* episodes — recent
    experience from a still-running (long) episode isn't available yet.
    Fine for this app's small/fast environments, where episodes finish in
    well under a second; would matter more for environments with very long
    or unbounded episodes.
    """

    def __init__(self, capacity: int, obs_shape: tuple[int, ...]) -> None:
        self.capacity = capacity  # total *transitions* across all stored episodes
        self.obs_shape = obs_shape
        self._episodes: deque[dict[str, np.ndarray]] = deque()
        self._current: list[tuple] = []
        self._size = 0

    def add(self, obs, action: int, reward: float, next_obs, done: bool) -> None:
        self._current.append((obs, action, reward, next_obs, done))
        if done:
            self._finish_episode()

    def _finish_episode(self) -> None:
        if not self._current:
            return
        obs, actions, rewards, next_obs, dones = zip(*self._current)
        episode = {
            "obs": np.asarray(obs, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.int64),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "next_obs": np.asarray(next_obs, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.float32),
        }
        self._episodes.append(episode)
        self._size += len(self._current)
        self._current = []
        while self._size > self.capacity and len(self._episodes) > 1:
            dropped = self._episodes.popleft()
            self._size -= len(dropped["dones"])

    def __len__(self) -> int:
        return self._size

    def sample(self, batch_size: int, seq_len: int) -> dict[str, np.ndarray]:
        """Returns `(B, T, ...)` windows padded to `seq_len` with a `mask`
        (`1.0` for real steps, `0.0` for padding past a short episode's
        end) — padded steps also get `dones=1.0` so they can never
        contribute a bootstrapped target even if the loss weighting is
        ever forgotten somewhere."""
        obs = np.zeros((batch_size, seq_len, *self.obs_shape), dtype=np.float32)
        next_obs = np.zeros((batch_size, seq_len, *self.obs_shape), dtype=np.float32)
        actions = np.zeros((batch_size, seq_len), dtype=np.int64)
        rewards = np.zeros((batch_size, seq_len), dtype=np.float32)
        dones = np.ones((batch_size, seq_len), dtype=np.float32)
        mask = np.zeros((batch_size, seq_len), dtype=np.float32)
        episodes = list(self._episodes)
        for b in range(batch_size):
            ep = episodes[np.random.randint(len(episodes))]
            ep_len = len(ep["dones"])
            window = min(seq_len, ep_len)
            start = np.random.randint(0, ep_len - window + 1)
            obs[b, :window] = ep["obs"][start : start + window]
            next_obs[b, :window] = ep["next_obs"][start : start + window]
            actions[b, :window] = ep["actions"][start : start + window]
            rewards[b, :window] = ep["rewards"][start : start + window]
            dones[b, :window] = ep["dones"][start : start + window]
            mask[b, :window] = 1.0
        return {"obs": obs, "actions": actions, "rewards": rewards, "next_obs": next_obs, "dones": dones, "mask": mask}


class NStepPrioritizedReplayBuffer:
    """Replay buffer for Rainbow DQN — bolts two of Rainbow's ingredients
    onto the plain `ReplayBuffer` above:

    - **N-step returns**: instead of storing single-step `(r, s')`
      transitions, a small deque accumulates up to `n_step` raw transitions
      and flushes the oldest one with its reward replaced by the discounted
      sum of the next `n_step` rewards (or fewer, if the episode ends
      first) and its "next state" moved `n_step` steps ahead. This
      propagates reward signal back faster than 1-step TD without the
      variance of full Monte-Carlo returns.
    - **Prioritized replay** (Schaul et al., 2016): each transition has a
      priority (new transitions get `max priority seen so far`, updated to
      `|TD-error|` after every train step touching them); sampling
      probability is `priority^alpha` normalized over the whole buffer.
      Uses plain O(n) proportional sampling over a numpy array rather than
      a sum-tree — simpler, and plenty fast at the buffer sizes (tens of
      thousands) this app actually trains with.

    Bootstrapping in the Q-update should scale the target by `gamma **
    n_step` (not `gamma`) — see `RainbowDQN._train_step`. Whenever the
    stored window is shorter than `n_step` (episode ended inside it),
    `dones` is `True` for that transition, so the `(1 - done)` bootstrap
    mask already zeroes out the (otherwise wrongly-scaled) target term.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        n_step: int = 3,
        gamma: float = 0.99,
        alpha: float = 0.6,
        eps: float = 1e-5,
    ) -> None:
        self.capacity = capacity
        self.n_step = max(1, n_step)
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self._ptr = 0
        self._size = 0
        self._max_priority = 1.0
        self._pending: deque = deque()

    def __len__(self) -> int:
        return self._size

    def _store(self, obs, action: int, reward: float, next_obs, done: bool) -> None:
        i = self._ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        # New transitions get the highest priority seen so far, so they're
        # guaranteed to be sampled (and get a real TD-error-based priority)
        # at least once soon instead of languishing at whatever a stale
        # zero-init would imply.
        self.priorities[i] = self._max_priority
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def _flush_oldest(self) -> None:
        obs0, action0, _, _, _ = self._pending[0]
        discounted_reward = 0.0
        next_obs_n, done_n = obs0, True
        for k, (_, _, r_k, next_obs_k, done_k) in enumerate(self._pending):
            discounted_reward += (self.gamma**k) * r_k
            next_obs_n, done_n = next_obs_k, done_k
            if done_k:
                break
        self._store(obs0, action0, discounted_reward, next_obs_n, done_n)
        self._pending.popleft()

    def add(self, obs, action: int, reward: float, next_obs, done: bool) -> None:
        self._pending.append((obs, action, reward, next_obs, done))
        if len(self._pending) >= self.n_step:
            self._flush_oldest()
        if done:
            while self._pending:
                self._flush_oldest()

    def sample(self, batch_size: int, beta: float) -> dict[str, np.ndarray]:
        """`beta` (importance-sampling exponent, annealed 0→1 over training)
        corrects the bias PER's non-uniform sampling introduces — samples
        that were drawn more often get their gradient contribution scaled
        down proportionally."""
        priorities = self.priorities[: self._size].astype(np.float64) + self.eps
        probs = priorities**self.alpha
        probs /= probs.sum()
        idx = np.random.choice(self._size, size=batch_size, p=probs)

        weights = (self._size * probs[idx]) ** (-beta)
        weights /= weights.max()

        return {
            "indices": idx,
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
            "weights": weights.astype(np.float32),
        }

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        priorities = np.abs(td_errors) + self.eps
        self.priorities[indices] = priorities
        self._max_priority = max(self._max_priority, float(priorities.max()))
