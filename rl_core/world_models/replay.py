"""Experience storage for world model training. Real transitions feed two
rather different consumers, so there are two buffer shapes:

- `SequenceReplayBuffer` — whole episodes, sampled as fixed-length
  *contiguous* windows (needed by the RSSM and the MDN-RNN, both of which
  are recurrent and need to see genuine temporal order to fit anything
  meaningful). Same "only a fully-completed episode is sampleable"
  convention as `rl_core/algorithms/native/buffers.py::EpisodeSequenceReplayBuffer`,
  generalized to continuous *and* discrete actions (every action is stored
  pre-converted to a flat float vector — one-hot for `Discrete`, raw for
  `Box` — via the exact same `obs_to_array` this app already uses for
  observations, since a GRU/LSTM input has no reason to care which kind of
  space produced it).
- `rl_core.algorithms.native.buffers.ContinuousReplayBuffer` (reused
  as-is, not reimplemented here) — plain shuffled single-step transitions,
  which is all the ensemble dynamics model (no recurrence at all) or
  MBPO's SAC critic ever need.

`collect_rollout` drives a real (single, non-vectorized) env with either a
random policy (the default — this is exactly how the original World
Models paper and most model-based-RL pretraining collects its first batch
of data) or a caller-supplied one, pushing every transition into whichever
buffer(s) are passed in. `collect_rollout_vec` is its `num_envs>1`
counterpart — see its own docstring for why (and how much) that speeds
real-experience collection up; `collect_rollout_auto` picks whichever of
the two fits `env` so callers (`trainer.py`, `world_models_ha.py`) don't
need an `is_vector_env` branch of their own.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Callable

import numpy as np
import gymnasium as gym

from rl_core.algorithms.native.preprocessing import obs_flat_dim, obs_to_array


class SequenceReplayBuffer:
    """`num_lanes > 1` lets `add(..., lane=i)` accumulate `num_lanes`
    independent in-progress episodes concurrently (one per env lane of a
    `VectorEnv`) without their interleaved transitions getting spliced
    into one another — each lane keeps its own `_current` accumulator,
    flushed into the shared `_episodes` pool independently whenever *that*
    lane's own episode ends. `num_lanes=1` (the default) is exactly the
    original single-episode-at-a-time behavior, so every existing caller
    (`dreamer.py`, `world_models_ha.py`'s ES phase, single-env
    `trainer.py` runs) is unaffected."""

    def __init__(self, capacity: int, obs_shape: tuple[int, ...], action_dim: int, num_lanes: int = 1) -> None:
        self.capacity = capacity  # total *transitions* across every stored episode
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self._episodes: deque[dict[str, np.ndarray]] = deque()
        self._current: list[list[tuple[Any, Any, float, bool]]] = [[] for _ in range(max(1, num_lanes))]
        self._size = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, done: bool, lane: int = 0) -> None:
        self._current[lane].append((obs, action, reward, done))
        if done:
            self._finish_episode(lane)

    def _finish_episode(self, lane: int = 0) -> None:
        current = self._current[lane]
        if not current:
            return
        obs, actions, rewards, dones = zip(*current)
        episode = {
            "obs": np.asarray(obs, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.float32),
        }
        self._episodes.append(episode)
        self._size += len(current)
        self._current[lane] = []
        while self._size > self.capacity and len(self._episodes) > 1:
            dropped = self._episodes.popleft()
            self._size -= len(dropped["dones"])

    def __len__(self) -> int:
        return self._size

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    def sample(self, batch_size: int, seq_len: int) -> dict[str, np.ndarray]:
        """`(B, T, ...)` contiguous windows — an episode shorter than
        `seq_len` gets its very last real step repeated for the remainder
        (rather than zero-padded) so the GRU/LSTM never sees a discontinuous
        jump partway through a window; `dones=1` for every padded step
        either way, so the continue-head's own training target still says
        exactly what actually happened (the episode really did end there)."""
        episodes = list(self._episodes)
        obs = np.zeros((batch_size, seq_len, *self.obs_shape), dtype=np.float32)
        actions = np.zeros((batch_size, seq_len, self.action_dim), dtype=np.float32)
        rewards = np.zeros((batch_size, seq_len), dtype=np.float32)
        dones = np.ones((batch_size, seq_len), dtype=np.float32)
        for b in range(batch_size):
            ep = episodes[np.random.randint(len(episodes))]
            ep_len = len(ep["dones"])
            window = min(seq_len, ep_len)
            start = np.random.randint(0, ep_len - window + 1)
            obs[b, :window] = ep["obs"][start : start + window]
            actions[b, :window] = ep["actions"][start : start + window]
            rewards[b, :window] = ep["rewards"][start : start + window]
            dones[b, :window] = ep["dones"][start : start + window]
            if window < seq_len:
                obs[b, window:] = ep["obs"][start + window - 1]
        return {"obs": obs, "actions": actions, "rewards": rewards, "dones": dones}


def collect_rollout(
    env: gym.Env,
    action_space: gym.Space,
    num_steps: int,
    policy_fn: Callable[[Any], Any] | None = None,
    sequence_buffer: SequenceReplayBuffer | None = None,
    transition_buffer: Any | None = None,
    seed: int | None = None,
    obs: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Steps `env` for `num_steps` env steps, pushing every transition into
    whichever buffer(s) are given (either, both, or neither — a `None`
    buffer is just skipped). `policy_fn(obs) -> action` defaults to
    uniform-random actions from `action_space` — the standard way both
    this app's own preview rollouts (`rl_core/envs/previews.py`) and the
    original World Models paper collect their first batch of exploration
    data. Pass `obs` (the caller's last observation) back in on the next
    call to continue the same ongoing episode instead of resetting —
    returns `(final_obs, stats)` for exactly that."""
    if obs is None:
        obs, _info = env.reset(seed=seed)
    ep_reward = 0.0
    ep_length = 0
    completed_rewards: list[float] = []
    completed_lengths: list[int] = []

    for _ in range(num_steps):
        action = action_space.sample() if policy_fn is None else policy_fn(obs)
        next_obs, reward, terminated, truncated, _info = env.step(action)
        done = bool(terminated or truncated)
        obs_arr = obs_to_array(obs, env.observation_space)
        action_arr = obs_to_array(action, action_space)
        if sequence_buffer is not None:
            sequence_buffer.add(obs_arr, action_arr, float(reward), done)
        if transition_buffer is not None:
            next_obs_arr = obs_to_array(next_obs, env.observation_space)
            transition_buffer.add(obs_arr, action_arr, float(reward), next_obs_arr, done)
        ep_reward += float(reward)
        ep_length += 1
        if done:
            completed_rewards.append(ep_reward)
            completed_lengths.append(ep_length)
            ep_reward, ep_length = 0.0, 0
            obs, _info = env.reset()
        else:
            obs = next_obs

    stats = {
        "episodes": len(completed_rewards),
        "mean_reward": float(np.mean(completed_rewards)) if completed_rewards else None,
        "mean_length": float(np.mean(completed_lengths)) if completed_lengths else None,
    }
    return obs, stats


def collect_rollout_vec(
    env: Any,
    action_space: gym.Space,
    num_steps_per_lane: int,
    policy_fn: Callable[[Any], Any] | None = None,
    sequence_buffer: SequenceReplayBuffer | None = None,
    transition_buffer: Any | None = None,
    seed: int | None = None,
    obs: list[Any] | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """`collect_rollout`'s `num_envs > 1` counterpart — `env` a
    `gymnasium.vector.VectorEnv` built via
    `rl_core.algorithms.vec_env.make_env_or_vec(..., num_envs=N)` (one
    subprocess worker per lane, each auto-seeded `seed + lane` by
    `vec_reset`). Steps every lane `num_steps_per_lane` times *in
    parallel* — `num_steps_per_lane * N` real transitions collected in
    roughly `1/N` of the wall-clock time plain `collect_rollout` would
    take for the same total transition count, since each lane's env.step
    genuinely runs on its own CPU core. This is the concrete "run several
    environments in parallel with different seeds" lever for standalone
    World Model training (and any other random-policy warm-up phase).

    `sequence_buffer.add(..., lane=i)` keeps each lane's in-progress
    episode from splicing into another lane's (see
    `SequenceReplayBuffer`'s own docstring)."""
    from rl_core.algorithms.vec_env import num_envs_of, obs_space, vec_reset, vec_step

    num_lanes = num_envs_of(env)
    single_obs_space = obs_space(env)
    if obs is None:
        obs = vec_reset(env, seed=seed)
    ep_reward = [0.0] * num_lanes
    ep_length = [0] * num_lanes
    completed_rewards: list[float] = []
    completed_lengths: list[int] = []

    for _ in range(num_steps_per_lane):
        actions = [action_space.sample() if policy_fn is None else policy_fn(obs[i]) for i in range(num_lanes)]
        next_obs, rewards, terminated, truncated, _infos = vec_step(env, actions)
        for i in range(num_lanes):
            done = bool(terminated[i] or truncated[i])
            obs_arr = obs_to_array(obs[i], single_obs_space)
            action_arr = obs_to_array(actions[i], action_space)
            if sequence_buffer is not None:
                sequence_buffer.add(obs_arr, action_arr, float(rewards[i]), done, lane=i)
            if transition_buffer is not None:
                next_obs_arr = obs_to_array(next_obs[i], single_obs_space)
                transition_buffer.add(obs_arr, action_arr, float(rewards[i]), next_obs_arr, done)
            ep_reward[i] += float(rewards[i])
            ep_length[i] += 1
            if done:
                completed_rewards.append(ep_reward[i])
                completed_lengths.append(ep_length[i])
                ep_reward[i], ep_length[i] = 0.0, 0
        obs = next_obs

    stats = {
        "episodes": len(completed_rewards),
        "mean_reward": float(np.mean(completed_rewards)) if completed_rewards else None,
        "mean_length": float(np.mean(completed_lengths)) if completed_lengths else None,
    }
    return obs, stats


def collect_rollout_auto(
    env: Any,
    action_space: gym.Space,
    num_steps: int,
    policy_fn: Callable[[Any], Any] | None = None,
    sequence_buffer: SequenceReplayBuffer | None = None,
    transition_buffer: Any | None = None,
    seed: int | None = None,
    obs: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Dispatches to `collect_rollout` or `collect_rollout_vec` depending
    on whether `env` is a plain `gym.Env` or a `VectorEnv` — one call site
    that works either way, so callers never need their own
    `is_vector_env` branch. `num_steps` is always the *total* transition
    budget across every lane (divided evenly per lane in the vectorized
    case), matching `collect_rollout`'s own single-env meaning exactly, so
    a caller's `collect_steps_per_iter`-style hyperparam means the same
    thing regardless of `training.num_envs`."""
    from rl_core.algorithms.vec_env import is_vector_env, num_envs_of

    if not is_vector_env(env):
        return collect_rollout(env, action_space, num_steps, policy_fn, sequence_buffer, transition_buffer, seed, obs)
    num_lanes = num_envs_of(env)
    steps_per_lane = max(1, num_steps // num_lanes)
    return collect_rollout_vec(env, action_space, steps_per_lane, policy_fn, sequence_buffer, transition_buffer, seed, obs)


def action_flat_dim(action_space: gym.Space) -> int:
    return obs_flat_dim(action_space)
