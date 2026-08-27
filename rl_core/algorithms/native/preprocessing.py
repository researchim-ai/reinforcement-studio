"""Turns raw Gym observations into plain numpy arrays our networks can batch.

Stable-Baselines3 does this internally (one-hot for `Discrete`, dtype casts,
...); since the native algorithms don't depend on SB3 at all, we own this
step ourselves. Kept tiny on purpose — only the observation space shapes
that actually show up in our environment catalog need to be handled (see
`rl_core/envs/registry.py`): scalar `Discrete` (a few Toy Text envs), vector
`Box` (classic control / Box2D / MuJoCo), and image `Box` (Atari / CarRacing,
already `Tuple`/`Dict`-flattened by `sb3_runner._make_env` beforehand).
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym


def is_image_space(space: gym.Space) -> bool:
    """(H, W, C) or (C, H, W) uint8-ish Box — treat as a CNN input."""
    return (
        isinstance(space, gym.spaces.Box)
        and len(space.shape) == 3
        and min(space.shape) <= 4
    )


def obs_flat_dim(space: gym.Space) -> int:
    """Flat feature count fed to an MLP (ignored for image spaces)."""
    if isinstance(space, gym.spaces.Discrete):
        return int(space.n)
    return int(np.prod(space.shape))


def obs_to_array(obs, space: gym.Space) -> np.ndarray:
    """Single (non-batched) observation -> numpy array ready for stacking.

    - `Discrete` -> one-hot float32 vector (SB3 does the same by default).
    - Image `Box` -> kept as (H, W, C) uint8/float32, normalized in the CNN.
    - Everything else (vector `Box`) -> flat float32 vector.
    """
    if isinstance(space, gym.spaces.Discrete):
        vec = np.zeros(int(space.n), dtype=np.float32)
        vec[int(obs)] = 1.0
        return vec
    arr = np.asarray(obs)
    if is_image_space(space):
        return arr.astype(np.float32)
    return arr.reshape(-1).astype(np.float32)


def obs_batch_to_array(obs_list: list, space: gym.Space) -> np.ndarray:
    """`obs_list`: a length-N list of raw (non-batched) per-lane
    observations, e.g. from `rl_core.algorithms.vec_env.vec_reset`/
    `vec_step` — stacks `obs_to_array(...)` over each into one `(N, ...)`
    float32 array. `N=1` (the `num_envs=1` default) degenerates to a
    length-1 batch of exactly what `obs_to_array` alone would return."""
    return np.stack([obs_to_array(o, space) for o in obs_list])


def action_to_env(action: np.ndarray, action_space: gym.Space):
    """Native nets always produce a small numpy array per action; env.step
    wants a Python int for `Discrete`, a properly-shaped/clamped array for
    `Box`."""
    if isinstance(action_space, gym.spaces.Discrete):
        return int(np.asarray(action).reshape(-1)[0])
    low, high = action_space.low, action_space.high
    return np.clip(np.asarray(action, dtype=np.float32).reshape(action_space.shape), low, high)
