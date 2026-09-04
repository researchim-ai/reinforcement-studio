"""Uniform "batch of N envs" view over either a plain `gymnasium.Env` or a
`gymnasium.vector.VectorEnv`, so every native algorithm's collection loop
can be written once for `num_envs >= 1` instead of twice.

Deliberately does *not* always wrap a single env in a vector env (even
though that would look more "unified") — `CustomAlgorithm.env` is a
documented part of the plugin contract (see `rl_core/algorithms/base.py`
and the templates on the Plugins page), and user-authored plugins built
against a plain `env.observation_space`/`env.action_space` would silently
break if those turned into their *batched* vector-space equivalents just
because `num_envs` defaults to 1 somewhere upstream. So: `num_envs=1` keeps
`env` a plain `gym.Env`, exactly as before — zero behavior change for
existing plugins/tests/saved configs. `num_envs>1` wraps it in an
`AsyncVectorEnv` (one subprocess worker per lane, true multi-core
simulation). Either way, the helpers below (`obs_space`/`action_space`/
`vec_reset`/`vec_step`) give built-in algorithms' `learn()` loops one
consistent "list/array of N" shape to work with.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv, VectorEnv


def _vector_env_context() -> str:
    """Always spawn — fork after PyTorch/CUDA init (every training run) can
    deadlock worker processes; spawn is also required on Windows/macOS."""
    return "spawn"


def _make_registered_gym_env(env_id: str) -> gym.Env:
    """Build a gym env inside an `AsyncVectorEnv` worker — re-registers
    custom envs (POMDP, MiniGrid, Highway-env, NetHack/MiniHack, robotics
    mazes, industrial, trading, Atari, ...) that only exist in the parent
    process when using the spawn start method."""
    from rl_core.envs.pomdp import register_pomdp_envs

    register_pomdp_envs()
    try:
        from rl_core.envs.minigrid_envs import register_minigrid_envs

        register_minigrid_envs()
    except ImportError:
        pass
    try:
        from rl_core.envs.highway_envs import register_highway_envs

        register_highway_envs()
    except ImportError:
        pass
    try:
        from rl_core.envs.nethack_envs import register_nethack_envs

        register_nethack_envs()
    except ImportError:
        pass
    try:
        from rl_core.envs.robotics_envs import register_robotics_envs

        register_robotics_envs()
    except ImportError:
        pass
    from rl_core.envs.finrl_envs import register_finrl_envs
    from rl_core.envs.industrial_envs import register_industrial_envs
    from rl_core.envs.trading_envs import register_trading_envs

    register_industrial_envs()
    register_trading_envs()
    register_finrl_envs()
    try:
        import ale_py

        gym.register_envs(ale_py)
    except ImportError:
        pass
    return gym.make(env_id)


def make_gym_env_factory(env_id: str) -> Callable[[], gym.Env]:
    """Picklable zero-arg factory for tests and plugins — safe in subprocess
    workers unlike a bare `lambda: gym.make(...)`."""
    from functools import partial

    return partial(_make_registered_gym_env, env_id)


def make_env_or_vec(
    env_factory: Callable[[], gym.Env],
    num_envs: int,
    seed: int | None = None,
    *,
    parallel: bool = True,
) -> gym.Env | VectorEnv:
    """`env_factory` is a zero-arg callable building one fresh env (must be
    picklable when `parallel=True` — use a module-level function with
    `functools.partial`, not a closure/lambda, for Windows/Electron). Called
    once for `num_envs=1`, `num_envs` times (independent instances in
    separate worker processes) otherwise."""
    del seed  # callers pass seeds through `vec_reset`, not here
    num_envs = max(1, int(num_envs))
    if num_envs == 1:
        return env_factory()
    env_fns = [env_factory for _ in range(num_envs)]
    if parallel:
        return AsyncVectorEnv(
            env_fns, context=_vector_env_context(), autoreset_mode=AutoresetMode.SAME_STEP,
        )
    return SyncVectorEnv(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)


def is_vector_env(env: Any) -> bool:
    return isinstance(env, VectorEnv)


def num_envs_of(env: Any) -> int:
    return int(env.num_envs) if is_vector_env(env) else 1


def obs_space(env: Any) -> gym.Space:
    """The *single* (unbatched) observation space regardless of whether
    `env` is a plain env or a `VectorEnv` — this is what network input
    dims/`obs_to_array` must always be built from, never the vector env's
    own (batched) `observation_space`."""
    return env.single_observation_space if is_vector_env(env) else env.observation_space


def action_space(env: Any) -> gym.Space:
    """See `obs_space` — the single (unbatched) action space."""
    return env.single_action_space if is_vector_env(env) else env.action_space


def _split_vector_infos(infos: dict[str, Any], num_envs: int) -> list[dict[str, Any]]:
    """`VectorEnv.step`'s `infos` is a dict-of-arrays (one array per key
    that *any* lane's info dict had), not a list of per-lane dicts — and
    any key whose per-lane value is itself a dict (e.g. `Monitor`'s
    `"episode"` key, present only on the step a lane's episode ends) gets
    aggregated the same way, recursively, rather than turned into a plain
    array. Each key also gets a same-shape `f"_{key}"` boolean mask
    alongside it marking which lanes actually had that key this step (the
    rest hold an arbitrary filler value that must not leak into a lane
    that never reported it). Recurses through that nested structure to
    rebuild the length-`num_envs` list of plain per-lane dicts every
    caller in this app actually wants."""
    result: list[dict[str, Any]] = [{} for _ in range(num_envs)]
    for key, value in infos.items():
        if key.startswith("_"):
            continue
        mask = infos.get(f"_{key}")
        if isinstance(value, dict):
            per_lane = _split_vector_infos(value, num_envs)
        else:
            per_lane = list(value)
        for i in range(num_envs):
            if mask is None or mask[i]:
                result[i][key] = per_lane[i]
    return result


def vec_reset(env: Any, seed: int | None = None) -> list[Any]:
    """Returns a length-`num_envs_of(env)` list of raw (non-batched, one
    per lane) observations — same shape of thing whether `env` is a plain
    env (`[obs]`) or a `VectorEnv` (its batched array/tuple, unpacked along
    axis 0). Every lane gets a distinct seed (`seed + i`) so `num_envs>1`
    doesn't just replay `num_envs` identical trajectories."""
    if is_vector_env(env):
        seeds = [seed + i for i in range(env.num_envs)] if seed is not None else None
        obs, _info = env.reset(seed=seeds)
        return list(obs)
    obs, _info = env.reset(seed=seed)
    return [obs]


def vec_step(
    env: Any, actions: Sequence[Any],
) -> tuple[list[Any], np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """`actions`: length-`num_envs_of(env)` sequence of *unbatched* per-lane
    actions (whatever `action_to_env(...)` produces for the single action
    space — a plain `int` for `Discrete`, a shaped `np.ndarray` for `Box`).
    Returns `(obs_list, rewards, terminated, truncated, infos)`, every one
    of them length-N — `obs_list` a plain list (see `vec_reset`), the rest
    numpy arrays, `infos` a length-N list of per-lane dicts (not
    `VectorEnv`'s own dict-of-arrays shape, so callers never need to
    branch on vector-vs-not to read a lane's info).

    Vector envs are constructed with `SAME_STEP` autoreset: a terminal
    call returns the freshly reset observation for the next policy call,
    while the real terminal observation/info are preserved as
    `info["final_obs"]`/`info["final_info"]`. This avoids NEXT_STEP's
    otherwise unavoidable dummy iteration where the supplied action is
    ignored and callers accidentally record `terminal -> reset` as a
    zero-reward transition. Plain envs are normalized to the same
    contract below.
    """
    if is_vector_env(env):
        obs, rewards, terminated, truncated, infos = env.step(np.asarray(actions))
        num_envs = env.num_envs
        per_lane_infos = _split_vector_infos(infos, num_envs)
        # SAME_STEP nests the terminal env's info under `final_info` and
        # uses the top level for reset info. Preserve the explicit nested
        # field, but also expose terminal keys at the top level to retain
        # vec_step's historical per-lane info contract (notably Monitor's
        # `episode` entry).
        for lane_info in per_lane_infos:
            final_info = lane_info.get("final_info")
            if isinstance(final_info, dict):
                for key, value in final_info.items():
                    lane_info.setdefault(key, value)
        return (
            list(obs),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminated, dtype=bool),
            np.asarray(truncated, dtype=bool),
            per_lane_infos,
        )
    obs, reward, terminated, truncated, info = env.step(actions[0])
    if terminated or truncated:
        final_obs, final_info = obs, info
        obs, reset_info = env.reset()
        info = {**reset_info, **final_info, "final_obs": final_obs, "final_info": final_info}
    return (
        [obs],
        np.array([reward], dtype=np.float32),
        np.array([terminated], dtype=bool),
        np.array([truncated], dtype=bool),
        [info],
    )
