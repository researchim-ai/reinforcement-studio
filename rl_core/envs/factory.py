"""Central env construction — gym.make for registered envs, SceneMultiAgentEnv
for user-built scenes (`scene:{slug}`), PettingZooVectorEnv for the built-in
PettingZoo multi-agent benchmarks (`petting:{slug}`), and TupleMarlVectorEnv
for the built-in Tuple-space MARL benchmarks — RWARE/LBForaging
(`marlgym:{slug}`, see `rl_core/envs/tuple_marl_envs.py`)."""
from __future__ import annotations

from typing import Any

import gymnasium as gym

from rl_core import scene_store
from rl_core.envs.pettingzoo_envs import is_pettingzoo_env_id, make_pettingzoo_env, pettingzoo_slug_from_env_id
from rl_core.envs.scene_env import SceneMultiAgentEnv, SceneRenderEnv, make_scene_env
from rl_core.envs.tuple_marl_envs import is_tuple_marl_env_id, make_tuple_marl_env, tuple_marl_slug_from_env_id
from rl_core.envs.wrappers import apply_wrappers


def make_training_env(
    env_id: str,
    wrapper_specs: list[dict] | None = None,
    *,
    render: bool = False,
) -> Any:
    """Build the env a training run uses. Scene/PettingZoo/tuple-MARL envs
    skip wrappers (MVP) — all three are already-shared multi-agent
    worlds, not a single-agent pipeline the wrapper catalog (frame
    stacking, reward clipping, ...) was designed against."""
    if scene_store.is_scene_env_id(env_id):
        slug = scene_store.scene_slug_from_env_id(env_id)
        if render:
            return make_scene_env(slug, render_mode="rgb_array", for_gif=True)
        return make_scene_env(slug, render_mode=None, for_gif=False)

    if is_pettingzoo_env_id(env_id):
        slug = pettingzoo_slug_from_env_id(env_id)
        if render:
            return make_pettingzoo_env(slug, render_mode="rgb_array", for_gif=True)
        return make_pettingzoo_env(slug, render_mode=None, for_gif=False)

    if is_tuple_marl_env_id(env_id):
        slug = tuple_marl_slug_from_env_id(env_id)
        if render:
            return make_tuple_marl_env(slug, render_mode="rgb_array", for_gif=True)
        return make_tuple_marl_env(slug, render_mode=None, for_gif=False)

    env = gym.make(env_id, render_mode="rgb_array" if render else None)
    return apply_wrappers(env, wrapper_specs or [])


def is_shared_world_env_id(env_id: str) -> bool:
    """Multi-agent envs where `training.num_envs` means "lanes in one
    shared world" (agent count fixed by the env itself), not "N
    independent env copies" — Scene Builder scenes and the built-in
    PettingZoo/tuple-MARL benchmarks alike. See `run_custom_algorithm` in
    `rl_core/algorithms/runner_utils.py`, the only caller that needs this
    distinction."""
    return (
        scene_store.is_scene_env_id(env_id)
        or is_pettingzoo_env_id(env_id)
        or is_tuple_marl_env_id(env_id)
    )


def make_inspect_env(env_id: str, wrapper_specs: list[dict] | None = None) -> Any:
    if scene_store.is_scene_env_id(env_id):
        slug = scene_store.scene_slug_from_env_id(env_id)
        return SceneMultiAgentEnv(scene_store.load(slug))
    if is_pettingzoo_env_id(env_id):
        slug = pettingzoo_slug_from_env_id(env_id)
        return make_pettingzoo_env(slug)
    if is_tuple_marl_env_id(env_id):
        slug = tuple_marl_slug_from_env_id(env_id)
        return make_tuple_marl_env(slug)
    env = gym.make(env_id)
    return apply_wrappers(env, wrapper_specs or [])
