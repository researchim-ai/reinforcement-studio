"""Central env construction — gym.make for registered envs, SceneMultiAgentEnv
for user-built scenes (`scene:{slug}`)."""
from __future__ import annotations

from typing import Any

import gymnasium as gym

from rl_core import scene_store
from rl_core.envs.scene_env import SceneMultiAgentEnv, SceneRenderEnv, make_scene_env
from rl_core.envs.wrappers import apply_wrappers


def make_training_env(
    env_id: str,
    wrapper_specs: list[dict] | None = None,
    *,
    render: bool = False,
) -> gym.Env | SceneMultiAgentEnv:
    """Build the env a training run uses. Scene envs skip wrappers (MVP)."""
    if scene_store.is_scene_env_id(env_id):
        slug = scene_store.scene_slug_from_env_id(env_id)
        if render:
            return make_scene_env(slug, render_mode="rgb_array", for_gif=True)
        return make_scene_env(slug, render_mode=None, for_gif=False)

    env = gym.make(env_id, render_mode="rgb_array" if render else None)
    return apply_wrappers(env, wrapper_specs or [])


def make_inspect_env(env_id: str, wrapper_specs: list[dict] | None = None) -> gym.Env | SceneMultiAgentEnv:
    if scene_store.is_scene_env_id(env_id):
        slug = scene_store.scene_slug_from_env_id(env_id)
        return SceneMultiAgentEnv(scene_store.load(slug))
    env = gym.make(env_id)
    return apply_wrappers(env, wrapper_specs or [])
