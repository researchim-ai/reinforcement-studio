"""Highway-env (Leurent, 2018) autonomous-driving decision-making tasks.

Most of highway-env's scenarios (`highway-v0`, `merge-v1`, `roundabout-v1`,
`intersection-v1`, ...) already default to a flat-enough observation —
`Box(N, 5)` "Kinematics" (N nearby vehicles x 5 features each) or similar —
that both the native algorithms and SB3 handle out of the box exactly like
any other vector `Box` observation (see `is_image_space()`/`obs_flat_dim()`
in `rl_core/algorithms/native/preprocessing.py`: a 2D `Box` that isn't
image-shaped just gets flattened), so those are registered directly under
highway-env's own ids in `rl_core/envs/registry.py` — no wrapper needed.

`parking-v0` is the one exception: it's goal-conditioned, so its raw
observation is `Dict(observation=Box(6,), achieved_goal=Box(6,),
desired_goal=Box(6,))` — three plain `Box`es, trivially flattenable (unlike
MiniGrid's `mission` field, a `Text`-like space nothing here can flatten at
all — see `rl_core/envs/minigrid_envs.py`), just not automatically the way
a single top-level `Box` observation is. Registered below under our own
`parking-Flat-v0` id with `gymnasium.wrappers.FlattenObservation` pre-
applied, so it "just works" the same as everything else in the gallery
(training still optimizes the env's own dense/sparse reward as normal —
flattening only affects how the *observation* is shaped, not what's being
optimized).
"""
from __future__ import annotations

import gymnasium as gym

_REGISTERED = False


def _make_flat_parking(**kwargs) -> gym.Env:
    import highway_env  # noqa: F401  (side-effect: registers highway-env's own ids)

    return gym.wrappers.FlattenObservation(gym.make("parking-v0", **kwargs))


def register_highway_envs() -> None:
    """Idempotent — safe to import/call from multiple modules, even when
    `highway_env` isn't installed (a plain no-op then, exactly like every
    other optional-extra env in `rl_core/envs/registry.py` — `_gym_available()`
    is what actually decides whether these show as greyed-out in the
    gallery). Importing `highway_env` here also registers its own ids
    (`highway-v0`/`merge-v1`/`roundabout-v1`/`intersection-v1`/...), so
    callers only need this one function to get every highway-env id used
    in `rl_core/envs/registry.py` working."""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True
    try:
        import highway_env  # noqa: F401
    except ImportError:
        return
    gym.register(id="parking-Flat-v0", entry_point=_make_flat_parking)


# Self-registers at import time, exactly like `rl_core/envs/pomdp.py` — so
# any module that merely imports this one (even just for the
# `register_highway_envs` name) gets the ids registered for free, the same
# guarantee `pomdp.py`'s custom envs already have.
register_highway_envs()
