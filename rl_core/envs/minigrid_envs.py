"""MiniGrid (Chevalier-Boisvert et al., 2018-2023) gridworld environments —
partially-observable navigation/exploration/puzzle tasks on a small grid,
registered under this app's own `*-Img-v0` ids with `ImgObsWrapper`
pre-applied.

MiniGrid's raw envs return `Dict(image=Box(7,7,3) uint8, direction=
Discrete(4), mission=MissionSpace)` — the `mission` field is a natural-
language instruction string that neither the native algorithms
(`rl_core/algorithms/native/preprocessing.py`, which only understands
`Discrete`/`Box`) nor SB3's built-in policies can consume at all, so unlike
every other Dict/Tuple obs in this app it can't just be flattened away (see
`sb3_runner.py`'s `FlattenObservation` fallback) — it has to be dropped
explicitly. `ImgObsWrapper` keeps just the `image` field: a 7x7x3 uint8
egocentric view (the agent's limited field of view, not the full map),
already exactly what `is_image_space()`/`SmallCNN` in networks.py expect,
and also the standard way MiniGrid's own README trains every baseline —
losing `direction` doesn't hurt much in practice since it's already
implicit in how the visible cells shift frame to frame.

Deliberately egocentric-image-only (no `FlatObsWrapper` variant): partial
observability from a genuinely limited field of view is exactly the kind of
task `rl_core/envs/pomdp.py`'s `VisualMemoryMazeEnv` was already added to
cover, and `MiniGrid-MemoryS9-v0` below is literally the flagship benchmark
the recurrent (LSTM/GRU) algorithms in this app's Network Builder are meant
to be tried against.

Registered as ordinary gymnasium env ids via `gym.register(...)`, exactly
like `rl_core/envs/pomdp.py`'s custom envs — works anywhere a Gym id
already does (native/SB3 runners, live GIF preview, `/environments/inspect`).
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym

_REGISTERED = False

# base MiniGrid id -> our wrapped id (kept as a lookup rather than a plain
# string replace so unrelated future MiniGrid ids without a trailing
# "-v0" never silently produce a malformed wrapped id).
_MINIGRID_BASE_IDS = [
    "MiniGrid-Empty-8x8-v0",
    "MiniGrid-FourRooms-v0",
    "MiniGrid-DoorKey-8x8-v0",
    "MiniGrid-LavaCrossingS9N1-v0",
    "MiniGrid-Dynamic-Obstacles-8x8-v0",
    "MiniGrid-MultiRoom-N4-S5-v0",
    "MiniGrid-MemoryS9-v0",
    "MiniGrid-KeyCorridorS3R1-v0",
]


def _make_wrapped_minigrid(base_id: str, **kwargs: Any) -> gym.Env:
    import minigrid  # noqa: F401  (side-effect: registers MiniGrid-* base ids)
    from minigrid.wrappers import ImgObsWrapper

    return ImgObsWrapper(gym.make(base_id, **kwargs))


def _entry_point_for(base_id: str):
    def _entry(**kwargs: Any) -> gym.Env:
        return _make_wrapped_minigrid(base_id, **kwargs)
    return _entry


def wrapped_id_for(base_id: str) -> str:
    return base_id.replace("-v0", "-Img-v0")


def register_minigrid_envs() -> None:
    """Idempotent — safe to import/call from multiple modules. Registering
    the wrapped ids never itself needs `minigrid` installed (the entry
    point only imports it lazily, the first time one of these ids is
    actually `gym.make()`d) — `_gym_available()` in registry.py is what
    decides whether these show as greyed-out in the gallery when it
    isn't."""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True
    for base_id in _MINIGRID_BASE_IDS:
        gym.register(id=wrapped_id_for(base_id), entry_point=_entry_point_for(base_id))


# Self-registers at import time, exactly like `rl_core/envs/pomdp.py` — so
# any module that merely imports this one (even just for the
# `register_minigrid_envs` name) gets the ids registered for free, the same
# guarantee `pomdp.py`'s custom envs already have.
register_minigrid_envs()
