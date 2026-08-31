"""NetHack Learning Environment (NLE, Küttler et al., 2020) and MiniHack
(Samvelyan et al., 2021) — procedurally-varied, hard-exploration dungeon-
crawling tasks, registered under this app's own `*-Img-v0` ids with the raw
Dict observation flattened down to a single-channel image.

Both packages return `Dict(glyphs=Box(21,79) int16, chars=Box(21,79) uint8,
colors=..., blstats=Box(27,) int64, message=..., ...)` (MiniHack additionally
exposes `*_crop` egocentric-crop variants). Like MiniGrid's `mission` field,
none of this is directly consumable by the native algorithms
(`rl_core/algorithms/native/preprocessing.py` only understands `Discrete`/
`Box`) — but unlike MiniGrid there's no single field that's already "the
image" we want: `glyphs` ranges up to 5976 (not a byte), `chars` is the
ASCII code drawn on the terminal cell (0-255, already the exact range
NatureCNN/SmallCNN's `/255.0` normalization expects) and is what NLE's own
baseline agent trains against, so `_CharObs` below extracts just that field
and reshapes it to a single-channel `(H, W, 1)` uint8 image. MiniHack's
`chars_crop` is preferred where available — a small (9x9 by default)
egocentric crop rather than the full 21x79 dungeon, matching MiniGrid's
egocentric-view convention and keeping `SmallCNN` (tuned for <50px inputs)
the right conv stack instead of needing the bigger `NatureCNN`.

Registered as ordinary gymnasium env ids via `gym.register(...)`, exactly
like `rl_core/envs/minigrid_envs.py` and `rl_core/envs/pomdp.py` — works
anywhere a Gym id already does (native/SB3 runners, live GIF preview,
`/environments/inspect`).
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

_REGISTERED = False

# (base id registered by nle/minihack, our wrapped id, obs field to use as
# the image). Kept as an explicit table rather than a suffix-strip helper
# so unrelated future ids never silently produce a malformed wrapped id.
_NETHACK_BASE_IDS: list[tuple[str, str, str]] = [
    ("NetHackScore-v0", "NetHackScore-Img-v0", "chars"),
    ("NetHackEat-v0", "NetHackEat-Img-v0", "chars"),
]

# MiniHack tasks all default to a `chars_crop` observation key once the
# `MH_AUTOCROP`... no such flag is needed — `chars_crop` is present on every
# MiniHack env out of the box (it's the crop *around* the full `chars`).
_MINIHACK_BASE_IDS: list[tuple[str, str, str]] = [
    ("MiniHack-Room-5x5-v0", "MiniHack-Room-5x5-Img-v0", "chars_crop"),
    ("MiniHack-Room-15x15-v0", "MiniHack-Room-15x15-Img-v0", "chars_crop"),
    ("MiniHack-Corridor-R2-v0", "MiniHack-Corridor-R2-Img-v0", "chars_crop"),
    ("MiniHack-MazeWalk-9x9-v0", "MiniHack-MazeWalk-9x9-Img-v0", "chars_crop"),
    ("MiniHack-KeyRoom-S5-v0", "MiniHack-KeyRoom-5x5-Img-v0", "chars_crop"),
    ("MiniHack-Eat-v0", "MiniHack-Eat-Img-v0", "chars_crop"),
]


class _CharObs(gym.ObservationWrapper):
    """Dict obs -> single-channel uint8 image built from one `obs[key]`
    grid. `key` is already byte-valued (0-255 ASCII/curses codes), so no
    rescaling is needed before NatureCNN/SmallCNN's own `/255.0` step."""

    def __init__(self, env: gym.Env, key: str) -> None:
        super().__init__(env)
        self._key = key
        h, w = env.observation_space[key].shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(h, w, 1), dtype=np.uint8)

    def observation(self, observation: dict) -> np.ndarray:
        return observation[self._key][:, :, None].astype(np.uint8)


def _rgb_array_to_pixel(kwargs: dict[str, Any]) -> dict[str, Any]:
    """NLE/MiniHack's own `render_modes` are `{"human", "ansi", "full",
    "pixel"}` — "pixel" (a font-rendered screenshot of the terminal, not
    the `chars`/`glyphs` observation) is the one that actually returns an
    RGB array from `render()`, unlike every other env id in this app which
    speaks Gymnasium's usual `"rgb_array"`. Translated here, once, rather
    than asking every caller that constructs envs generically (the live
    training GIF preview in `metrics_callback.py`, the Environments
    gallery's local-rollout preview in `previews.py`, ...) to special-case
    this one family of envs. Harmless if `render_mode` is absent/anything
    else — passed through unchanged."""
    if kwargs.get("render_mode") == "rgb_array":
        kwargs = {**kwargs, "render_mode": "pixel"}
    return kwargs


def _make_wrapped_nethack(base_id: str, key: str, **kwargs: Any) -> gym.Env:
    import nle  # noqa: F401  (side-effect: registers NetHack*-v0 base ids)

    return _CharObs(gym.make(base_id, **_rgb_array_to_pixel(kwargs)), key=key)


def _make_wrapped_minihack(base_id: str, key: str, **kwargs: Any) -> gym.Env:
    import minihack  # noqa: F401  (side-effect: registers MiniHack-*-v0 base ids)

    return _CharObs(gym.make(base_id, **_rgb_array_to_pixel(kwargs)), key=key)


def _entry_point_for(maker, base_id: str, key: str):
    def _entry(**kwargs: Any) -> gym.Env:
        return maker(base_id, key, **kwargs)
    return _entry


def register_nethack_envs() -> None:
    """Idempotent — safe to import/call from multiple modules. Like
    `register_minigrid_envs`, registering the wrapped ids never itself
    needs `nle`/`minihack` installed (the entry point only imports them
    lazily, the first time one of these ids is actually `gym.make()`d) —
    `_gym_available()` in registry.py decides whether they show as
    greyed-out in the gallery when the packages aren't there."""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True
    for base_id, img_id, key in _NETHACK_BASE_IDS:
        gym.register(id=img_id, entry_point=_entry_point_for(_make_wrapped_nethack, base_id, key))
    for base_id, img_id, key in _MINIHACK_BASE_IDS:
        gym.register(id=img_id, entry_point=_entry_point_for(_make_wrapped_minihack, base_id, key))


# Self-registers at import time, exactly like `minigrid_envs.py`/`pomdp.py`
# — any module that merely imports this one gets the ids registered for
# free.
register_nethack_envs()
