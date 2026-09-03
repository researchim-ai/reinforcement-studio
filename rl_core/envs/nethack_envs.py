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


_RENDER_TILE_PX = 16  # `nle.nethack.TILE_X`/`TILE_Y` - not imported just to keep this module import-light
_RENDER_VIEWPORT_COLS = 26  # tiles - see `_crop_pixel_frame_to_viewport`'s docstring


def _crop_pixel_frame_to_viewport(frame: np.ndarray, base_env: gym.Env) -> np.ndarray:
    """NLE's own `render_mode="pixel"` always draws the *entire* dungeon
    screen - `nethack.TILE_RENDER_SHAPE`, `(336, 1264, 3)` = 21x79 tiles -
    regardless of how much of the level the player has actually explored.
    That's correct (an unexplored dungeon tile genuinely *is* black, same
    as a human playing would see), but it means an un-cropped frame is
    routinely 80%+ dead black space around whatever small room the player
    happens to be standing in right now - which is exactly what makes the
    live-preview GIF look "all black" at a glance (worse still once
    `_build_gif_bytes` thumbnails the whole wide-and-mostly-empty frame
    down to `_GIF_MAX_SIDE`). Cropping to a fixed-width window centered on
    the player's current column (from `blstats`, the same field NLE's own
    task reward functions - e.g. `NetHackStaircasePet._is_episode_end`, see
    `nle/env/tasks.py` - already read player position from) fixes that
    without needing to know anything about the *current* room's actual
    shape. Height is left uncropped (21 rows is already a reasonable size,
    and vertically centering too would need the same near-edge clamping
    for comparatively little benefit - NetHack's dungeon is far wider than
    it is tall, so the width is where nearly all the dead space comes
    from)."""
    try:
        base = base_env.unwrapped
        blstats = base.last_observation[base._blstats_index]
        player_col = int(blstats[0])  # nethack.NLE_BL_X
    except Exception:
        return frame
    total_cols = frame.shape[1] // _RENDER_TILE_PX
    width = min(_RENDER_VIEWPORT_COLS, total_cols)
    start_col = max(0, min(player_col - width // 2, total_cols - width))
    px = start_col * _RENDER_TILE_PX
    return frame[:, px : px + width * _RENDER_TILE_PX]


class _CharObs(gym.ObservationWrapper):
    """Dict obs -> single-channel uint8 image built from one `obs[key]`
    grid. `key` is already byte-valued (0-255 ASCII/curses codes), so no
    rescaling is needed before NatureCNN/SmallCNN's own `/255.0` step.

    Also crops the live-preview `render(mode="pixel")` frame down to a
    viewport around the player (`_crop_pixel_frame_to_viewport` above) -
    piggybacking on this wrapper rather than adding a dedicated one since
    it's already the single choke point both the NetHack and MiniHack
    entry points route through below."""

    def __init__(self, env: gym.Env, key: str) -> None:
        super().__init__(env)
        self._key = key
        h, w = env.observation_space[key].shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(h, w, 1), dtype=np.uint8)

    def observation(self, observation: dict) -> np.ndarray:
        return observation[self._key][:, :, None].astype(np.uint8)

    def render(self) -> Any:
        frame = self.env.render()
        if self.render_mode == "pixel" and isinstance(frame, np.ndarray):
            return _crop_pixel_frame_to_viewport(frame, self.env)
        return frame


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
