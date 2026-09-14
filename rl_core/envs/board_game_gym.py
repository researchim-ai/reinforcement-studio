"""Gymnasium view of `BoardGame` for MuZero-family self-play.

AlphaZero talks to `BoardGame` directly (`rl_core/alphazero/self_play.py`).
EfficientZero / UniZero / ResearchImZero / LatentImZero speak Gym:
observation in, discrete action out, `info["action_mask"]` for legal
moves. This wrapper is that bridge, with the MuZero two-player contract:

- observation is always from the player to move (`BoardGame.encode()`);
- each `step` is one ply of the *same* policy sitting in both seats;
- reward is +1/0/−1 at the end; chess additionally emits normalized
  mover-relative material deltas so sparse draw-heavy self-play can learn;
- `info["two_player"]` tells the Zero algorithms to backup with ``-γ``.

Registered under the same ids as the AlphaZero catalog (`tic_tac_toe`,
`chess`, ...) so the Designer can keep one card per game and only switch
the run `kind` when a Gym-track Zero algorithm is selected.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rl_core.games import GAME_REGISTRY, make_game

_REGISTERED = False

_MAX_EPISODE_STEPS = {
    "tic_tac_toe": 9,
    "connect_four": 42,
    "gomoku": 81,
    "chess": 512,
    "go_9x9": 162,
}


def _encode_hwc(enc: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """`BoardGame.encode()` is channel-first `(C, H, W)`; Gym image nets
    here expect channel-last `(H, W, C)` uint8 (see `SmallCNN`)."""
    arr = np.asarray(enc, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"board encode must be 3D, got {arr.shape}")
    if arr.shape[0] == rows and arr.shape[1] == cols:
        hwc = arr
    elif arr.shape[1] == rows and arr.shape[2] == cols:
        hwc = np.transpose(arr, (1, 2, 0))
    else:
        hwc = np.transpose(arr, (1, 2, 0))
    return np.clip(np.round(hwc * 255.0), 0, 255).astype(np.uint8)


_CHESS_LETTERS = {1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K"}
_LIGHT_SQ = (232, 213, 181)
_DARK_SQ = (181, 136, 99)
_WHITE_TOKEN = (244, 241, 234)
_BLACK_TOKEN = (32, 30, 28)
_WHITE_INK = (36, 32, 28)
_BLACK_INK = (244, 241, 234)
_CHESS_MATERIAL_VALUES = np.asarray([0, 1, 3, 3, 5, 9, 0], dtype=np.float32)
_CHESS_TOTAL_MATERIAL = 39.0


def _chess_material_for(board: np.ndarray, player: int) -> float:
    pieces = np.asarray(board, dtype=np.int32)
    values = _CHESS_MATERIAL_VALUES[np.clip(np.abs(pieces), 0, 6)]
    signs = np.sign(pieces) * int(player)
    return float(np.sum(values * signs))


def _piece_font(size: int):
    from PIL import ImageFont

    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def render_board_rgb(board: np.ndarray, game_id: str = "") -> np.ndarray:
    """Human-facing RGB frame for Training Monitor GIFs.

    Occupied cells used to be solid white/black rectangles — fine for
    Tic-Tac-Toe, meaningless for chess (every piece looked like a square).
    Chess draws lettered tokens (P/N/B/R/Q/K); the ±1 games draw stones.
    """
    from PIL import Image, ImageDraw

    grid = np.asarray(board, dtype=np.int32)
    if grid.ndim != 2:
        raise ValueError(f"board must be 2D, got {grid.shape}")
    rows, cols = int(grid.shape[0]), int(grid.shape[1])
    chess = game_id == "chess" or (rows == 8 and cols == 8 and int(np.max(np.abs(grid))) > 1)
    cell = 40 if rows >= 8 else 52
    img = Image.new("RGB", (cols * cell, rows * cell), (21, 21, 32))
    draw = ImageDraw.Draw(img)

    if chess or game_id in ("gomoku", "go_9x9"):
        light, dark = _LIGHT_SQ, _DARK_SQ
    elif game_id == "connect_four":
        light = dark = (30, 90, 170)
    else:
        light, dark = (52, 52, 68), (36, 36, 48)

    for row in range(rows):
        for col in range(cols):
            x0, y0 = col * cell, row * cell
            fill = light if (row + col) % 2 == 0 else dark
            draw.rectangle((x0, y0, x0 + cell - 1, y0 + cell - 1), fill=fill)

    font = _piece_font(max(14, cell * 11 // 20)) if chess else None
    for row in range(rows):
        for col in range(cols):
            value = int(grid[row, col])
            if value == 0:
                continue
            cx = col * cell + cell // 2
            cy = row * cell + cell // 2
            white = value > 0
            if chess:
                radius = cell // 2 - 4
                token = _WHITE_TOKEN if white else _BLACK_TOKEN
                ink = _WHITE_INK if white else _BLACK_INK
                draw.ellipse(
                    (cx - radius, cy - radius, cx + radius, cy + radius),
                    fill=token,
                    outline=(70, 58, 42),
                )
                glyph = _CHESS_LETTERS.get(min(abs(value), 6), "?")
                bbox = draw.textbbox((0, 0), glyph, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text(
                    (cx - tw / 2 - bbox[0], cy - th / 2 - bbox[1] - 1),
                    glyph,
                    font=font,
                    fill=ink,
                )
            else:
                radius = cell // 2 - 5
                fill = _WHITE_TOKEN if white else _BLACK_TOKEN
                outline = (40, 40, 48) if white else (210, 210, 216)
                draw.ellipse(
                    (cx - radius, cy - radius, cx + radius, cy + radius),
                    fill=fill,
                    outline=outline,
                    width=2,
                )
    return np.asarray(img, dtype=np.uint8)


class BoardGameSelfPlayEnv(gym.Env):
    """Single-agent Gym episode = one two-player self-play game."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4, "two_player": True}

    def __init__(self, game_id: str, render_mode: str | None = None) -> None:
        super().__init__()
        if game_id not in GAME_REGISTRY:
            raise ValueError(f"Unknown board game: {game_id}")
        self.game_id = game_id
        self.game = make_game(game_id)
        self.two_player = True
        self.render_mode = render_mode
        self._steps = 0
        planes = int(self.game.input_planes)
        h, w = int(self.game.rows), int(self.game.cols)
        self.observation_space = spaces.Box(0, 255, (h, w, planes), dtype=np.uint8)
        self.action_space = spaces.Discrete(int(self.game.action_size))

    def _obs(self) -> np.ndarray:
        return _encode_hwc(self.game.encode(), self.game.rows, self.game.cols)

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(int(self.action_space.n), dtype=np.int8)
        if not self.game.done:
            mask[np.asarray(self.game.legal_actions(), dtype=np.int64)] = 1
        return mask

    def _info(self) -> dict:
        return {
            "action_mask": self.action_masks(),
            "two_player": True,
            "board": self.game.board_list(),
            "current_player": int(self.game.current_player),
            "winner": (
                None if self.game.winner is None else int(self.game.winner)
            ),
            "game_done": bool(self.game.done),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.game.reset()
        self._steps = 0
        return self._obs(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = int(action)
        if self.game.done:
            return self._obs(), 0.0, True, False, self._info()

        legal = self.game.legal_actions()
        if action not in legal:
            # Search/warmup should never do this when the mask is honoured.
            # Treat as an immediate loss for the mover so a VectorEnv worker
            # doesn't die on `BoardGame.step`'s ValueError.
            mover = int(self.game.current_player)
            self.game.done = True
            self.game.winner = -mover
            self.game.current_player = -mover
            info = self._info()
            info["illegal_action"] = True
            return self._obs(), -1.0, True, False, info

        mover = int(self.game.current_player)
        material_before = _chess_material_for(self.game.board, mover) if self.game_id == "chess" else 0.0
        _enc, done, winner = self.game.step(action)
        self._steps += 1
        # `step` flips `current_player` even on the terminal ply, so the
        # mover is the opposite of whoever is to-move now.
        if self.game_id == "chess" and not done:
            # Pure terminal chess reward gives a randomly initialized
            # model virtually no signal: most self-play games terminate
            # by repetition/ply limit as draws.  A normalized material
            # delta is a zero-sum, mover-relative transition reward; with
            # the Zero algorithms' -gamma backup, our captures are
            # positive and the opponent's captures are negative.
            material_after = _chess_material_for(self.game.board, mover)
            reward = (material_after - material_before) / _CHESS_TOTAL_MATERIAL
        elif not done:
            reward = 0.0
        elif winner in (0, None):
            reward = 0.0
        else:
            reward = 1.0 if int(winner) == mover else -1.0
        return self._obs(), float(reward), bool(done), False, self._info()

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        return render_board_rgb(self.game.board_list(), self.game_id)


def register_board_game_envs() -> None:
    """Idempotent — safe from the catalog, factory, and vector-env workers."""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True
    for game_id in GAME_REGISTRY:
        gym.register(
            id=game_id,
            entry_point="rl_core.envs.board_game_gym:BoardGameSelfPlayEnv",
            kwargs={"game_id": game_id},
            max_episode_steps=_MAX_EPISODE_STEPS.get(game_id, 256),
        )
