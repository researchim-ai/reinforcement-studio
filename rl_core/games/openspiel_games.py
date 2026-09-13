"""OpenSpiel board games wrapped as `BoardGame` for AlphaZero self-play.

Chess and Go do not fit the built-in "drop a stone on a ±1/0 grid" games
(Tic-Tac-Toe / Connect Four / Gomoku): chess has piece types and from-to
moves (OpenSpiel: 20×8×8 observation, 4674 distinct actions), Go has
captures, ko and pass. OpenSpiel (`pip install open_spiel`, import
`pyspiel`) is the standard self-play implementation of both — C++ engine,
legal-action masks, `State.clone()` for MCTS.

Player ids in OpenSpiel are not consistent across games (chess starts at
player 1 = White; Go starts at player 0 = Black). We snapshot whoever is
to move after `reset()` as "our +1" and map `returns()[that_id]` onto the
same winner convention the rest of AlphaZero uses.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from rl_core.games.base import BoardGame

_GAME_CACHE: dict[str, Any] = {}

_FEN_PIECE = {
    "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
    "p": -1, "n": -2, "b": -3, "r": -4, "q": -5, "k": -6,
}


def _pyspiel_game(name: str):
    game = _GAME_CACHE.get(name)
    if game is None:
        import pyspiel

        game = pyspiel.load_game(name)
        _GAME_CACHE[name] = game
    return game


def _fen_to_board(fen: str) -> np.ndarray:
    placement = fen.split()[0]
    rows: list[list[int]] = []
    for rank in placement.split("/"):
        row: list[int] = []
        for ch in rank:
            if ch.isdigit():
                row.extend([0] * int(ch))
            else:
                row.append(_FEN_PIECE[ch])
        rows.append(row)
    return np.asarray(rows, dtype=np.int8)


def _go_to_board(text: str, size: int) -> np.ndarray:
    """Parse OpenSpiel `GoState.to_string()` into ±1 stones, rank N at row 0."""
    grid = np.zeros((size, size), dtype=np.int8)
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or not stripped[0].isdigit():
            continue
        rank_s, _, rest = stripped.partition(" ")
        try:
            rank = int(rank_s)
        except ValueError:
            continue
        if not (1 <= rank <= size):
            continue
        cells = rest.strip()
        row = size - rank
        for col, ch in enumerate(cells[:size]):
            if ch in "Xx":
                grid[row, col] = 1
            elif ch in "Oo":
                grid[row, col] = -1
    return grid


class OpenSpielBoardGame(BoardGame):
    """Shared wrapper. Subclasses set `id`, `name`, `rows`, `cols`,
    `openspiel_name`."""

    openspiel_name: str
    win_length = 0

    def __init__(self) -> None:
        self._game = _pyspiel_game(self.openspiel_name)
        self._action_n = int(self._game.num_distinct_actions())
        self._obs_shape = tuple(int(x) for x in self._game.observation_tensor_shape())
        self._first_os_player = 0
        self._state = None
        self.reset()

    @property
    def action_size(self) -> int:
        return self._action_n

    @property
    def input_planes(self) -> int:
        return int(self._obs_shape[0])

    def reset(self) -> np.ndarray:
        self._state = self._game.new_initial_state()
        while self._state.is_chance_node():
            action, _prob = self._state.chance_outcomes()[0]
            self._state.apply_action(action)
        self._first_os_player = int(self._state.current_player())
        self.current_player = 1
        self.done = False
        self.winner = None
        self.last_move = None
        self._sync_board()
        return self.encode()

    def legal_actions(self) -> list[int]:
        if self.done:
            return []
        return list(self._state.legal_actions())

    def encode(self) -> np.ndarray:
        # Terminal OpenSpiel states have current_player == −4, and
        # observation_tensor() then throws. MCTS/self-play never read the
        # encoding of a finished position (they already have the winner).
        if self._state.is_terminal():
            return np.zeros(self._obs_shape, dtype=np.float32)
        flat = np.asarray(self._state.observation_tensor(), dtype=np.float32)
        return flat.reshape(self._obs_shape)

    def step(self, action: int) -> tuple[np.ndarray, bool, int | None]:
        if self.done:
            raise ValueError("Game already finished")
        action = int(action)
        if action not in self._state.legal_actions():
            raise ValueError(f"Illegal OpenSpiel action: {action}")
        self._state.apply_action(action)
        if self._state.is_terminal():
            self.done = True
            returns = self._state.returns()
            first_ret = float(returns[self._first_os_player])
            if first_ret > 0:
                self.winner = 1
            elif first_ret < 0:
                self.winner = -1
            else:
                self.winner = 0
            # Same "flip even on the terminal ply" convention as BoardGame
            # (see mcts.py) — not OpenSpiel's terminal player id (−4).
            self.current_player = -self.current_player
        else:
            os_p = int(self._state.current_player())
            self.current_player = 1 if os_p == self._first_os_player else -1
        self._sync_board()
        return self.encode(), self.done, self.winner

    def clone(self) -> BoardGame:
        cloned = self.__class__.__new__(self.__class__)
        cloned._game = self._game
        cloned._action_n = self._action_n
        cloned._obs_shape = self._obs_shape
        cloned._first_os_player = self._first_os_player
        cloned._state = self._state.clone()
        cloned.board = self.board.copy()
        cloned.current_player = self.current_player
        cloned.done = self.done
        cloned.winner = self.winner
        cloned.last_move = self.last_move
        return cloned

    def _sync_board(self) -> None:
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)

    def render_ascii(self, symbols: tuple[str, str, str] = (".", "X", "O")) -> str:
        fn = getattr(self._state, "debug_string", None) or getattr(self._state, "to_string", None)
        if fn is not None:
            return str(fn())
        return super().render_ascii(symbols)


class Chess(OpenSpielBoardGame):
    id = "chess"
    name = "Chess"
    rows = 8
    cols = 8
    openspiel_name = "chess"

    def _sync_board(self) -> None:
        self.board = _fen_to_board(self._state.board().to_fen())


class Go9x9(OpenSpielBoardGame):
    id = "go_9x9"
    name = "Go 9×9"
    rows = 9
    cols = 9
    openspiel_name = "go(board_size=9,komi=7.5)"

    def _sync_board(self) -> None:
        self.board = _go_to_board(self._state.to_string(), self.rows)
