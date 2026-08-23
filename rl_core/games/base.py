"""Shared two-player, zero-sum board game interface used by AlphaZero.

Boards are stored as an int8 grid with +1 for the player who moved first,
-1 for the other player, 0 for empty. `current_player` is +1 or -1 and
flips every move, so the same win-checking code works for any game built
on this base regardless of absolute player identity.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def count_in_direction(board: np.ndarray, row: int, col: int, dr: int, dc: int, player: int) -> int:
    rows, cols = board.shape
    count = 0
    r, c = row + dr, col + dc
    while 0 <= r < rows and 0 <= c < cols and board[r, c] == player:
        count += 1
        r += dr
        c += dc
    return count


def check_win_at(board: np.ndarray, row: int, col: int, player: int, k: int) -> bool:
    """True if placing `player` at (row, col) completes a run of length k."""
    for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        total = 1 + count_in_direction(board, row, col, dr, dc, player) + count_in_direction(
            board, row, col, -dr, -dc, player
        )
        if total >= k:
            return True
    return False


class BoardGame(ABC):
    id: str
    name: str
    rows: int
    cols: int
    win_length: int

    def __init__(self) -> None:
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1
        self.done = False
        self.winner: int | None = None  # +1, -1, or 0 for draw
        self.last_move: tuple[int, int] | None = None

    @property
    def action_size(self) -> int:
        return self.rows * self.cols

    def reset(self) -> np.ndarray:
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1
        self.done = False
        self.winner = None
        self.last_move = None
        return self.encode()

    @abstractmethod
    def legal_actions(self) -> list[int]:
        ...

    @abstractmethod
    def step(self, action: int) -> tuple[np.ndarray, bool, int | None]:
        """Applies `action` for the current player. Returns (encoded_state, done, winner)."""
        ...

    def encode(self) -> np.ndarray:
        """3 planes from the perspective of the player to move: own stones,
        opponent stones, and a constant plane (helps the conv net know whose
        turn it is when the board itself is symmetric, e.g. an empty board)."""
        own = (self.board == self.current_player).astype(np.float32)
        opp = (self.board == -self.current_player).astype(np.float32)
        turn = np.full_like(own, 1.0 if self.current_player == 1 else 0.0)
        return np.stack([own, opp, turn], axis=0)

    def clone(self) -> "BoardGame":
        clone = self.__class__()
        clone.board = self.board.copy()
        clone.current_player = self.current_player
        clone.done = self.done
        clone.winner = self.winner
        clone.last_move = self.last_move
        return clone

    def render_ascii(self, symbols: tuple[str, str, str] = (".", "X", "O")) -> str:
        chars = {0: symbols[0], 1: symbols[1], -1: symbols[2]}
        return "\n".join(
            " ".join(chars[int(v)] for v in row) for row in self.board
        )

    def board_list(self) -> list[list[int]]:
        return self.board.tolist()
