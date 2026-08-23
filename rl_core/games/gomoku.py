from __future__ import annotations

import numpy as np

from rl_core.games.base import BoardGame, check_win_at


class Gomoku(BoardGame):
    id = "gomoku"
    name = "Gomoku"
    rows = 9
    cols = 9
    win_length = 5

    def legal_actions(self) -> list[int]:
        if self.done:
            return []
        flat = self.board.reshape(-1)
        return [i for i, v in enumerate(flat) if v == 0]

    def step(self, action: int) -> tuple[np.ndarray, bool, int | None]:
        if self.done:
            raise ValueError("Game already finished")
        row, col = divmod(action, self.cols)
        if self.board[row, col] != 0:
            raise ValueError(f"Illegal move: cell {action} occupied")

        self.board[row, col] = self.current_player
        self.last_move = (row, col)

        if check_win_at(self.board, row, col, self.current_player, self.win_length):
            self.done = True
            self.winner = self.current_player
        elif not self.legal_actions():
            self.done = True
            self.winner = 0

        self.current_player = -self.current_player
        return self.encode(), self.done, self.winner
