from __future__ import annotations

import numpy as np

from rl_core.games.base import BoardGame, check_win_at


class ConnectFour(BoardGame):
    id = "connect_four"
    name = "Connect Four"
    rows = 6
    cols = 7
    win_length = 4

    @property
    def action_size(self) -> int:
        # Actions are column indices — a piece always drops to the lowest
        # empty row, unlike Tic-Tac-Toe/Gomoku where you can place anywhere.
        return self.cols

    def legal_actions(self) -> list[int]:
        if self.done:
            return []
        return [c for c in range(self.cols) if self.board[0, c] == 0]

    def _drop_row(self, col: int) -> int:
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, col] == 0:
                return row
        raise ValueError(f"Column {col} is full")

    def step(self, action: int) -> tuple[np.ndarray, bool, int | None]:
        if self.done:
            raise ValueError("Game already finished")
        if self.board[0, action] != 0:
            raise ValueError(f"Illegal move: column {action} is full")

        row = self._drop_row(action)
        self.board[row, action] = self.current_player
        self.last_move = (row, action)

        if check_win_at(self.board, row, action, self.current_player, self.win_length):
            self.done = True
            self.winner = self.current_player
        elif not self.legal_actions():
            self.done = True
            self.winner = 0

        self.current_player = -self.current_player
        return self.encode(), self.done, self.winner
