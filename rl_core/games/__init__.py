from rl_core.games.base import BoardGame
from rl_core.games.tic_tac_toe import TicTacToe
from rl_core.games.connect_four import ConnectFour
from rl_core.games.gomoku import Gomoku

GAME_REGISTRY: dict[str, type[BoardGame]] = {
    "tic_tac_toe": TicTacToe,
    "connect_four": ConnectFour,
    "gomoku": Gomoku,
}


def make_game(game_id: str) -> BoardGame:
    if game_id not in GAME_REGISTRY:
        raise ValueError(f"Unknown board game: {game_id}")
    return GAME_REGISTRY[game_id]()


__all__ = ["BoardGame", "TicTacToe", "ConnectFour", "Gomoku", "GAME_REGISTRY", "make_game"]
