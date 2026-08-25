"""Generates self-play games used as AlphaZero training examples."""
from __future__ import annotations

from typing import Any

import numpy as np

from rl_core.alphazero.mcts import MCTS, visit_counts_to_policy
from rl_core.games.base import BoardGame

TrainingExample = tuple[np.ndarray, np.ndarray, float]


def play_self_play_game(
    game_cls: type[BoardGame],
    network,
    num_simulations: int = 50,
    c_puct: float = 1.5,
    temperature_moves: int = 8,
    device: str = "cpu",
) -> tuple[list[TrainingExample], dict[str, Any]]:
    """Plays one game of the network against itself via MCTS.

    Returns training examples `(state, policy, value)` — value is relative
    to the player who was to move in that state — plus a compact move
    history usable to replay the game in the AlphaZero Arena page.
    """
    game = game_cls()
    game.reset()
    mcts = MCTS(network, num_simulations=num_simulations, c_puct=c_puct, device=device)

    raw_examples: list[tuple[np.ndarray, np.ndarray, int]] = []
    move_history: list[dict[str, Any]] = []
    # One board snapshot per move (post-move state) — lets the Training
    # Monitor (and the AlphaZero Arena replay page) play the *entire* game
    # move by move instead of only ever seeing the final position.
    board_history: list[list[list[int]]] = [game.board_list()]
    move_count = 0

    while not game.done:
        temperature = 1.0 if move_count < temperature_moves else 0.1
        visit_counts = mcts.run(game, add_noise=True)
        policy = visit_counts_to_policy(visit_counts, game.action_size, temperature=temperature)

        raw_examples.append((game.encode(), policy, game.current_player))

        probs = policy / policy.sum() if policy.sum() > 0 else None
        if probs is None:
            action = int(np.random.choice(game.legal_actions()))
        else:
            action = int(np.random.choice(len(policy), p=probs))

        move_history.append({"action": action, "player": int(game.current_player)})
        game.step(action)
        board_history.append(game.board_list())
        move_count += 1

    winner = game.winner
    examples: list[TrainingExample] = []
    for state, policy, player in raw_examples:
        value = 0.0 if winner == 0 else (1.0 if winner == player else -1.0)
        examples.append((state, policy, value))

    record = {
        "moves": move_history,
        "winner": int(winner) if winner is not None else 0,
        "num_moves": move_count,
        "final_board": game.board_list(),
        "board_history": board_history,
    }
    return examples, record
