"""Pits two networks against each other (training accept/reject gate) and
answers "what would the AI play here?" for the interactive Arena page.
"""
from __future__ import annotations

from typing import Any

from rl_core.alphazero.mcts import MCTS
from rl_core.games.base import BoardGame


def play_match(
    game_cls: type[BoardGame],
    network_a,
    network_b,
    num_simulations: int = 40,
    num_games: int = 10,
    device: str = "cpu",
) -> dict[str, int]:
    """Alternates who plays first across games so neither net gets a first-move edge."""
    mcts_a = MCTS(network_a, num_simulations=num_simulations, device=device)
    mcts_b = MCTS(network_b, num_simulations=num_simulations, device=device)

    wins_a = wins_b = draws = 0
    for i in range(num_games):
        a_is_first = i % 2 == 0
        game = game_cls()
        game.reset()
        while not game.done:
            mcts = mcts_a if (game.current_player == 1) == a_is_first else mcts_b
            visit_counts = mcts.run(game, add_noise=False)
            action = max(visit_counts, key=visit_counts.get)
            game.step(action)

        if game.winner == 0:
            draws += 1
        elif (game.winner == 1) == a_is_first:
            wins_a += 1
        else:
            wins_b += 1

    return {"wins_a": wins_a, "wins_b": wins_b, "draws": draws, "games": num_games}


def suggest_move(
    game: BoardGame,
    network,
    num_simulations: int = 80,
    device: str = "cpu",
) -> dict[str, Any]:
    """Runs MCTS from the current position and returns the AI's chosen move
    plus the full visit-count distribution, so the UI can render a heatmap
    of "how much the AI considered each cell"."""
    mcts = MCTS(network, num_simulations=num_simulations, device=device)
    visit_counts = mcts.run(game, add_noise=False)
    if not visit_counts:
        return {"action": None, "visit_counts": {}}
    best_action = max(visit_counts, key=visit_counts.get)
    total = sum(visit_counts.values()) or 1
    return {
        "action": int(best_action),
        "visit_counts": {int(a): c for a, c in visit_counts.items()},
        "visit_probs": {int(a): c / total for a, c in visit_counts.items()},
    }
