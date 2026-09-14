"""Pits two networks against each other (training accept/reject gate) and
answers "what would the AI play here?" for the interactive Arena page.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
    workers: int = 1,
) -> dict[str, int]:
    """Alternates who plays first across games so neither net gets a first-move edge."""
    def play_one(i: int) -> str:
        # MCTS instances own mutable trees, so each concurrent game gets
        # its own pair while the read-only network weights are shared.
        mcts_a = MCTS(network_a, num_simulations=num_simulations, device=device)
        mcts_b = MCTS(network_b, num_simulations=num_simulations, device=device)
        a_is_first = i % 2 == 0
        game = game_cls()
        game.reset()
        while not game.done:
            mcts = mcts_a if (game.current_player == 1) == a_is_first else mcts_b
            visit_counts = mcts.run(game, add_noise=False)
            action = max(visit_counts, key=visit_counts.get)
            game.step(action)

        if game.winner == 0:
            return "draw"
        return "a" if (game.winner == 1) == a_is_first else "b"

    worker_count = min(max(1, int(workers)), max(1, int(num_games)))
    if worker_count == 1:
        outcomes = list(map(play_one, range(num_games)))
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="alphazero-arena") as pool:
            outcomes = list(pool.map(play_one, range(num_games)))

    return {
        "wins_a": outcomes.count("a"),
        "wins_b": outcomes.count("b"),
        "draws": outcomes.count("draw"),
        "games": num_games,
    }


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
