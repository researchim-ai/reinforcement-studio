"""PUCT Monte-Carlo Tree Search guided by an AlphaZeroNet.

Convention (matches games/base.py): a game's `current_player` always flips
after every move, including the terminal one. So at any state, `encode()`
and the network's value output are from the perspective of "the player
about to move" — even at a just-finished game, that's the player who did
NOT make the winning/final move. This makes the backup step a plain sign
flip at every ply, with no special-casing for terminal nodes.
"""
from __future__ import annotations

import math

import numpy as np

from rl_core.games.base import BoardGame


class Node:
    __slots__ = ("prior", "children", "visit_count", "value_sum")

    def __init__(self, prior: float) -> None:
        self.prior = prior
        self.children: dict[int, "Node"] = {}
        self.visit_count = 0
        self.value_sum = 0.0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0


class MCTS:
    def __init__(
        self,
        network,
        num_simulations: int = 50,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        device: str = "cpu",
    ) -> None:
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.device = device

    def _evaluate_and_expand(self, node: Node, game: BoardGame) -> float:
        probs, value = self.network.predict(game.encode(), device=self.device)
        legal = game.legal_actions()
        mask = np.zeros_like(probs)
        mask[legal] = 1.0
        probs = probs * mask
        total = probs.sum()
        probs = probs / total if total > 0 else mask / max(len(legal), 1)

        for action in legal:
            node.children[action] = Node(prior=float(probs[action]))
        return value

    def run(self, game: BoardGame, add_noise: bool = True) -> dict[int, int]:
        root = Node(prior=1.0)
        self._evaluate_and_expand(root, game)

        if add_noise and root.children:
            actions = list(root.children.keys())
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
            for action, n in zip(actions, noise):
                child = root.children[action]
                child.prior = (1 - self.dirichlet_eps) * child.prior + self.dirichlet_eps * n

        for _ in range(self.num_simulations):
            node = root
            sim_game = game.clone()
            path = [node]

            while node.expanded():
                action, node = self._select_child(node)
                _, done, winner = sim_game.step(action)
                path.append(node)
                if done:
                    break

            if sim_game.done:
                value = 0.0 if sim_game.winner == 0 else (
                    1.0 if sim_game.winner == sim_game.current_player else -1.0
                )
            else:
                value = self._evaluate_and_expand(node, sim_game)

            self._backpropagate(path, value)

        return {action: child.visit_count for action, child in root.children.items()}

    def _select_child(self, node: Node) -> tuple[int, Node]:
        total_visits = sum(c.visit_count for c in node.children.values())
        best_score = -float("inf")
        best_action, best_child = -1, None
        for action, child in node.children.items():
            q = -child.value()  # child value is from opponent's perspective
            u = self.c_puct * child.prior * math.sqrt(total_visits + 1) / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score, best_action, best_child = score, action, child
        return best_action, best_child

    def _backpropagate(self, path: list[Node], value: float) -> None:
        # `value` is from the perspective of the player to move at the leaf.
        # Each ancestor alternates perspective, so flip sign every step up.
        v = value
        for node in reversed(path):
            node.value_sum += v
            node.visit_count += 1
            v = -v


def visit_counts_to_policy(visit_counts: dict[int, int], action_size: int, temperature: float = 1.0) -> np.ndarray:
    policy = np.zeros(action_size, dtype=np.float32)
    if not visit_counts:
        return policy
    actions = np.array(list(visit_counts.keys()))
    counts = np.array(list(visit_counts.values()), dtype=np.float32)
    if temperature <= 1e-3:
        best = actions[np.argmax(counts)]
        policy[best] = 1.0
        return policy
    counts = counts ** (1.0 / temperature)
    counts = counts / counts.sum()
    policy[actions] = counts
    return policy
