"""Batched self-play and strength evaluation for the built-in AlphaZero.

`SelfPlayPool` keeps N games in flight at once and advances all of them by
one ply per call, with a single batched `gumbel_search` for the whole pool.
Finished games are immediately replaced by fresh ones, so the network
always sees a full batch.

Each game draws its own exploration depth ``switch ~ U[0, exploration_moves)``:
before that ply the move is sampled from the search's visit counts (to
diversify openings), after it the search's greedy Gumbel choice is played.
Exploratory positions are *not* used for training — their targets would
describe the search, but the game continuation (and so the value target)
comes from a deliberately noisy move.

Strength is measured on a fixed-opponent ladder rather than against the
previous iteration: first against a uniformly random player, then — once
the model beats its anchor convincingly twice in a row — against a frozen
copy of itself, and so on. Each rung adds the measured Elo difference, so
``ladder_elo`` is comparable across the whole run.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generator

import numpy as np

from rl_core.alphazero.gumbel import (
    Answer, Evaluated, Evaluator, Request, SearchConfig, _step, gumbel_search, gumbel_search_steps, run_steps,
)
from rl_core.alphazero.replay import DRAW, LOSS, WIN
from rl_core.games.base import BoardGame

if TYPE_CHECKING:
    import torch


@dataclass
class _Slot:
    game: BoardGame
    switch: int
    ply: int = 0
    positions: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = field(default_factory=list)
    moves: list[dict[str, int]] = field(default_factory=list)
    boards: list[Any] | None = None
    root: Evaluated | None = None


@dataclass
class FinishedGame:
    winner: int
    plies: int
    truncated: bool
    record: dict[str, Any] | None


class SelfPlayPool:
    def __init__(
        self,
        game_cls: type[BoardGame],
        num_games: int,
        rng: np.random.Generator,
        exploration_moves: int,
        max_plies: int = 0,
        tracked_games: int = 3,
    ) -> None:
        self.game_cls = game_cls
        self.rng = rng
        self.exploration_moves = max(0, int(exploration_moves))
        self.max_plies = max(0, int(max_plies))
        self.slots = [self._new_slot(track=i < tracked_games) for i in range(max(1, int(num_games)))]

    def _new_slot(self, track: bool) -> _Slot:
        game = self.game_cls()
        switch = int(self.rng.integers(0, self.exploration_moves)) if self.exploration_moves > 0 else 0
        return _Slot(game=game, switch=switch, boards=[game.board_list()] if track else None)

    def forget_evaluations(self) -> None:
        """Drops the cached root evaluations — call after the network
        changed, so no search starts from the old network's numbers."""
        for slot in self.slots:
            slot.root = None

    def play_ply(
        self, evaluate: Evaluator, cfg: SearchConfig,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray, int]], list[FinishedGame]]:
        """Advances every game by one move. Returns the training samples of
        the games that finished on this ply, and those games' summaries."""
        return run_steps(self.play_ply_steps(cfg), evaluate)

    def play_ply_steps(
        self, cfg: SearchConfig,
    ) -> Generator[Request, Answer, tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray, int]], list[FinishedGame]]]:
        """`play_ply` as a request-yielding generator (see gumbel.py)."""
        results = yield from gumbel_search_steps(
            [s.game for s in self.slots], cfg, self.rng, roots=[s.root for s in self.slots],
        )
        samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
        finished: list[FinishedGame] = []
        for idx, (slot, res) in enumerate(zip(self.slots, results)):
            game = slot.game
            player = int(game.current_player)
            if slot.ply < slot.switch:
                probs = res.visits / res.visits.sum()
                action = int(self.rng.choice(res.legal, p=probs))
            else:
                action = res.action
                slot.positions.append((res.obs.astype(np.float16), res.legal, res.policy, player))
            slot.moves.append({"action": action, "player": player})
            _step(game, action)
            slot.root = res.children.get(action)
            slot.ply += 1
            if slot.boards is not None:
                slot.boards.append(game.board_list())
            truncated = not game.done and self.max_plies > 0 and slot.ply >= self.max_plies
            if not (game.done or truncated):
                continue
            winner = int(game.winner or 0) if game.done else 0
            for obs, legal, policy, mover in slot.positions:
                outcome = DRAW if winner == 0 else (WIN if winner == mover else LOSS)
                samples.append((obs, legal, policy, outcome))
            record = None
            if slot.boards is not None:
                record = {
                    "moves": slot.moves,
                    "winner": winner,
                    "num_moves": slot.ply,
                    "final_board": game.board_list(),
                    "board_history": slot.boards,
                    "truncated": truncated,
                }
            finished.append(FinishedGame(winner=winner, plies=slot.ply, truncated=truncated, record=record))
            self.slots[idx] = self._new_slot(track=slot.boards is not None)
        return samples, finished


def make_openings(game_cls: type[BoardGame], count: int, plies: int, seed: int = 20260924) -> list[list[int]]:
    """``count`` distinct random opening lines of ``plies`` moves (fewer if
    the game does not have that many), fixed by ``seed`` so every evaluation
    of a run starts from the same positions."""
    count = max(1, int(count))
    if plies <= 0:
        return [[] for _ in range(count)]
    rng = np.random.default_rng(seed)
    seen: set[tuple[int, ...]] = set()
    lines: list[list[int]] = []
    for _ in range(count * 20):
        game = game_cls()
        line: list[int] = []
        for _ in range(plies):
            legal = game.legal_actions()
            if game.done or not legal:
                break
            a = int(rng.choice(legal))
            line.append(a)
            game.step(a)
        if game.done:
            continue
        key = tuple(line)
        if key in seen:
            continue
        seen.add(key)
        lines.append(line)
        if len(lines) >= count:
            break
    while len(lines) < count:
        lines.append(lines[len(lines) % max(1, len(seen))] if lines else [])
    return lines


def play_match(
    game_cls: type[BoardGame],
    agent: Evaluator,
    opponent: Evaluator | None,
    openings: list[list[int]],
    cfg: SearchConfig,
    rng: np.random.Generator,
    max_plies: int = 0,
) -> dict[str, float]:
    """Plays every opening twice (agent as each colour) with all games
    batched together. ``opponent=None`` is the uniformly random player.
    Games still running at ``max_plies`` count as draws."""
    games: list[BoardGame] = []
    agent_side: list[int] = []
    for line in openings:
        for side in (1, -1):
            g = game_cls()
            for a in line:
                g.step(a)
            games.append(g)
            agent_side.append(side)
    plies = [0] * len(games)
    active = [i for i, g in enumerate(games) if not g.done]
    while active:
        mine = [i for i in active if games[i].current_player == agent_side[i]]
        theirs = [i for i in active if games[i].current_player != agent_side[i]]
        moves: dict[int, int] = {}
        for i, res in zip(mine, gumbel_search([games[i] for i in mine], agent, cfg, rng)):
            moves[i] = res.action
        if opponent is None:
            for i in theirs:
                moves[i] = int(rng.choice(games[i].legal_actions()))
        else:
            for i, res in zip(theirs, gumbel_search([games[i] for i in theirs], opponent, cfg, rng)):
                moves[i] = res.action
        for i, a in moves.items():
            _step(games[i], a)
            plies[i] += 1
        active = [i for i in active if not games[i].done and not (max_plies > 0 and plies[i] >= max_plies)]

    wins = draws = losses = 0
    for g, side in zip(games, agent_side):
        winner = int(g.winner or 0) if g.done else 0
        if winner == 0:
            draws += 1
        elif winner == side:
            wins += 1
        else:
            losses += 1
    total = max(1, wins + draws + losses)
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "games": wins + draws + losses,
        "score": (wins + 0.5 * draws) / total,
    }


def elo_from_score(score: float, games: int) -> float:
    """Elo difference implied by a match score, clamped so a clean sweep of
    ``games`` games reads as "about half a game better" instead of ±∞."""
    eps = 0.5 / max(1, games)
    s = min(max(score, eps), 1.0 - eps)
    return 400.0 * math.log10(s / (1.0 - s))


class EloLadder:
    def __init__(self, promote_score: float = 0.75, patience: int = 2, random_max_elo: float = 0.0) -> None:
        self.promote_score = float(promote_score)
        self.patience = max(1, int(patience))
        self.random_max_elo = random_max_elo
        self.anchor_state: dict[str, torch.Tensor] | None = None
        self.anchor_elo = 0.0
        self.rung = 0
        self.streak = 0
        self.ladder_elo = 0.0

    def update(self, score: float, games: int, net: torch.nn.Module) -> bool:
        """Records a score against the current anchor; returns True if the
        model was promoted to be the new anchor."""
        delta = elo_from_score(score, games)
        if self.anchor_state is None and self.random_max_elo is not None:
            delta = min(delta, float(self.random_max_elo))
        self.ladder_elo = self.anchor_elo + delta
        self.streak = self.streak + 1 if score >= self.promote_score else 0
        if self.streak < self.patience:
            return False
        self.anchor_state = copy.deepcopy({k: v.detach().cpu() for k, v in net.state_dict().items()})
        self.anchor_elo = self.ladder_elo
        self.rung += 1
        self.streak = 0
        return True
