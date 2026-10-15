"""Batched Gumbel AlphaZero search, "1sh" variant (after nanoAlphaZero).

For every root position in the batch:

1. Evaluate the root; add Gumbel noise to its (temperature-scaled) prior
   logits over the legal moves and take the top ``num_considered`` moves.
2. Play each considered move on a real copy of the game and evaluate the
   child: ``q1 = −V(child)`` (or the exact result if the move ends the game).
3. Keep the top ``num_survivors`` of them by ``g + logit + σ(q1)``; for each
   survivor play the opponent's greedy reply and evaluate the grandchild —
   ``q2 = V(grandchild)`` — and average: ``q = (q1 + q2) / 2``.
4. Complete Q for the unvisited moves with the mixed value and build the
   improved policy ``softmax(logit + σ(completed_q))`` — the training target —
   and pick ``argmax(g + logit + σ)`` among the survivors as the move.

Every search step is one batched network call for all roots at once, so the
cost of a whole self-play ply is 3 forward passes regardless of how many
games run in parallel. Values stay in [−1, 1] (no min-max rescaling), and
σ = (maxvisit_init + max_visits) · value_scale · q, exactly like the
reference implementation.

The search is written as a generator (`gumbel_search_steps`) that yields its
three network requests instead of calling the network itself, so the same
code runs in-process (`gumbel_search`) and inside self-play worker
processes that talk to a network server (`selfplay_workers.py`). A request
is ``(encodings, rows, actions)``; the answer is ``(logits[rows, actions],
values)`` — only the logits of legal moves ever leave the GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Generator

import numpy as np

from rl_core.games.base import BoardGame

if TYPE_CHECKING:
    import torch

Request = tuple[np.ndarray, np.ndarray, np.ndarray]
Answer = tuple[np.ndarray, np.ndarray]
Evaluator = Callable[[np.ndarray, np.ndarray, np.ndarray], Answer]


@dataclass
class SearchConfig:
    num_considered: int = 8
    num_survivors: int = 4
    root_temperature: float = 1.0
    gumbel_scale: float = 1.0
    maxvisit_init: float = 50.0
    value_scale: float = 1.0


@dataclass
class Evaluated:
    """A position the network has already seen: encoding, legal moves and
    their logits, value for the side to move."""

    obs: np.ndarray
    legal: np.ndarray
    logits: np.ndarray
    value: float


@dataclass
class SearchResult:
    action: int
    legal: np.ndarray
    policy: np.ndarray
    visits: np.ndarray
    root_value: float
    obs: np.ndarray
    # Evaluated positions after each considered (non-terminal) move — the
    # next search's root once one of them is played.
    children: dict[int, Evaluated]


def _step(game: BoardGame, action: int) -> None:
    fast = getattr(game, "step_trusted", None)
    if fast is not None:
        fast(int(action))
    else:
        game.step(int(action))


def _result_for(winner: int | None, player: int) -> float:
    if not winner:
        return 0.0
    return 1.0 if winner == player else -1.0


def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max())
    return z / z.sum()


def _request(encodings: list[np.ndarray], obs_shape: tuple[int, ...], legal: list[np.ndarray] | None) -> Request:
    enc = np.stack(encodings) if encodings else np.zeros((0, *obs_shape), dtype=np.float32)
    if not legal:
        empty = np.zeros(0, dtype=np.int64)
        return enc, empty, empty
    rows = np.repeat(np.arange(len(legal), dtype=np.int64), [len(x) for x in legal])
    return enc, rows, np.concatenate(legal).astype(np.int64, copy=False)


def run_steps(gen: Generator[Request, Answer, object], evaluate: Evaluator):
    """Drives a request-yielding generator with an in-process evaluator."""
    try:
        request = next(gen)
        while True:
            request = gen.send(evaluate(*request))
    except StopIteration as stop:
        return stop.value


def gumbel_search_steps(
    games: list[BoardGame],
    cfg: SearchConfig,
    rng: np.random.Generator,
    roots: list[Evaluated | None] | None = None,
) -> Generator[Request, Answer, list[SearchResult]]:
    """One 1sh Gumbel search per (non-terminal) game in ``games``. Always
    yields exactly three requests (some may be empty). ``roots[i]``, if
    given, is game i's already-evaluated current position and is not sent
    to the network again."""
    n = len(games)
    if n == 0:
        return []
    roots = list(roots) if roots is not None else [None] * n
    missing = [i for i in range(n) if roots[i] is None]
    missing_enc = [games[i].encode() for i in missing]
    missing_legal = [np.asarray(games[i].legal_actions(), dtype=np.int64) for i in missing]
    obs_shape = (missing_enc[0] if missing_enc else roots[0].obs).shape
    flat_logits, values = yield _request(missing_enc, obs_shape, missing_legal)
    offset = 0
    for k, i in enumerate(missing):
        count = len(missing_legal[k])
        roots[i] = Evaluated(missing_enc[k], missing_legal[k], flat_logits[offset:offset + count], float(values[k]))
        offset += count
    root_enc = [r.obs for r in roots]
    legal_of = [r.legal for r in roots]

    logits_of: list[np.ndarray] = []
    gumbel_of: list[np.ndarray] = []
    considered_of: list[np.ndarray] = []
    q_of: list[np.ndarray] = []
    visits_of: list[np.ndarray] = []
    # (root, legal slot) -> index into the child batch; None = terminal child.
    child_index: list[dict[int, int | None]] = []
    children: list[BoardGame] = []
    child_legal: list[np.ndarray] = []

    child_enc: list[np.ndarray] = []
    temperature = cfg.root_temperature
    for i, game in enumerate(games):
        legal = legal_of[i]
        count = len(legal)
        lg = roots[i].logits.astype(np.float64) / temperature
        lg -= lg.max()
        gn = rng.gumbel(size=count) * cfg.gumbel_scale if cfg.gumbel_scale > 0 else np.zeros(count)
        k = min(cfg.num_considered, count)
        considered = np.argsort(-(gn + lg), kind="stable")[:k]
        q = np.zeros(count)
        visits = np.zeros(count, dtype=np.int64)
        mover = game.current_player
        index: dict[int, int | None] = {}
        for j in considered.tolist():
            child = game.clone()
            _step(child, legal[j])
            visits[j] = 1
            if child.done:
                q[j] = _result_for(child.winner, mover)
                index[j] = None
            else:
                index[j] = len(children)
                children.append(child)
                child_enc.append(child.encode())
                child_legal.append(np.asarray(child.legal_actions(), dtype=np.int64))
        logits_of.append(lg)
        gumbel_of.append(gn)
        considered_of.append(considered)
        q_of.append(q)
        visits_of.append(visits)
        child_index.append(index)

    child_flat, child_values = yield _request(child_enc, obs_shape, child_legal)
    child_offsets = np.concatenate([[0], np.cumsum([len(x) for x in child_legal])]).astype(np.int64)

    sigma_scale_1 = (cfg.maxvisit_init + 1.0) * cfg.value_scale
    survivors_of: list[np.ndarray] = []
    grandchildren: list[BoardGame] = []
    grand_ref: list[tuple[int, int, float]] = []
    for i in range(n):
        q, index = q_of[i], child_index[i]
        for j, ci in index.items():
            if ci is not None:
                q[j] = -float(child_values[ci])
        considered = considered_of[i]
        score = gumbel_of[i][considered] + logits_of[i][considered] + sigma_scale_1 * q[considered]
        survivors = considered[np.argsort(-score, kind="stable")[: min(cfg.num_survivors, len(considered))]]
        survivors_of.append(survivors)
        for j in survivors.tolist():
            visits_of[i][j] = 2
            ci = index[j]
            if ci is None:
                continue
            child = children[ci]
            lo, hi = child_offsets[ci], child_offsets[ci + 1]
            reply = int(child_legal[ci][int(np.argmax(child_flat[lo:hi]))])
            grand = child.clone()
            reply_mover = child.current_player
            _step(grand, reply)
            q1 = q[j]
            if grand.done:
                q[j] = 0.5 * (q1 - _result_for(grand.winner, reply_mover))
            else:
                grand_ref.append((i, j, q1))
                grandchildren.append(grand)

    _, grand_values = yield _request([g.encode() for g in grandchildren], obs_shape, None)
    for (i, j, q1), v in zip(grand_ref, grand_values):
        q_of[i][j] = 0.5 * (q1 + float(v))

    results: list[SearchResult] = []
    for i in range(n):
        legal, lg, q, visits = legal_of[i], logits_of[i], q_of[i], visits_of[i]
        visited = visits > 0
        prior = _softmax(lg)
        total_visits = float(visits.sum())
        p_visited = float(prior[visited].sum())
        weighted_q = float((prior[visited] * q[visited]).sum() / max(p_visited, 1e-12))
        root_v = roots[i].value
        mixed = (root_v + total_visits * weighted_q) / (total_visits + 1.0)
        completed_q = np.where(visited, q, mixed)
        sigma = (cfg.maxvisit_init + float(visits.max())) * cfg.value_scale * completed_q
        improved = _softmax(lg + sigma)
        survivors = survivors_of[i]
        best = survivors[int(np.argmax(gumbel_of[i][survivors] + lg[survivors] + sigma[survivors]))]
        results.append(SearchResult(
            action=int(legal[best]),
            legal=legal,
            policy=improved.astype(np.float32),
            visits=visits,
            root_value=root_v,
            obs=root_enc[i],
            children={
                int(legal[j]): Evaluated(
                    child_enc[ci], child_legal[ci],
                    child_flat[child_offsets[ci]:child_offsets[ci + 1]], float(child_values[ci]),
                )
                for j, ci in child_index[i].items() if ci is not None
            },
        ))
    return results


def gumbel_search(
    games: list[BoardGame],
    evaluate: Evaluator,
    cfg: SearchConfig,
    rng: np.random.Generator,
) -> list[SearchResult]:
    return run_steps(gumbel_search_steps(games, cfg, rng), evaluate)


class NetEvaluator:
    """Wraps a policy/value network as an `Evaluator`: values are from the
    point of view of the side to move, and only the requested
    ``logits[rows, actions]`` are gathered (on the device) and returned.
    Works with any net whose ``forward`` returns ``(policy_logits,
    scalar_value)``."""

    def __init__(self, net: "torch.nn.Module", device: str, chunk: int = 2048) -> None:
        import torch

        self.net = net
        self.device = device
        self.chunk = max(1, int(chunk))
        self.use_bf16 = str(device).startswith("cuda") and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.channels_last = str(device).startswith("cuda")

    def __call__(self, encodings: np.ndarray, rows: np.ndarray, actions: np.ndarray) -> Answer:
        import torch

        n = len(encodings)
        if n == 0:
            return np.zeros(len(rows), dtype=np.float32), np.zeros(0, dtype=np.float32)
        was_training = self.net.training
        self.net.eval()
        logits_parts: list[torch.Tensor] = []
        value_parts: list[torch.Tensor] = []
        try:
            with torch.inference_mode():
                for start in range(0, n, self.chunk):
                    x = torch.from_numpy(np.ascontiguousarray(encodings[start:start + self.chunk]))
                    x = x.to(self.device, non_blocking=True).float()
                    if self.channels_last and x.dim() == 4:
                        x = x.contiguous(memory_format=torch.channels_last)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                        logits, value = self.net(x)
                    logits_parts.append(logits)
                    value_parts.append(value.reshape(-1))
                values = torch.cat(value_parts).float().cpu().numpy()
                if len(rows) == 0:
                    return np.zeros(0, dtype=np.float32), values
                logits = torch.cat(logits_parts) if len(logits_parts) > 1 else logits_parts[0]
                r = torch.from_numpy(np.ascontiguousarray(rows, dtype=np.int64)).to(self.device, non_blocking=True)
                a = torch.from_numpy(np.ascontiguousarray(actions, dtype=np.int64)).to(self.device, non_blocking=True)
                gathered = logits[r, a].float().cpu().numpy()
        finally:
            if was_training:
                self.net.train()
        return gathered, values
