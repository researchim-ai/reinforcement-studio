"""Two-player self-play helpers shared by the MuZero-family algorithms.

Board games in this app are alternating zero-sum: one policy plays both
seats, each Gym step is one ply, and the terminal reward is the outcome
for the player who just moved. Search backups and n-step TD then use
discount ``-γ`` so the next state's value (from the opponent's view) is
negated — the standard MuZero board-game trick.

Root expansion is restricted to ``info["action_mask"]`` (chess has 4674
distinct actions; expanding them all is both illegal and too slow).
Interior latent nodes have no real legal set, so they keep at most
``MAX_INTERIOR_ACTIONS`` children, ranked by the policy logits.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from rl_core.algorithms.vec_env import is_vector_env

MAX_INTERIOR_ACTIONS = 32
LARGE_ACTION_SPACE = 64

ZERO_SELFPLAY_ALGORITHMS = (
    "efficientzero",
    "unizero",
    "researchimzero",
    "latentimzero",
)


def is_two_player_env(env: Any) -> bool:
    """True when the (possibly wrapped / vectorized) env is alternating
    two-player self-play — see `BoardGameSelfPlayEnv.two_player`."""
    candidates: list[Any] = [env]
    if is_vector_env(env):
        inner = getattr(env, "unwrapped", None)
        if inner is not None:
            candidates.append(inner)
    seen: set[int] = set()
    for start in candidates:
        cur = start
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            if bool(getattr(cur, "two_player", False)):
                return True
            meta = getattr(cur, "metadata", None) or {}
            if meta.get("two_player"):
                return True
            nxt = getattr(cur, "env", None)
            if nxt is None or nxt is cur:
                unwrapped = getattr(cur, "unwrapped", None)
                cur = unwrapped if unwrapped is not cur else None
            else:
                cur = nxt
    return False


def attach_two_player(algo: Any, env: Any) -> None:
    """Set `two_player` / `search_discount` on a Zero algorithm. Call after
    `algo.gamma` is assigned."""
    algo.two_player = is_two_player_env(env)
    algo.search_discount = -float(algo.gamma) if algo.two_player else float(algo.gamma)


def maybe_enable_two_player_from_infos(algo: Any, infos: list[dict[str, Any]]) -> None:
    """VectorEnv workers don't always expose `two_player` on the parent;
    the Gym wrapper also puts it in `info` every step."""
    if getattr(algo, "two_player", False):
        return
    if any(bool(info.get("two_player")) for info in infos):
        algo.two_player = True
        algo.search_discount = -float(algo.gamma)


def action_mask_from_info(info: dict[str, Any], n_actions: int) -> np.ndarray | None:
    mask = info.get("action_mask")
    if mask is None:
        return None
    arr = np.asarray(mask).reshape(-1)
    if arr.size != n_actions:
        return None
    return (arr > 0).astype(np.int8)


def masks_from_infos(infos: list[dict[str, Any]], n_actions: int) -> np.ndarray | None:
    masks = [action_mask_from_info(info, n_actions) for info in infos]
    if not masks or any(m is None for m in masks):
        return None
    return np.stack(masks)


def mask_from_unwrapped_env(env: Any, n_actions: int) -> np.ndarray | None:
    """Best-effort legal mask for `predict()`, which has no `info` dict."""
    cur = env
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        fn = getattr(cur, "action_masks", None)
        if callable(fn):
            mask = np.asarray(fn()).reshape(-1)
            if mask.size == n_actions:
                return (mask > 0).astype(np.int8)
        nxt = getattr(cur, "env", None)
        cur = nxt if nxt is not cur else getattr(cur, "unwrapped", None)
    return None


def sample_masked_action(action_space: Any, mask: np.ndarray | None) -> int:
    if mask is None:
        return int(action_space.sample())
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        return int(action_space.sample())
    return int(np.random.choice(legal))


def uniform_legal_policy(n_actions: int, mask: np.ndarray | None) -> np.ndarray:
    policy = np.zeros(n_actions, dtype=np.float32)
    if mask is None:
        policy[:] = 1.0 / max(n_actions, 1)
        return policy
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        policy[:] = 1.0 / max(n_actions, 1)
        return policy
    policy[legal] = 1.0 / legal.size
    return policy


def discrete_expand_slots(
    logits: np.ndarray,
    *,
    action_mask: np.ndarray | None = None,
    max_children: int | None = None,
    as_logits: bool = True,
) -> tuple[np.ndarray, list[int]]:
    """Pick the children a discrete search node should expand.

    `logits` is the raw policy head `(n_actions,)`. When `as_logits` is
    True the returned priors stay in log-space (Gumbel-Top-k); otherwise
    they are a softmax over the selected actions (PUCT).
    """
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    n = int(logits.shape[0])
    if action_mask is not None:
        legal = np.flatnonzero(np.asarray(action_mask).reshape(-1) > 0)
        if legal.size == 0:
            legal = np.arange(n)
    else:
        legal = np.arange(n)
    if max_children is not None and legal.size > max_children:
        scores = logits[legal]
        keep = np.argpartition(-scores, max_children - 1)[:max_children]
        legal = legal[keep]
    child_logits = logits[legal]
    action_ids = [int(a) for a in legal.tolist()]
    if as_logits:
        return child_logits, action_ids
    shifted = child_logits - child_logits.max()
    exp = np.exp(shifted)
    return exp / max(float(exp.sum()), 1e-12), action_ids


def interior_max_children(n_actions: int) -> int | None:
    if n_actions > LARGE_ACTION_SPACE:
        return MAX_INTERIOR_ACTIONS
    return None


def scatter_policy(child_policy: np.ndarray, action_ids: list[int], n_actions: int) -> np.ndarray:
    full = np.zeros(n_actions, dtype=np.float32)
    for prob, action_id in zip(child_policy, action_ids):
        full[int(action_id)] += float(prob)
    total = float(full.sum())
    if total > 0:
        full /= total
    return full


def child_action_ids(node: Any) -> list[int]:
    ids: list[int] = []
    for i, child in enumerate(node.children):
        action = getattr(child, "candidate_action", None)
        ids.append(int(action) if action is not None else i)
    return ids


def discrete_policy_target(node: Any, child_policy: np.ndarray, n_actions: int) -> np.ndarray:
    ids = child_action_ids(node)
    if ids == list(range(len(node.children))) and len(node.children) == n_actions:
        return np.asarray(child_policy, dtype=np.float32)
    return scatter_policy(child_policy, ids, n_actions)


def discrete_env_action(node: Any, child_index: int) -> int:
    child = node.children[child_index]
    action = getattr(child, "candidate_action", None)
    return int(action) if action is not None else int(child_index)


def chosen_discrete_action(leaf: Any, parent: Any) -> int:
    action = getattr(leaf, "candidate_action", None)
    if action is not None:
        return int(action)
    return int(parent.children.index(leaf))
