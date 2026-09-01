"""EfficientZero V2 (Wang, Liu, Ye, You & Gao, ICML 2024 Spotlight -
https://arxiv.org/abs/2403.00564), built on EfficientZero (Ye et al.,
NeurIPS 2021 - https://arxiv.org/abs/2111.00210) and MuZero (Schrittwieser
et al., 2020 - https://arxiv.org/abs/1911.08265). Like every other
model-based algorithm in this app it learns a world model from real
experience - but where Dreamer imagines whole rollouts for an actor-critic
and PETS/MBPO plan with a CEM search over an ensemble, EfficientZero
learns a *compact* representation/dynamics/prediction triple (a la MuZero)
and plans with a small, budgeted tree search at every real step, using
that search's own output both to pick the action and as the training
target for the policy/value heads.

Four MuZero-family ingredients carried over from EfficientZero (V1 and V2):
- Representation `h(obs) -> s`, dynamics `g(s, a) -> s', reward` and
  prediction `f(s) -> (policy, value)` networks - the same three-network
  split as MuZero itself, trained *only* through what the search touches
  (no pixel/vector reconstruction loss anywhere, unlike Dreamer/World
  Models' decoders) plus one extra self-supervised term below.
- Temporal consistency ("self-supervised consistency loss", Ye et al.
  2021, SimSiam-style): the dynamics net's predicted next latent is
  pushed towards the representation net's *real* encoding of the actual
  next observation, through an asymmetric projector/predictor pair with a
  stop-gradient on the real-encoding branch (`_consistency_loss` below) -
  this is what keeps `s` predictive of the real observation instead of
  collapsing to whatever is easiest for the reward/value heads alone.
- Value prefix: the dynamics net's reward head is an LSTM carried across
  one training unroll, predicting the *cumulative* real reward since the
  unroll's start at each step rather than each step's reward in
  isolation - reduces compounding one-step reward-prediction error over
  a short unroll, exactly like the paper.
- Gumbel search (Danihelka et al., 2021 - the search EfficientZero V2
  itself reuses for its discrete case): sample a small set of candidate
  root actions via the Gumbel-Top-k trick, then narrow that set with
  Sequential Halving instead of exhaustively visiting a huge tree -
  cheap enough to run *every* real step, unlike classic PUCT-MCTS.

EfficientZero V2's own headline addition - supporting continuous action
spaces at all, which Gumbel search alone cannot (it needs a finite
candidate set) - is here too: root candidates are instead *sampled*, half
from the current Gaussian policy and half from a deliberately wider
version of it (the paper's `A_{S1}`/`A_{S2}` split), then the same
Sequential-Halving bandit picks a winner among them by simulated return.

The search itself (`_SearchNode`/`search()` below) is a *real* Gumbel
search tree - not a single-path lookahead - following "Policy improvement
by planning with Gumbel" (Danihelka, Guez, Schrittwieser & Silver, ICLR
2022, https://openreview.net/pdf?id=bERaNdoegnO), the search EfficientZero
V2 itself builds on: every one of `num_simulations` simulations descends
an actual tree by one edge per level (root: equal-visit round-robin over
the surviving top-`num_top_actions` candidates; every deeper node: the
paper's improved-policy-minus-visit-fraction rule, both using the same
`v_mix`/"completed Q" backup - Appendix D - so even *unvisited* children
get a sensible Q via their parent's value-mixed estimate instead of being
silently ignored), expands exactly one new leaf per lane via one batched
dynamics+prediction forward pass, and backs the resulting value up the
whole path with proper visit counts - genuinely the same mechanism the
paper (and EfficientZero V2's own `ez/mcts/py_mcts.py`) uses, just in pure
Python/PyTorch instead of their compiled `ctree_v2` (fine here: this app's
envs and `num_simulations` budgets are far smaller than Atari-with-a-
compiled-tree territory). Final action/policy target come from the same
formulas as the paper: Sequential Halving's surviving top action, and the
softmax of (root priors + transformed completed Qs) as the "improved
policy" distilled into the policy head at training time.

`training.num_envs > 1` collects real experience from that many parallel
env lanes (`AsyncVectorEnv`, one subprocess worker each), same as
Dreamer/World Models (Ha) - and here it's not just "run N independent
searches", it's a genuinely *batched* search: `search()` takes the whole
`(N, *obs_shape)` batch of lanes at once and runs all N trees' simulations
in lockstep, so every simulation's leaf-expansion dynamics/prediction call
is one batched `(N, ...)` forward pass, not N separate ones. Each lane
still gets its own tree/root/`_MinMaxStats` and its own slot in
`_EfficientZeroBuffer` (`add(..., lane=i)`) so lanes' episodes never splice
into one another. This is exactly Dreamer's own "single biggest
real-world speedup lever" story: the *search* is roughly the same total
compute either way (same handful of tiny batched forward passes, batch
dimension N instead of 1), but real-experience *collection* - the part
every model-based algorithm here is fundamentally bottlenecked on until
the model is decent - now gets `num_envs` real transitions per wall-clock
env.step() instead of one.

Remaining simplifications vs. the paper (kept explicit here rather than
silent, matching this app's convention elsewhere - see e.g. `qmix.py`'s
own "Stability note"):
- "Search-Based Value Estimation" (SVE) - the paper additionally re-rolls
  out imagined trajectories with the *latest* policy/model to re-estimate
  *training-time* value targets and correct for off-policy staleness on
  top of what the tree search above already gives. Here the training
  value target is instead a plain `td_steps`-step return (real rewards,
  truncated at episode end) bootstrapped by the *current* value net -
  cheaper to compute for every training sample, and value-network
  bootstrapping is itself already the standard TD fix for the same
  staleness problem, just without SVE's extra re-rollout step. The search
  tree itself (used to pick real actions and produce the policy target)
  is unabridged.
- Latent normalization is a plain `tanh` bound on `s`, not the paper's
  learned min-max rescaling - same spirit (keep the model's own latent
  space from exploding, so squared-error/cosine losses on it stay
  meaningful), simpler mechanism.
- No Dirichlet exploration noise mixed into root priors (the paper's
  standard MCTS-family exploration boost) - the Gumbel-Top-k sampling at
  the root already injects fresh per-step randomness into which actions
  even get a chance to be searched, which is Gumbel search's own built-in
  substitute for it.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from rl_core.algorithms.base import CustomAlgorithm, TrainingCallback
from rl_core.algorithms.native.preprocessing import action_to_env, obs_batch_to_array, obs_flat_dim, obs_to_array
from rl_core.algorithms.vec_env import (
    action_space as venv_action_space,
    num_envs_of,
    obs_space as venv_obs_space,
    vec_reset,
    vec_step,
)
from rl_core.world_models.nets import ObsEncoder

DEFAULT_HYPERPARAMS = {
    "learning_rate": 2e-4,
    "latent_dim": 64,
    "hidden_dim": 128,
    "proj_dim": 64,
    "buffer_size": 2_000,
    "batch_size": 64,
    "unroll_steps": 5,
    "td_steps": 5,
    "num_sampled_actions": 8,
    "num_simulations": 32,
    "num_top_actions": 8,
    "c_visit": 50.0,
    "c_scale": 0.1,
    "value_minmax_delta": 0.01,
    "gamma": 0.99,
    "value_loss_coef": 0.5,
    "policy_loss_coef": 1.0,
    "reward_loss_coef": 1.0,
    "consistency_loss_coef": 1.0,
    "continuous_prior_scale": 2.5,
    "policy_target_temperature": 1.0,
    "train_freq": 1,
    "train_steps_per_iter": 1,
    "learning_starts": 500,
    "max_grad_norm": 5.0,
}


class _Representation(nn.Module):
    """`h(obs) -> s` - reuses the exact same image/vector auto-detecting
    feature extractor every world model family in this app shares
    (`rl_core/world_models/nets.py::ObsEncoder`), plus a small projection
    head down to the (much smaller) latent size the dynamics/prediction
    nets actually plan in."""

    def __init__(self, observation_space: gym.Space, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = ObsEncoder(observation_space)
        self.head = nn.Sequential(nn.Linear(self.encoder.out_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, latent_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.head(self.encoder(obs)))


class _Dynamics(nn.Module):
    """`g(s, a) -> (s', value_prefix)` - deterministic latent transition
    (no stochastic branching, unlike Dreamer's RSSM: a MuZero-style latent
    is meant to be planned through cheaply many times per real step, which
    a stochastic one would make far more expensive) plus the LSTM "value
    prefix" reward head described in the module docstring. The LSTM state
    must be reset (`initial_lstm_state`) at the start of every fresh
    unroll - one training unroll or one search rollout - since its output
    is a *cumulative* sum from wherever that reset happened, not a
    standalone per-step reward."""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_encoder = nn.Linear(latent_dim, hidden_dim)
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.trunk = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ELU())
        self.next_latent_head = nn.Linear(hidden_dim, latent_dim)
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        self.value_prefix_head = nn.Linear(hidden_dim, 1)

    def initial_lstm_state(self, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        return h, c

    def forward(
        self, s: torch.Tensor, action: torch.Tensor, lstm_state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = torch.cat([self.state_encoder(s), self.action_encoder(action)], dim=-1)
        h = self.trunk(x)
        next_s = torch.tanh(self.next_latent_head(h))
        lstm_h, lstm_c = self.lstm(h, lstm_state)
        value_prefix = self.value_prefix_head(lstm_h).squeeze(-1)
        return next_s, value_prefix, (lstm_h, lstm_c)


class _Prediction(nn.Module):
    """`f(s) -> (policy, value)` - discrete actions get plain softmax
    logits (fed to the search as Gumbel-Top-k proposal scores); continuous
    actions get a diagonal Gaussian, exactly like `dreamer.py`'s `_Actor`
    (same tanh-squash-and-rescale-to-bounds convention)."""

    def __init__(self, latent_dim: int, action_space: gym.Space, hidden_dim: int) -> None:
        super().__init__()
        self.discrete = isinstance(action_space, gym.spaces.Discrete)
        self.trunk = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ELU())
        self.value_head = nn.Linear(hidden_dim, 1)
        if self.discrete:
            self.action_dim = int(action_space.n)
            self.policy_head = nn.Linear(hidden_dim, self.action_dim)
        else:
            self.action_dim = int(np.prod(action_space.shape))
            self.mean_head = nn.Linear(hidden_dim, self.action_dim)
            self.log_std_head = nn.Linear(hidden_dim, self.action_dim)
            low = np.where(np.isfinite(action_space.low), action_space.low, -1.0).reshape(-1)
            high = np.where(np.isfinite(action_space.high), action_space.high, 1.0).reshape(-1)
            self.register_buffer("action_scale", torch.as_tensor((high - low) / 2.0, dtype=torch.float32))
            self.register_buffer("action_bias", torch.as_tensor((high + low) / 2.0, dtype=torch.float32))

    def value(self, s: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.trunk(s)).squeeze(-1)

    def policy_logits(self, s: torch.Tensor) -> torch.Tensor:
        return self.policy_head(self.trunk(s))

    def policy_gaussian(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(s)
        mean = torch.tanh(self.mean_head(h)) * self.action_scale + self.action_bias
        log_std = self.log_std_head(h).clamp(-5.0, 1.0)
        return mean, log_std.exp()


class _Projector(nn.Module):
    """SimSiam-style projector `P1` (module docstring's consistency loss)."""

    def __init__(self, latent_dim: int, proj_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, proj_dim), nn.ELU(), nn.Linear(proj_dim, proj_dim))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class _Predictor(nn.Module):
    """SimSiam-style predictor `P2` - only applied on the *predicted*
    (dynamics) branch, never the stop-gradiented real-encoding branch,
    which is what makes the two branches asymmetric and keeps this loss
    from collapsing to a trivial constant (see Chen & He, 2021)."""

    def __init__(self, proj_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(proj_dim, proj_dim), nn.ELU(), nn.Linear(proj_dim, proj_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


class _MinMaxStats:
    """Running min/max of every backed-up value seen so far in one search
    tree, used to normalize Q-values into `[0, 1]` before they're compared
    against each other (Danihelka et al., 2022, Appendix A) - without this
    a tree whose rewards/values happen to live on a different scale than
    `c_visit`/`c_scale` were tuned for would search essentially randomly."""

    def __init__(self, delta: float) -> None:
        self.maximum = -float("inf")
        self.minimum = float("inf")
        self.delta = max(1e-6, delta)

    def update(self, value: float) -> None:
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            value = min(max(value, self.minimum), self.maximum)
            value = (value - self.minimum) / max(self.maximum - self.minimum, self.delta)
        return min(max(value, 0.0), 1.0)


class _SearchNode:
    """One node of the Gumbel search tree (Danihelka et al., 2022) - either
    a root (one per lane being searched this real step) or an imagined
    node reached from its parent by taking one of the parent's candidate
    actions and stepping the dynamics net. Every candidate action slot
    gets its own (initially un-expanded) child node the instant its parent
    is expanded - "un-expanded child" and "child the tree hasn't
    visited/expanded yet" are the same thing here, exactly like the
    reference tree."""

    __slots__ = (
        "prior", "parent", "children", "visit_count", "value_sum", "value_prefix",
        "reset_value_prefix", "state", "lstm_state", "candidate_action", "selected_children_idx",
    )

    def __init__(self, prior: float, parent: "_SearchNode | None" = None) -> None:
        self.prior = prior
        self.parent = parent
        self.children: list[_SearchNode] = []
        self.visit_count = 0
        self.value_sum = 0.0
        self.value_prefix = 0.0
        self.reset_value_prefix = True
        self.state: torch.Tensor | None = None
        self.lstm_state: tuple[torch.Tensor, torch.Tensor] | None = None
        self.candidate_action: Any = None  # continuous only: the sampled action leading to this node
        self.selected_children_idx: list[int] = []  # root only: Sequential Halving survivors

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def reward(self) -> float:
        """Real reward this node's edge contributed - value prefix is a
        *cumulative* sum since the last LSTM reset, so a node reached right
        after a reset reports its value prefix as-is, otherwise the delta
        against its parent's (Ye et al., 2021 value-prefix convention)."""
        if self.reset_value_prefix or self.parent is None:
            return self.value_prefix
        return self.value_prefix - self.parent.value_prefix

    def children_visit_sum(self) -> int:
        return sum(c.visit_count for c in self.children)

    def q_of_child(self, idx: int, discount: float) -> float:
        child = self.children[idx]
        return child.reward() + discount * child.value()

    def v_mix(self, discount: float) -> float:
        """Appendix D of Danihelka et al., 2022: blend this node's own raw
        value estimate with however many of its children are already
        expanded, weighted by how much of the node's total prior
        probability mass those expanded children cover - gives
        `completed_qs` something better than "value at the root" to use
        as a stand-in Q for children the tree hasn't visited at all yet."""
        priors = np.array([c.prior for c in self.children], dtype=np.float64)
        pi = _softmax(priors)
        pi_sum = 0.0
        pi_q_sum = 0.0
        for i, child in enumerate(self.children):
            if child.expanded():
                pi_sum += pi[i]
                pi_q_sum += pi[i] * self.q_of_child(i, discount)
        if pi_sum < 1e-6:
            return self.value()
        visit_sum = self.children_visit_sum()
        return (self.value() + visit_sum * pi_q_sum / pi_sum) / (1.0 + visit_sum)

    def completed_qs(self, discount: float, minmax: _MinMaxStats) -> np.ndarray:
        v_mix = self.v_mix(discount)
        out = np.empty(len(self.children), dtype=np.float64)
        for i, child in enumerate(self.children):
            raw = self.q_of_child(i, discount) if child.expanded() else v_mix
            out[i] = minmax.normalize(raw)
        return out

    def improved_policy(self, transformed_completed_qs: np.ndarray) -> np.ndarray:
        priors = np.array([c.prior for c in self.children], dtype=np.float64)
        return _softmax(priors + transformed_completed_qs)

    def expand(
        self, priors: np.ndarray, state: torch.Tensor, value_prefix: float,
        lstm_state: tuple[torch.Tensor, torch.Tensor], reset_value_prefix: bool,
        candidate_actions: list[Any] | None = None,
    ) -> None:
        self.state = state
        self.value_prefix = value_prefix
        self.lstm_state = lstm_state
        self.reset_value_prefix = reset_value_prefix
        self.children = [_SearchNode(float(p), self) for p in priors]
        if candidate_actions is not None:
            for child, action in zip(self.children, candidate_actions):
                child.candidate_action = action


def _transformed_completed_qs(node: _SearchNode, minmax: _MinMaxStats, discount: float, c_visit: float, c_scale: float) -> np.ndarray:
    completed = node.completed_qs(discount, minmax)
    max_visit = max((c.visit_count for c in node.children), default=0)
    return (c_visit + max_visit) * c_scale * completed


def _select_action(node: _SearchNode, minmax: _MinMaxStats, discount: float, c_visit: float, c_scale: float) -> int:
    """Root: equal-visit round robin over the current Sequential-Halving
    survivors (`selected_children_idx`) - ties broken towards whichever
    survivor is earliest in Gumbel-Top-k rank, matching the reference
    `do_equal_visit`. Non-root: the paper's deterministic selection rule -
    argmax of (improved policy - visit fraction) over *all* children, no
    Sequential Halving below the root."""
    if node.parent is None:
        best_idx, best_visits = -1, float("inf")
        for idx in node.selected_children_idx:
            visits = node.children[idx].visit_count
            if visits < best_visits:
                best_visits, best_idx = visits, idx
        return best_idx
    transformed = _transformed_completed_qs(node, minmax, discount, c_visit, c_scale)
    improved = node.improved_policy(transformed)
    visits = np.array([c.visit_count for c in node.children], dtype=np.float64)
    denom = 1.0 + node.children_visit_sum()
    scores = improved - visits / denom
    return int(np.argmax(scores))


def _sequential_halving(
    root: _SearchNode, gumbel: np.ndarray, minmax: _MinMaxStats, keep: int, discount: float, c_visit: float, c_scale: float,
) -> None:
    """Cut the root's currently-surviving candidate set in half (by score =
    Gumbel noise + prior + transformed completed Q), same formula the
    reference `sequential_halving` uses past its first phase - the first
    phase (Gumbel-Top-k over *all* actions) is instead folded into where
    `selected_children_idx` gets initialized (simulation 0 of `search()`
    below), which is exactly equivalent given the reference's own
    equal-visit round robin always exhausts the top `num_top_actions`
    candidates before any lower-ranked one ever gets a look."""
    selected = root.selected_children_idx
    if len(selected) <= 1:
        return
    priors = np.array([c.prior for c in root.children], dtype=np.float64)
    transformed = _transformed_completed_qs(root, minmax, discount, c_visit, c_scale)
    scores = np.array([gumbel[i] + priors[i] + transformed[i] for i in selected])
    order = np.argsort(-scores)
    keep = max(1, min(keep, len(selected)))
    root.selected_children_idx = [selected[i] for i in order[:keep]]


def _backpropagate(path: list[_SearchNode], leaf_value: float, minmax: _MinMaxStats, discount: float) -> None:
    value = leaf_value
    for node in reversed(path):
        node.value_sum += value
        node.visit_count += 1
        value = node.reward() + discount * value
        minmax.update(value)


class _HalvingSchedule:
    """Ports the reference `ready_for_next_gumble_phase`'s visit-budget
    schedule verbatim: phase 0 gets `floor(n / (log2(m) * m)) * m`
    simulations (always a multiple of `m = num_top_actions`, so the root's
    equal-visit round robin exhausts exactly the top-`m` Gumbel-Top-k
    candidates without ever spilling into a lower-ranked one - see
    `_sequential_halving`'s docstring), then each subsequent phase gets
    `floor(n / (log2(m) * current_m)) * current_m` more (or whatever's
    left, once `current_m <= 2`), halving `current_m` every time, until
    the budget of `n = num_simulations` simulations runs out."""

    def __init__(self, num_simulations: int, num_top_actions: int) -> None:
        self.n = max(1, num_simulations)
        self.m = max(2, num_top_actions)
        self.current_top = self.m
        self.next_cutoff = self._span(self.current_top, used=0)

    def _span(self, current_top: int, used: int) -> int:
        log2m = max(math.log2(self.m), 1e-9)
        if current_top > 2:
            span = math.floor(self.n / (log2m * current_top)) * current_top
        else:
            span = self.n - used
        return min(self.n, max(1, int(span)))

    def maybe_advance(self, simulation_idx: int) -> bool:
        ready = (simulation_idx + 1) >= self.next_cutoff
        if ready and self.current_top > 1:
            used = self.next_cutoff
            self.current_top = max(1, self.current_top // 2)
            self.next_cutoff = min(self.n, used + self._span(self.current_top, used))
        return ready


class _EfficientZeroBuffer:
    """Stores whole real episodes: `obs[t]`, `action[t]` (flat one-hot/raw
    - same convention `world_models/nets.py` already uses), `reward[t]`,
    `next_obs[t]` and `policy_target[t]` (this step's Gumbel/sampling
    search result at collection time - see module docstring). A dedicated
    buffer rather than reusing `SequenceReplayBuffer`
    (`rl_core/world_models/replay.py`) or `ContinuousReplayBuffer`
    (`rl_core/algorithms/native/buffers.py`): neither has a slot for the
    search-derived policy target, which is this algorithm's actual
    training signal for the policy head (the cross-entropy/regression
    target in `_train_step`, not the raw played action).

    `num_lanes > 1` (`training.num_envs > 1`, see module docstring) lets
    `add(..., lane=i)` accumulate `num_lanes` independent in-progress
    episodes concurrently without their interleaved transitions splicing
    into one another - same convention as `SequenceReplayBuffer`
    (`rl_core/world_models/replay.py`)."""

    def __init__(
        self, capacity_episodes: int, obs_shape: tuple[int, ...], action_dim: int, policy_target_dim: int,
        num_lanes: int = 1,
    ) -> None:
        self.capacity = max(1, capacity_episodes)
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.policy_target_dim = policy_target_dim
        self.episodes: list[dict[str, np.ndarray]] = []
        self._cur: list[dict[str, list[Any]]] = [self._new_episode() for _ in range(max(1, num_lanes))]

    @staticmethod
    def _new_episode() -> dict[str, list[Any]]:
        return {"obs": [], "action": [], "reward": [], "next_obs": [], "policy_target": []}

    def add(
        self, obs: np.ndarray, action_flat: np.ndarray, reward: float, next_obs: np.ndarray,
        policy_target: np.ndarray, done: bool, lane: int = 0,
    ) -> None:
        cur = self._cur[lane]
        cur["obs"].append(np.asarray(obs, dtype=np.float32))
        cur["action"].append(np.asarray(action_flat, dtype=np.float32))
        cur["reward"].append(float(reward))
        cur["next_obs"].append(np.asarray(next_obs, dtype=np.float32))
        cur["policy_target"].append(np.asarray(policy_target, dtype=np.float32))
        if done:
            self._flush_episode(lane)

    def _flush_episode(self, lane: int = 0) -> None:
        cur = self._cur[lane]
        if not cur["obs"]:
            return
        self.episodes.append({
            "obs": np.stack(cur["obs"]),
            "action": np.stack(cur["action"]),
            "reward": np.asarray(cur["reward"], dtype=np.float32),
            "next_obs": np.stack(cur["next_obs"]),
            "policy_target": np.stack(cur["policy_target"]),
        })
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)
        self._cur[lane] = self._new_episode()

    def __len__(self) -> int:
        return sum(len(ep["reward"]) for ep in self.episodes) + sum(len(cur["reward"]) for cur in self._cur)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    def sample(self, batch_size: int, unroll_steps: int, td_steps: int, gamma: float) -> dict[str, np.ndarray]:
        """Random (episode, start index) per batch element, `unroll_steps`
        contiguous real transitions from there (padded/masked past episode
        end), plus a `td_steps`-step (truncated at episode end) real-reward
        sum + bootstrap observation/discount/mask for the value target -
        see module docstring's "Search-Based Value Estimation" note."""
        obs0 = np.zeros((batch_size, *self.obs_shape), dtype=np.float32)
        action = np.zeros((batch_size, unroll_steps, self.action_dim), dtype=np.float32)
        reward = np.zeros((batch_size, unroll_steps), dtype=np.float32)
        next_obs = np.zeros((batch_size, unroll_steps, *self.obs_shape), dtype=np.float32)
        policy_target = np.zeros((batch_size, unroll_steps, self.policy_target_dim), dtype=np.float32)
        mask = np.zeros((batch_size, unroll_steps), dtype=np.float32)
        td_reward = np.zeros((batch_size, unroll_steps), dtype=np.float32)
        td_obs = np.zeros((batch_size, unroll_steps, *self.obs_shape), dtype=np.float32)
        td_discount = np.zeros((batch_size, unroll_steps), dtype=np.float32)
        td_bootstrap_mask = np.zeros((batch_size, unroll_steps), dtype=np.float32)

        for b in range(batch_size):
            ep = self.episodes[np.random.randint(len(self.episodes))]
            t_len = len(ep["reward"])
            start = int(np.random.randint(t_len))
            obs0[b] = ep["obs"][start]
            for k in range(unroll_steps):
                idx = start + k
                if idx >= t_len:
                    continue
                mask[b, k] = 1.0
                action[b, k] = ep["action"][idx]
                reward[b, k] = ep["reward"][idx]
                next_obs[b, k] = ep["next_obs"][idx]
                policy_target[b, k] = ep["policy_target"][idx]

                n = min(td_steps, t_len - idx)
                td_sum, discount = 0.0, 1.0
                for j in range(n):
                    td_sum += discount * float(ep["reward"][idx + j])
                    discount *= gamma
                td_reward[b, k] = td_sum
                td_discount[b, k] = discount
                landing_idx = idx + n - 1  # obs reached after n real actions from idx
                td_obs[b, k] = ep["next_obs"][landing_idx]
                td_bootstrap_mask[b, k] = 0.0 if landing_idx == t_len - 1 else 1.0

        return {
            "obs0": obs0, "action": action, "reward": reward, "next_obs": next_obs,
            "policy_target": policy_target, "mask": mask, "td_reward": td_reward,
            "td_obs": td_obs, "td_discount": td_discount, "td_bootstrap_mask": td_bootstrap_mask,
        }


class NativeEfficientZero(CustomAlgorithm):
    def __init__(self, env: gym.Env, hyperparams: dict[str, Any], seed: int | None, device: str) -> None:
        super().__init__(env, hyperparams, seed, device)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self._obs_space = venv_obs_space(env)
        self._action_space = venv_action_space(env)
        self.discrete = isinstance(self._action_space, gym.spaces.Discrete)
        self.n_actions = int(self._action_space.n) if self.discrete else None

        latent_dim = int(hyperparams.get("latent_dim", 64))
        hidden_dim = int(hyperparams.get("hidden_dim", 128))
        proj_dim = int(hyperparams.get("proj_dim", 64))
        self.action_dim = obs_flat_dim(self._action_space)

        self.representation = _Representation(self._obs_space, latent_dim, hidden_dim).to(device)
        self.dynamics = _Dynamics(latent_dim, self.action_dim, hidden_dim).to(device)
        self.prediction = _Prediction(latent_dim, self._action_space, hidden_dim).to(device)
        self.projector = _Projector(latent_dim, proj_dim).to(device)
        self.predictor = _Predictor(proj_dim).to(device)
        self._params = (
            list(self.representation.parameters()) + list(self.dynamics.parameters())
            + list(self.prediction.parameters()) + list(self.projector.parameters())
            + list(self.predictor.parameters())
        )
        self.optimizer = torch.optim.Adam(self._params, lr=float(hyperparams.get("learning_rate", 2e-4)))

        self.gamma = float(hyperparams.get("gamma", 0.99))
        self.batch_size = int(hyperparams.get("batch_size", 64))
        self.unroll_steps = max(1, int(hyperparams.get("unroll_steps", 5)))
        self.td_steps = max(1, int(hyperparams.get("td_steps", 5)))
        self.num_sampled_actions = max(2, int(hyperparams.get("num_sampled_actions", 8)))
        self.num_simulations = max(2, int(hyperparams.get("num_simulations", 32)))
        self.num_top_actions = max(2, int(hyperparams.get("num_top_actions", 8)))
        self.c_visit = float(hyperparams.get("c_visit", 50.0))
        self.c_scale = float(hyperparams.get("c_scale", 0.1))
        self.value_minmax_delta = float(hyperparams.get("value_minmax_delta", 0.01))
        self.value_loss_coef = float(hyperparams.get("value_loss_coef", 0.5))
        self.policy_loss_coef = float(hyperparams.get("policy_loss_coef", 1.0))
        self.reward_loss_coef = float(hyperparams.get("reward_loss_coef", 1.0))
        self.consistency_loss_coef = float(hyperparams.get("consistency_loss_coef", 1.0))
        self.continuous_prior_scale = float(hyperparams.get("continuous_prior_scale", 2.5))
        self.policy_target_temperature = max(1e-6, float(hyperparams.get("policy_target_temperature", 1.0)))
        self.train_freq = max(1, int(hyperparams.get("train_freq", 1)))
        self.train_steps_per_iter = max(1, int(hyperparams.get("train_steps_per_iter", 1)))
        self.learning_starts = int(hyperparams.get("learning_starts", 500))
        self.max_grad_norm = float(hyperparams.get("max_grad_norm", 5.0))

        sample_obs_arr = obs_to_array(self._obs_space.sample(), self._obs_space)
        policy_target_dim = self.n_actions if self.discrete else self.action_dim
        self.buffer = _EfficientZeroBuffer(
            capacity_episodes=int(hyperparams.get("buffer_size", 2_000)),
            obs_shape=sample_obs_arr.shape, action_dim=self.action_dim, policy_target_dim=policy_target_dim,
            num_lanes=num_envs_of(env),
        )
        self._last_metrics: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Search - a genuine batched Gumbel search tree, see module docstring.
    # ------------------------------------------------------------------
    def _sample_continuous_candidates(self, mean: torch.Tensor, std: torch.Tensor) -> tuple[list[np.ndarray], np.ndarray]:
        """K = `num_sampled_actions` candidate child actions for one node,
        sampled from *that node's own* predicted Gaussian policy - `k1`
        "on policy" (its own std) plus `k2` "wide" (inflated std,
        `continuous_prior_scale`x), the paper's `A_{S1}`/`A_{S2}` split.
        Each candidate's prior is the (unnormalized) log-density of the
        on-policy Gaussian at that action - a consistent proposal-density
        prior for every slot regardless of which of the two samplers
        actually drew it, fed into the exact same Gumbel-Top-k/Sequential
        Halving/improved-policy formulas the discrete case uses."""
        mean_np = mean.squeeze(0).detach().cpu().numpy()
        std_np = std.squeeze(0).detach().cpu().numpy()
        low, high = self._action_space.low, self._action_space.high
        k1 = max(1, self.num_sampled_actions // 2)
        k2 = max(1, self.num_sampled_actions - k1)
        cand_policy = np.random.normal(mean_np, std_np, size=(k1, self.action_dim))
        cand_wide = np.random.normal(mean_np, std_np * self.continuous_prior_scale, size=(k2, self.action_dim))
        candidates = np.clip(np.concatenate([cand_policy, cand_wide], axis=0), low, high).astype(np.float32)
        var = np.clip(std_np, 1e-6, None) ** 2
        log_probs = -0.5 * (((candidates - mean_np) ** 2) / var + np.log(2 * np.pi * var)).sum(axis=-1)
        return list(candidates), log_probs.astype(np.float64)

    def search(self, obs_batch: np.ndarray, deterministic: np.ndarray | None = None) -> list[dict[str, Any]]:
        """Batched Gumbel search - `obs_batch`: `(B, *obs_shape)`. ALL B
        lanes' trees are searched together: one shared root
        representation/prediction pass, then `num_simulations` rounds
        where every lane descends its OWN tree by exactly one edge and all
        B lanes' chosen leaf expansions are computed in one batched
        dynamics/prediction forward pass (see module docstring). Returns a
        length-B list of `{"env_action", "policy_target", "value_target"}`.
        `deterministic`: `None`/scalar bool applies to every lane, or a
        length-B bool array for per-lane control (`predict()` vs `learn()`)."""
        batch_size = obs_batch.shape[0]
        det = np.zeros(batch_size, dtype=bool) if deterministic is None else np.broadcast_to(deterministic, (batch_size,))
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            s0 = self.representation(obs_t)
            root_value = self.prediction.value(s0)
            if self.discrete:
                root_logits = self.prediction.policy_logits(s0)
            else:
                root_mean, root_std = self.prediction.policy_gaussian(s0)

        n_slots = self.n_actions if self.discrete else self.num_sampled_actions
        num_top_actions = max(2, min(self.num_top_actions, n_slots))
        roots: list[_SearchNode] = []
        for i in range(batch_size):
            root = _SearchNode(prior=1.0)
            lstm0 = self.dynamics.initial_lstm_state(1, self.device)
            if self.discrete:
                priors = root_logits[i].detach().cpu().numpy().astype(np.float64)
                root.expand(priors, s0[i], 0.0, lstm0, True)
            else:
                candidates, priors = self._sample_continuous_candidates(root_mean[i : i + 1], root_std[i : i + 1])
                root.expand(priors, s0[i], 0.0, lstm0, True, candidate_actions=candidates)
            root.visit_count = 1
            root.value_sum = float(root_value[i].item())
            roots.append(root)

        minmax_list = [_MinMaxStats(self.value_minmax_delta) for _ in range(batch_size)]
        gumbel = np.random.gumbel(size=(batch_size, n_slots)) * self.policy_target_temperature
        schedule = _HalvingSchedule(self.num_simulations, num_top_actions)
        lstm_horizon = self.unroll_steps

        for sim_idx in range(self.num_simulations):
            leaf_nodes: list[_SearchNode] = []
            parent_states, parent_lstm_h, parent_lstm_c = [], [], []
            chosen_actions: list[Any] = []
            search_paths: list[list[_SearchNode]] = []
            for lane in range(batch_size):
                root = roots[lane]
                if sim_idx == 0:
                    scores = gumbel[lane] + np.array([c.prior for c in root.children])
                    root.selected_children_idx = list(np.argsort(-scores)[:num_top_actions])
                node = root
                path = [node]
                while node.expanded():
                    idx = _select_action(node, minmax_list[lane], self.gamma, self.c_visit, self.c_scale)
                    node = node.children[idx]
                    path.append(node)
                parent = path[-2]
                leaf_nodes.append(node)
                parent_states.append(parent.state)
                parent_lstm_h.append(parent.lstm_state[0])
                parent_lstm_c.append(parent.lstm_state[1])
                chosen_actions.append(node.candidate_action if not self.discrete else parent.children.index(node))
                search_paths.append(path)

            parent_states_t = torch.stack(parent_states, dim=0)
            lstm_state = (torch.cat(parent_lstm_h, dim=0), torch.cat(parent_lstm_c, dim=0))
            if self.discrete:
                idx_t = torch.as_tensor(chosen_actions, dtype=torch.long, device=self.device)
                action_t = F.one_hot(idx_t, num_classes=self.n_actions).float()
            else:
                action_t = torch.as_tensor(np.stack(chosen_actions), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                next_s, value_prefix, next_lstm_state = self.dynamics(parent_states_t, action_t, lstm_state)
                next_value = self.prediction.value(next_s)
                if self.discrete:
                    next_logits = self.prediction.policy_logits(next_s)
                else:
                    next_mean, next_std = self.prediction.policy_gaussian(next_s)

            search_lens = np.array([len(p) for p in search_paths])
            reset_mask = (search_lens % lstm_horizon == 0)
            for lane in range(batch_size):
                leaf = leaf_nodes[lane]
                lh, lc = next_lstm_state[0][lane : lane + 1], next_lstm_state[1][lane : lane + 1]
                if reset_mask[lane]:
                    lh, lc = torch.zeros_like(lh), torch.zeros_like(lc)
                vp = float(value_prefix[lane].item())
                if self.discrete:
                    priors = next_logits[lane].detach().cpu().numpy().astype(np.float64)
                    leaf.expand(priors, next_s[lane], vp, (lh, lc), bool(reset_mask[lane]))
                else:
                    candidates, priors = self._sample_continuous_candidates(
                        next_mean[lane : lane + 1], next_std[lane : lane + 1],
                    )
                    leaf.expand(priors, next_s[lane], vp, (lh, lc), bool(reset_mask[lane]), candidate_actions=candidates)
                _backpropagate(search_paths[lane], float(next_value[lane].item()), minmax_list[lane], self.gamma)

            if schedule.maybe_advance(sim_idx):
                for lane in range(batch_size):
                    _sequential_halving(
                        roots[lane], gumbel[lane], minmax_list[lane], schedule.current_top, self.gamma, self.c_visit, self.c_scale,
                    )

        results: list[dict[str, Any]] = []
        for lane in range(batch_size):
            root = roots[lane]
            transformed = _transformed_completed_qs(root, minmax_list[lane], self.gamma, self.c_visit, self.c_scale)
            improved = root.improved_policy(transformed)
            best_idx = int(root.selected_children_idx[0]) if root.selected_children_idx else int(np.argmax(improved))
            if self.discrete:
                policy_target = improved.astype(np.float32)
                if det[lane]:
                    env_action: Any = int(np.argmax(policy_target))
                else:
                    probs = policy_target / max(float(policy_target.sum()), 1e-8)
                    env_action = int(np.random.choice(len(probs), p=probs))
            else:
                candidates_arr = np.stack([c.candidate_action for c in root.children])
                policy_target = (improved[:, None] * candidates_arr).sum(axis=0).astype(np.float32)
                env_action = candidates_arr[best_idx].astype(np.float32)
            results.append({"env_action": env_action, "policy_target": policy_target, "value_target": root.value()})
        return results

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _train_step(self) -> dict[str, float]:
        batch = self.buffer.sample(self.batch_size, self.unroll_steps, self.td_steps, self.gamma)
        obs0 = torch.as_tensor(batch["obs0"], dtype=torch.float32, device=self.device)
        action = torch.as_tensor(batch["action"], dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(batch["reward"], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        policy_target = torch.as_tensor(batch["policy_target"], dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(batch["mask"], dtype=torch.float32, device=self.device)
        td_reward = torch.as_tensor(batch["td_reward"], dtype=torch.float32, device=self.device)
        td_obs = torch.as_tensor(batch["td_obs"], dtype=torch.float32, device=self.device)
        td_discount = torch.as_tensor(batch["td_discount"], dtype=torch.float32, device=self.device)
        td_bootstrap_mask = torch.as_tensor(batch["td_bootstrap_mask"], dtype=torch.float32, device=self.device)
        cum_reward = torch.cumsum(reward, dim=1)

        batch_n = obs0.shape[0]
        s = self.representation(obs0)
        lstm_state = self.dynamics.initial_lstm_state(batch_n, self.device)
        reward_loss = torch.zeros((), device=self.device)
        value_loss = torch.zeros((), device=self.device)
        policy_loss = torch.zeros((), device=self.device)
        consistency_loss = torch.zeros((), device=self.device)

        # Representation only ever runs on the *real* initial observation
        # (`obs0`) above - every subsequent `s` in this loop is the
        # dynamics net's own predicted latent, exactly like MuZero's own
        # training unroll: the model is trained to be accurate enough to
        # unroll through by itself, real observations only ever show up
        # again as loss *targets* (`next_obs`, `td_obs` below), never fed
        # back in to keep the rollout "on track".
        for k in range(self.unroll_steps):
            m = mask[:, k]
            denom = m.sum().clamp_min(1.0)
            next_s, value_prefix, lstm_state = self.dynamics(s, action[:, k], lstm_state)

            r_loss = F.mse_loss(value_prefix, cum_reward[:, k], reduction="none")
            reward_loss = reward_loss + (r_loss * m).sum() / denom

            value_pred = self.prediction.value(s)
            with torch.no_grad():
                bootstrap_value = self.prediction.value(self.representation(td_obs[:, k])) * td_bootstrap_mask[:, k]
                value_target = td_reward[:, k] + td_discount[:, k] * bootstrap_value
            v_loss = F.mse_loss(value_pred, value_target, reduction="none")
            value_loss = value_loss + (v_loss * m).sum() / denom

            if self.discrete:
                logp = F.log_softmax(self.prediction.policy_logits(s), dim=-1)
                p_loss = -(policy_target[:, k] * logp).sum(-1)
            else:
                mean, _std = self.prediction.policy_gaussian(s)
                p_loss = F.mse_loss(mean, policy_target[:, k], reduction="none").sum(-1)
            policy_loss = policy_loss + (p_loss * m).sum() / denom

            with torch.no_grad():
                true_next_s = self.representation(next_obs[:, k])
            p_true = F.normalize(self.projector(true_next_s), dim=-1).detach()
            p_pred = F.normalize(self.predictor(self.projector(next_s)), dim=-1)
            c_loss = -(p_true * p_pred).sum(-1)
            consistency_loss = consistency_loss + (c_loss * m).sum() / denom

            s = next_s

        n = float(self.unroll_steps)
        reward_loss, value_loss, policy_loss, consistency_loss = (
            reward_loss / n, value_loss / n, policy_loss / n, consistency_loss / n,
        )
        total_loss = (
            self.reward_loss_coef * reward_loss + self.value_loss_coef * value_loss
            + self.policy_loss_coef * policy_loss + self.consistency_loss_coef * consistency_loss
        )
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._params, self.max_grad_norm)
        self.optimizer.step()
        return {
            "reward_loss": float(reward_loss.item()),
            "value_loss": float(value_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "consistency_loss": float(consistency_loss.item()),
            "total_loss": float(total_loss.item()),
        }

    # ------------------------------------------------------------------
    # CustomAlgorithm contract
    # ------------------------------------------------------------------
    def learn(self, total_timesteps: int, callback: TrainingCallback) -> None:
        n_envs = num_envs_of(self.env)
        obs_list = vec_reset(self.env, seed=self.seed)
        obs_arr = obs_batch_to_array(obs_list, self._obs_space)
        ep_reward = np.zeros(n_envs, dtype=np.float64)
        ep_length = np.zeros(n_envs, dtype=np.int64)
        num_timesteps = 0

        while num_timesteps < total_timesteps:
            env_actions: list[Any] = []
            action_flats: list[np.ndarray] = []
            policy_targets: list[np.ndarray] = []
            if num_timesteps < self.learning_starts:
                for _ in range(n_envs):
                    raw_action = self._action_space.sample()
                    policy_target = (
                        np.full(self.n_actions, 1.0 / self.n_actions, dtype=np.float32)
                        if self.discrete
                        else np.asarray(raw_action, dtype=np.float32).reshape(-1)
                    )
                    env_action = action_to_env(raw_action, self._action_space)
                    env_actions.append(env_action)
                    action_flats.append(obs_to_array(env_action, self._action_space))
                    policy_targets.append(policy_target)
            else:
                # One shared, batched `search()` call plans all `n_envs`
                # lanes' trees together this real step (see module
                # docstring) - not a Python loop over independent searches.
                results = self.search(obs_arr, deterministic=None)
                for result in results:
                    env_action = action_to_env(result["env_action"], self._action_space)
                    env_actions.append(env_action)
                    action_flats.append(obs_to_array(env_action, self._action_space))
                    policy_targets.append(result["policy_target"])

            next_obs_list, rewards, terminated, truncated, _infos = vec_step(self.env, env_actions)
            dones = terminated | truncated
            next_obs_arr = obs_batch_to_array(next_obs_list, self._obs_space)
            for lane in range(n_envs):
                self.buffer.add(
                    obs_arr[lane], action_flats[lane], float(rewards[lane]), next_obs_arr[lane],
                    policy_targets[lane], bool(dones[lane]), lane=lane,
                )
            obs_arr = next_obs_arr
            ep_reward += rewards
            ep_length += 1
            prev_num_timesteps = num_timesteps
            num_timesteps += n_envs

            if (
                num_timesteps >= self.learning_starts
                and num_timesteps // self.train_freq != prev_num_timesteps // self.train_freq
                and self.buffer.num_episodes >= 1
            ):
                for _ in range(self.train_steps_per_iter):
                    self._last_metrics = self._train_step()

            any_done = False
            for lane in range(n_envs):
                if not dones[lane]:
                    continue
                any_done = True
                keep_going = callback.on_step(
                    num_timesteps, float(ep_reward[lane]), int(ep_length[lane]), self._last_metrics,
                )
                ep_reward[lane], ep_length[lane] = 0.0, 0
                if not keep_going:
                    return
            if not any_done:
                keep_going = callback.on_step(num_timesteps, metrics=self._last_metrics)
                if not keep_going:
                    return

    def predict(self, obs: Any, deterministic: bool = True) -> tuple[Any, Any]:
        obs_arr = obs_to_array(obs, self._obs_space)
        results = self.search(obs_arr[None], deterministic=np.array([deterministic]))
        return action_to_env(results[0]["env_action"], self._action_space), None

    def save(self, path: Path) -> None:
        torch.save(
            {
                "representation_state_dict": self.representation.state_dict(),
                "dynamics_state_dict": self.dynamics.state_dict(),
                "prediction_state_dict": self.prediction.state_dict(),
                "projector_state_dict": self.projector.state_dict(),
                "predictor_state_dict": self.predictor.state_dict(),
                "hyperparams": self.hyperparams,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, env: gym.Env, device: str = "cpu") -> "NativeEfficientZero":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        algo = cls(env, payload.get("hyperparams", {}), None, device)
        algo.representation.load_state_dict(payload["representation_state_dict"])
        algo.dynamics.load_state_dict(payload["dynamics_state_dict"])
        algo.prediction.load_state_dict(payload["prediction_state_dict"])
        algo.projector.load_state_dict(payload["projector_state_dict"])
        algo.predictor.load_state_dict(payload["predictor_state_dict"])
        return algo
