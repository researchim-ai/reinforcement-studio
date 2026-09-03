"""UniZero (Pu, Zhao, Niu, Zhu, Chen, Ni, Song, Liu, ICLR 2025 -
https://arxiv.org/abs/2406.10667), from the LightZero project
(https://github.com/opendilab/LightZero). Every other MuZero-family
algorithm in this app (`efficientzero.py`) learns a *recurrent* latent
dynamics model - an LSTM/MLP steps one `s_k -> s_{k+1}` at a time, so the
model's only "memory" of everything before `s_k` is whatever got squeezed
into that one fixed-size vector. UniZero's actual contribution is
replacing that recurrence with a **causal Transformer over an explicit,
interleaved token sequence** `[obs_0, act_0, obs_1, act_1, ..., obs_t]`:
every past observation and action is its own token, still directly
attendable by the model at every later step, instead of being entangled
(and potentially half-forgotten) into one carried-over latent vector.
That's genuinely useful for anything with long-range dependencies
(remembering a key picked up many steps ago, a partially-observed maze) -
and, per the paper, even in fully-observed single-task settings it trains
*faster* than a recurrent world model because every real observation in a
training window contributes its own reward/value/policy loss term
directly, not just the first one before the model has to unroll blind.

Three ingredients ported from the paper/reference (`LightZero-main/lzero/
model/unizero_world_models/`, `lzero/policy/unizero.py`):
- **Tokenizer + interleaved sequence**: each observation is encoded to one
  token (`_Tokenizer`, reusing the same `ObsEncoder` every world model in
  this app shares), each action to one token (`_ActionEmbed`), and a
  bounded window of the most recent `context_length` real (obs, action)
  pairs plus the current observation form the token sequence a causal
  Transformer (`_CausalTransformer`) attends over.
- **Head placement**: reward is read off the hidden state at an *action*
  token's position (predicts that single action's real reward - no
  cumulative "value prefix" trick needed here, unlike EfficientZero's
  LSTM reward head: attention already gives the model the reward head's
  own view of everything relevant, so there's nothing to gain from
  additionally training it to sum); policy/value are read off an *obs*
  token's position (`_Heads`); both value and reward are categorical
  distributions over a fixed support (MuZero Appendix F's
  `signed_hyperbolic` scaling - same `_scalar_to_two_hot`/
  `_logits_to_scalar` machinery `efficientzero.py` uses, ported here
  rather than re-derived, for the exact same reason: bounding the loss/
  gradient regardless of how large one particular reward/value happens to
  be in a given minibatch).
- **Full-trajectory teacher forcing**: training feeds the Transformer a
  window of *real* interleaved tokens (context + `unroll_steps + 1` real
  observations + `unroll_steps` real actions) in **one** forward pass and
  reads every step's reward/policy/value loss off that single pass -
  every real observation in the window contributes directly (the paper's
  own "full trajectory utilization" advantage over recurrent unrolling).

Search is a genuine tree search, not a single-path lookahead - but rather
than reproducing the reference's classic PUCT/Dirichlet-noise MCTS (a
separate, compiled `ctree_muzero`), this reuses `efficientzero.py`'s own
*Gumbel* search tree bookkeeping (`_SearchNode`/`_MinMaxStats`/
`_HalvingSchedule`/Sequential Halving/`v_mix`/"completed Q" - see that
file's own module docstring for the full citation) essentially verbatim,
for the same reason EfficientZero V2 itself prefers it over classic MCTS:
cheap enough to run every real step. The only thing that changes between
the two files' searches is *what a tree node's "state" is* -
EfficientZero's is a fixed-size latent vector (`s`, plus an LSTM cell);
here it's the node's own slice of the growing raw token sequence, and
"stepping the dynamics" means appending two more raw tokens (one action,
one *predicted* next-observation) and asking the Transformer for a fresh
hidden state - see `_step_imagine` below.

**Positional encoding: RoPE, not learned absolute embeddings.** This is
what makes the KV-cache below both *correct* and *fast* at the same time,
so it's worth spelling out why up front. Rotary position embeddings
(Su et al., 2021, https://arxiv.org/abs/2104.09864) rotate each attention
head's query/key vectors by an angle proportional to that token's
position *before* the QK dot product, so `<q_p, k_q>` (and therefore every
attention weight) is a function of `p - q` alone - never of `p` or `q`'s
own absolute value. Two direct, load-bearing consequences:
- **Any two token sequences with the same relative spacing produce
  identical attention, regardless of where either one's positions
  "start counting from".** A training window that always labels its own
  first token position `0` and a self-play root that labels the exact
  same *kind* of token its true, ever-growing absolute step-count-since-
  episode-start position are therefore *mathematically the same
  computation* - there is no "convention" to keep in sync between the two
  call sites at all, unlike a learned additive `nn.Embedding` table (this
  file's own earlier approach, and the reference's default), where the
  value looked up for position `500` and position `0` are two unrelated
  learned vectors and the two call sites *must* agree on which one a
  given token gets.
- **Evicting the oldest entries from a KV-cache needs zero *positional*
  correction.** A cached key's rotation is baked in at the moment it's
  computed and never touched again; dropping unrelated old cache slots
  changes nothing about the relative offset between any two *surviving*
  tokens (or a future query and a surviving key), so trimming the front
  of a cache is a plain tensor slice (`_TransformerCache.evict_front`) -
  no analogue of the reference's own approximate "shift + re-based
  position-delta" fix (needed there specifically because *its* additive
  embeddings aren't shift-invariant) is needed here. This is a genuinely
  different, narrower claim than "eviction is a complete no-op", though:
  with `num_layers > 1`, a surviving token's own cached hidden state
  (hence its K/V in every layer *above* the first) was computed *while
  the now-evicted tokens were still present*, so it still carries a
  little of their influence forward through the residual stream - the
  same well-known property any multi-layer causal-attention KV-cache
  eviction scheme has (this file's own or the reference's), independent
  of positional encoding entirely. That's an intentional, bounded trade
  (see `_TransformerCache.evict_front`'s own docstring) - the actual
  alternative would be never evicting at all, i.e. paying `O(episode
  length)` compute for every real step's root instead of `O(1)` - not a
  free correctness upgrade being left on the table.
Concretely: `_CausalSelfAttention` rotates `q`/`k` (never `v`) by each
token's position right before the dot product, in both the full-sequence
path (`forward`, used by training's teacher-forcing pass - positions are
always plain `0..seq_len-1` per lane, per the first bullet above,
equivalent to whatever absolute position the same tokens would carry
anywhere else they're used) and the incremental path
(`forward_incremental_batch`, used by every self-play/search-tree call
below - positions come from each `_TransformerCache`'s own monotonic
`next_pos` counter, which - unlike `length`, the cache's *physical* token
count - never decreases, including across an eviction).

**Persistent, incrementally-extended per-lane root cache.** Because RoPE
makes eviction lossless, every real env step (`learn()`'s collection loop,
`predict()`'s eval-time stepping) now does exactly what the reference
itself does and what an earlier, pre-RoPE version of this file *tried* to
do but had to abandon (see prior revisions' own comments, since removed)
for being unsound with additive embeddings: keep one `_TransformerCache`
per lane that lives across the *entire episode*, extended by exactly two
tokens (one action, one observation) per real step and trimmed back down
to `2 * context_length` tokens (`_TransformerCache.evict_front`, called
from `NativeUniZero._advance_lane_caches`) whenever it grows past that
budget - never rebuilt from scratch. Reset to `None` (`first_step_flag`,
in the reference's own terms) whenever an episode ends. This is the
*actual* engineering payoff of a KV-cache the reference's own paper/README
flags as the hardest part to get right: a real step's own root token no
longer costs `O(context_length)` (replaying the whole retained window from
scratch every time, this file's own pre-RoPE approach) but a genuine
`O(1)` - exactly one incremental Transformer call, same as any single
step of `search()`'s own tree expansion (`_step_imagine`) already was.
`_reanalyze()`'s arbitrary historical (episode, timestep) samples have no
live per-lane cache to reuse (they're not "the current real step" for any
lane), so they still cold-start a cache from scratch by replaying their
own stored context (`_replay_context_to_cache`) - there's nothing to
persist across independent reanalyze calls anyway.

`training.num_envs > 1` batches the search across all lanes exactly like
`efficientzero.py` (one shared `_step_imagine` call handles every lane's
current simulation step at once); prioritized replay (Schaul et al.,
2016) and reanalyze (periodic *policy*-target-refresh, ported from the
reference's own always-on background reanalyze workers) are reused from
`efficientzero.py`'s `_EfficientZeroBuffer` - both concepts are entirely
orthogonal to "recurrent vs Transformer dynamics", so there was no reason
to simplify them away here just because the world model changed. Two
details differ from `efficientzero.py`'s own copy, deliberately, to match
the *reference UniZero*'s (not EfficientZero V2's) own choices here
(`lzero/mcts/buffer/game_buffer_unizero.py`,
`lzero/policy/unizero.py`/configs): the value target is a plain n-step TD
target - reanalyze's refreshed MCTS `search_value` only ever overwrites
the buffer's *policy* target, never blended into the value target via
`max()` (`efficientzero.py`'s own, correct-for-*that*-algorithm choice;
see `_train_step`'s value-target comment for why it's wrong here) - and
`DEFAULT_HYPERPARAMS["reanalyze_batch_size"]` defaults to `0` (off),
matching the reference UniZero configs' own `reanalyze_ratio = 0` default
(`efficientzero.py` reanalyzes by default; the reference UniZero mostly
doesn't).

Other simplifications vs. the paper/reference, in the same spirit:
- Discrete actions get Gumbel search over the real action set; continuous
  actions get EfficientZero V2's own answer (which the reference's
  "Sampled UniZero" variant also independently arrives at) - sample a
  fixed number of candidate actions from the current Gaussian policy (half
  "on-policy", half deliberately wider) and run the same Sequential-
  Halving bandit over that candidate set. Ported from `efficientzero.py`
  verbatim (`_sample_continuous_candidates`) rather than re-derived.
- No multitask task-token/shared-model machinery, no LPIPS/pixel
  reconstruction decoder, no open-loop consistency losses, no separate
  async collect/train/reanalyze worker processes (Ray) - reanalyze runs
  synchronously in-loop, same trade-off `efficientzero.py` itself makes.
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
    "embed_dim": 128,
    "num_layers": 2,
    "num_heads": 4,
    "dropout": 0.0,
    # Past real (obs, action) *transitions* kept before "now" in every
    # token sequence the Transformer ever sees (root of a real step, a
    # training window, a reanalyze pass) - `2 * context_length` tokens.
    "context_length": 6,
    "buffer_size": 2_000,
    "batch_size": 64,
    # Doubles as the training teacher-forcing window length *and* the max
    # search-tree depth budget (an imagined rollout can't run longer than
    # a real training unroll would need to correct it) - same dual role
    # `unroll_steps` plays in `efficientzero.py`.
    "unroll_steps": 5,
    "td_steps": 5,
    "num_sampled_actions": 8,
    "num_simulations": 32,
    "num_top_actions": 8,
    "c_visit": 50.0,
    "c_scale": 0.1,
    "value_minmax_delta": 0.01,
    "value_support_size": 300,
    "gamma": 0.99,
    "value_loss_coef": 0.25,
    "policy_loss_coef": 1.0,
    "reward_loss_coef": 1.0,
    "consistency_loss_coef": 2.0,
    "continuous_prior_scale": 2.5,
    "policy_target_temperature": 1.0,
    "train_freq": 1,
    "train_steps_per_iter": 1,
    "learning_starts": 500,
    "priority_alpha": 1.0,
    "priority_beta": 1.0,
    "min_priority": 1e-6,
    "reanalyze_freq": 200,
    # Off by default - matches the reference LightZero UniZero configs'
    # own default (`reanalyze_ratio = 0` in e.g.
    # `zoo/classic_control/cartpole/config/cartpole_unizero_config.py`
    # and `zoo/atari/config/atari_unizero_stack4_config.py`). Reanalyze
    # only ever refreshes *policy* targets from a fresh MCTS search over
    # old (episode, timestep) samples (`_reanalyze()`'s own docstring) -
    # it's a variance-reduction knob, not something the base algorithm
    # needs to be "valid" - set to e.g. `64` to opt back in.
    "reanalyze_batch_size": 0,
    "max_grad_norm": 5.0,
}


# ----------------------------------------------------------------------
# Tokenizer + embeddings - raw (pre-Transformer) per-token feature vectors.
# ----------------------------------------------------------------------
class _Tokenizer(nn.Module):
    """One observation -> one token embedding. Reuses the same image/vector
    auto-detecting feature extractor every world model family in this app
    shares (`rl_core/world_models/nets.py::ObsEncoder`)."""

    def __init__(self, observation_space: gym.Space, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = ObsEncoder(observation_space)
        self.head = nn.Sequential(nn.Linear(self.encoder.out_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, embed_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.head(self.encoder(obs)))


class _ActionEmbed(nn.Module):
    """One action (one-hot for `Discrete`, raw vector for `Box` - same
    `obs_to_array` convention every algorithm in this app uses) -> one
    token embedding, same bounded scale as `_Tokenizer`'s output so both
    token kinds share one embedding space."""

    def __init__(self, action_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(action_dim, embed_dim)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(action))


# ----------------------------------------------------------------------
# Categorical value/reward support - ported verbatim from
# `efficientzero.py` (MuZero Appendix F "scaling and squashing"); see that
# file's module docstring for the full rationale.
# ----------------------------------------------------------------------
def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _signed_hyperbolic(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.0) - 1.0) + eps * x


def _signed_parabolic(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    z = torch.sqrt(1.0 + 4.0 * eps * (torch.abs(x) + 1.0 + eps)) - 1.0
    return torch.sign(x) * ((z / (2.0 * eps)) ** 2 - 1.0)


def _scalar_to_two_hot(x: torch.Tensor, support_size: int) -> torch.Tensor:
    num_bins = 2 * support_size + 1
    x = _signed_hyperbolic(x).clamp(-support_size, support_size)
    lower = x.floor()
    upper = lower + 1
    lower_weight = (upper - x).clamp(0.0, 1.0)
    upper_weight = 1.0 - lower_weight
    lower_idx = (lower + support_size).long().clamp(0, num_bins - 1)
    upper_idx = (upper + support_size).long().clamp(0, num_bins - 1)
    out = torch.zeros(*x.shape, num_bins, device=x.device, dtype=torch.float32)
    out.scatter_add_(-1, lower_idx.unsqueeze(-1), lower_weight.unsqueeze(-1))
    out.scatter_add_(-1, upper_idx.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return out


def _logits_to_scalar(logits: torch.Tensor, support_size: int) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    bins = torch.arange(-support_size, support_size + 1, device=logits.device, dtype=torch.float32)
    x = (probs * bins).sum(-1)
    return _signed_parabolic(x)


# ----------------------------------------------------------------------
# Rotary position embeddings (Su et al., 2021, arXiv:2104.09864) - see
# module docstring's "Positional encoding" section for why this (rather
# than a learned additive `nn.Embedding` table) is what makes the
# incremental KV-cache below both correct *and* fast. No table, no
# maximum-position bound: `_rope_cos_sin` computes each token's rotation
# straight from its own (arbitrarily large) position index.
# ----------------------------------------------------------------------
def _rope_cos_sin(positions: torch.Tensor, head_dim: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """`positions`: `(...)` long/float, any shape. Returns `(cos, sin)`,
    each `(..., head_dim)`, ready to combine with a `(..., head_dim)`
    query/key tensor via `_apply_rope`'s rotate-half convention."""
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32, device=positions.device) / half))
    freqs = positions.float().unsqueeze(-1) * inv_freq  # (..., half)
    emb = torch.cat([freqs, freqs], dim=-1)  # (..., head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """`x`: `(..., head_dim)`. `cos`/`sin`: broadcastable to `x`'s shape
    (e.g. `x` is `(B, H, L, D)` and `cos`/`sin` are `(B, 1, L, D)`)."""
    return x * cos + _rotate_half(x) * sin


# ----------------------------------------------------------------------
# Incremental KV-cache (see module docstring's "Persistent,
# incrementally-extended per-lane root cache" section for the overall
# design). Two call conventions exist side by side on every
# Transformer-touching class below:
#   - `forward`/`forward(...)` (no suffix): the *full-sequence* path, one
#     matmul over a whole (padded, right-padded/compact-positions) batch
#     of token sequences at once - used only for training's teacher-
#     forcing forward pass, which fundamentally needs every position's
#     hidden state from one pass for gradients to flow correctly, and
#     gains nothing from a cache since nothing is "reused" across calls.
#   - `forward_incremental_batch` (this section): appends *exactly one*
#     new token per lane to an existing (possibly `None`/empty) cache and
#     returns only that new token's hidden state - used for everything
#     that legitimately reuses previously-computed attention: every real
#     step's persistent per-lane cache (`NativeUniZero._advance_lane_caches`),
#     every reanalyze sample's one-off cold-start context replay
#     (`NativeUniZero._replay_context_to_cache`), and search-tree
#     expansion on top of either root (`NativeUniZero._step_imagine`).
# A token's position for RoPE purposes is a cache's `next_pos` (see
# `_TransformerCache`) - monotonic, never rewound by eviction, unlike
# `length` (the cache's current *physical* token count).
# ----------------------------------------------------------------------
class _LayerKV:
    """One Transformer layer's cached keys/values for one lane or search
    node. `k`/`v`: `(1, num_heads, length, head_dim)` - batch dim is
    always exactly 1 (one cache belongs to one lane/node; cross-lane
    batching for one incremental step happens by *padding* several
    `_LayerKV`s together inside `_CausalSelfAttention.forward_incremental_batch`,
    not by giving a `_LayerKV` itself a batch dimension > 1)."""

    __slots__ = ("k", "v")

    def __init__(self, k: torch.Tensor, v: torch.Tensor) -> None:
        self.k = k
        self.v = v


class _TransformerCache:
    """One lane's or one search node's incremental KV-cache across every
    Transformer layer. Crucially, `forward_incremental_batch` never
    mutates a `_TransformerCache` it's given - `torch.cat`, which is how
    every layer extends `k`/`v`, always allocates a fresh output tensor
    and never writes back into its inputs - so it always *returns* a
    brand-new `_TransformerCache` rather than editing one in place. That
    means the exact same parent cache object can be hand to many
    `forward_incremental_batch` calls "in parallel" (e.g. every candidate
    action expanding the same search node within one simulation) with no
    aliasing risk and no explicit `.clone()` needed anywhere - the
    reference's "deep-clone KV on retrieve" requirement falls out for
    free from this class simply never being mutated to begin with.

    `length` (physical token count, shrinks on `evict_front`) and
    `next_pos` (the RoPE position the *next* appended token will get,
    monotonic, never shrinks) are deliberately two separate counters -
    see module docstring's "Positional encoding" section for why eviction
    only needs to touch the former."""

    __slots__ = ("layers", "length", "next_pos")

    def __init__(self, num_layers: int) -> None:
        self.layers: list[_LayerKV | None] = [None] * num_layers
        self.length = 0
        self.next_pos = 0

    def evict_front(self, keep_last: int) -> None:
        """Drops every physically-cached token except the most recent
        `keep_last` - a plain tensor slice. Lossless *positionally* under
        RoPE (module docstring's "Positional encoding" section): a
        surviving cached key's rotation was baked in at write time and
        depends only on its own `next_pos` at that time, never on what
        else shares the cache, so dropping unrelated old entries perturbs
        nothing about the *relative offsets* any future query will see.
        Not lossless in an absolute sense for `num_layers > 1`, though: a
        surviving token's own hidden state (and thus its K/V at every
        layer above the first) was computed *before* eviction, while the
        now-dropped tokens were still attendable, so a little of their
        influence remains baked into the survivors via the residual
        stream - unavoidable for any multi-layer causal-attention KV-cache
        eviction scheme, this file's own or the reference's, and
        independent of positional encoding entirely (see module
        docstring's "Positional encoding" section, second bullet, for why
        that's an accepted trade rather than a bug). `next_pos` itself is
        untouched; eviction only shrinks physical storage, never rewinds
        the position counter new tokens keep counting up from."""
        if keep_last >= self.length:
            return
        drop = self.length - keep_last
        for layer in self.layers:
            if layer is not None:
                layer.k = layer.k[:, :, drop:, :]
                layer.v = layer.v[:, :, drop:, :]
        self.length = keep_last


class _CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, attend_mask: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """`x`: `(B, L, E)`. `attend_mask`: `(B, L, L)` bool, `True` where
        query position `i` is allowed to attend to key position `j`
        (already combines causality with key-padding - see
        `_build_attend_mask`). `positions`: `(B, L)` long - each token's
        RoPE position (module docstring's "Positional encoding" section);
        rotates `q`/`k` only, never `v`, before the dot product."""
        b, seq_len, embed_dim = x.shape
        qkv = self.qkv(x).view(b, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos, sin = _rope_cos_sin(positions, self.head_dim)  # (B, L, D)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)  # (B, 1, L, D) broadcasts over heads
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        mask = attend_mask.unsqueeze(1)  # (B, 1, L, L) broadcasts over heads
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(b, seq_len, embed_dim)
        return self.proj(out)

    def forward_incremental_batch(
        self, x_new: torch.Tensor, positions: torch.Tensor, layer_caches: list[_LayerKV | None],
    ) -> tuple[torch.Tensor, list[_LayerKV]]:
        """`x_new`: `(B, E)` - exactly one new token per lane (already
        LN'd by the caller, same as the full-sequence path's `ln1(x)`).
        `positions`: `(B,)` long - this new token's own RoPE position
        (each lane's cache's `next_pos`, module docstring's "Positional
        encoding" section); already-cached keys need no re-rotation -
        their rotation was baked in when *they* were the new token.
        `layer_caches[i]`: lane `i`'s existing `(1, H, L_i, D)` K/V for
        *this* layer, or `None` for an empty cache - `L_i` may differ
        across lanes (different lanes/nodes have different history
        depths), so every lane's K/V is padded to this call's own common
        max length for one batched attention call; the query is always
        the newest token for its lane, so it's unconditionally allowed to
        attend to every one of its own lane's real cached keys (no
        causal masking needed beyond padding - nothing is *later* than
        the newest token, so there's nothing to hide from it)."""
        b, embed_dim = x_new.shape
        qkv = self.qkv(x_new).view(b, 3, self.num_heads, self.head_dim)
        q, k_new, v_new = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each (B, H, D)
        cos, sin = _rope_cos_sin(positions, self.head_dim)  # (B, D)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)  # (B, 1, D) broadcasts over heads
        q, k_new = _apply_rope(q, cos, sin), _apply_rope(k_new, cos, sin)
        lens = [0 if c is None else c.k.shape[2] for c in layer_caches]
        lmax = max(lens) if lens else 0
        k_all = torch.zeros(b, self.num_heads, lmax + 1, self.head_dim, device=x_new.device, dtype=x_new.dtype)
        v_all = torch.zeros_like(k_all)
        pad_mask = torch.zeros(b, lmax + 1, dtype=torch.bool, device=x_new.device)
        for i in range(b):
            n = lens[i]
            if n:
                k_all[i, :, :n] = layer_caches[i].k[0]
                v_all[i, :, :n] = layer_caches[i].v[0]
            k_all[i, :, n] = k_new[i]
            v_all[i, :, n] = v_new[i]
            pad_mask[i, : n + 1] = True
        mask = pad_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, Lmax+1) - one query per lane
        out = F.scaled_dot_product_attention(
            q.unsqueeze(2), k_all, v_all, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.squeeze(2).reshape(b, embed_dim)
        out = self.proj(out)
        new_layer_caches = [
            _LayerKV(k_all[i : i + 1, :, : lens[i] + 1].clone(), v_all[i : i + 1, :, : lens[i] + 1].clone())
            for i in range(b)
        ]
        return out, new_layer_caches


class _TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = _CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(), nn.Linear(4 * embed_dim, embed_dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attend_mask: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attend_mask, positions)
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_incremental_batch(
        self, x: torch.Tensor, positions: torch.Tensor, layer_caches: list[_LayerKV | None],
    ) -> tuple[torch.Tensor, list[_LayerKV]]:
        """`x`: `(B, E)` - one new token per lane's residual stream."""
        attn_out, new_layer_caches = self.attn.forward_incremental_batch(self.ln1(x), positions, layer_caches)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_layer_caches


def _build_attend_mask(pad_mask: torch.Tensor) -> torch.Tensor:
    """`pad_mask`: `(B, L)` bool, `True` = real token, `False` = leading
    context padding (a token slot that predates this episode's start -
    never appended-to-then-later-invalid, since real tokens are only ever
    appended at the *end* of a node's sequence, so once a position is real
    it stays real for the rest of that sequence's life). Combines with a
    plain lower-triangular causal mask so no query ever attends to a pad
    position *or* a future position."""
    b, seq_len = pad_mask.shape
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=pad_mask.device))
    key_ok = pad_mask.unsqueeze(1).expand(b, seq_len, seq_len)
    mask = causal.unsqueeze(0) & key_ok
    # A pad *query* position (there's no such thing as a pad key that's
    # also never a query - every position is both) whose row has zero
    # allowed keys would make softmax divide 0/0 -> NaN, which then
    # poisons every *valid* query's output too (a masked key's attention
    # *weight* is exactly 0, but `0 * NaN` is still `NaN` in IEEE754 - the
    # classic masked-attention footgun). Force every position to always
    # be able to attend to itself so its own row is never all-`False`;
    # harmless for real (non-pad) positions since diagonal attention was
    # already implied by causality, and pad positions' own (now finite,
    # if meaningless) output is never read by anything - `key_ok` still
    # keeps every *other* query from ever attending to a pad key.
    eye = torch.eye(seq_len, dtype=torch.bool, device=pad_mask.device).unsqueeze(0)
    return mask | eye


class _CausalTransformer(nn.Module):
    def __init__(self, embed_dim: int, num_layers: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        head_dim = embed_dim // num_heads
        assert head_dim % 2 == 0, "RoPE needs an even head_dim (embed_dim / num_heads)"
        self.embed_dim = embed_dim
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([_TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)])
        self.ln_out = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor, pad_mask: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """`tokens`: `(B, L, E)` raw embeddings. `pad_mask`: `(B, L)` bool,
        `True` = real. `positions`: `(B, L)` long, or `None` to default to
        plain `0..L-1` for every lane - safe under RoPE regardless of what
        absolute position the same tokens would carry anywhere else (see
        module docstring's "Positional encoding" section). Returns
        post-Transformer hidden states, same shape as `tokens`."""
        b, seq_len, _ = tokens.shape
        if positions is None:
            positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(b, seq_len)
        x = self.drop(tokens)
        attend_mask = _build_attend_mask(pad_mask)
        for block in self.blocks:
            x = block(x, attend_mask, positions)
        return self.ln_out(x)

    def forward_incremental_batch(
        self, token_embeds: torch.Tensor, positions: torch.Tensor, caches: list[_TransformerCache | None],
    ) -> tuple[torch.Tensor, list[_TransformerCache]]:
        """`token_embeds`: `(B, E)` - one new raw token per lane.
        `positions`: `(B,)` long - this token's own RoPE position;
        normally just `caches[i].next_pos` (or `0` if `None`), but passed
        explicitly since `_replay_context_to_cache` sometimes needs to
        skip a lane for one round (see that method). `caches[i]`: lane
        `i`'s existing cache, or `None`. Returns the new tokens'
        post-Transformer hidden states (`(B, E)`) and one fresh
        `_TransformerCache` per lane - the ones passed in are never
        mutated (see `_TransformerCache`'s own docstring)."""
        b = token_embeds.shape[0]
        num_layers = len(self.blocks)
        x = self.drop(token_embeds)
        layer_caches_by_layer = [[c.layers[i] if c is not None else None for c in caches] for i in range(num_layers)]
        new_layer_caches_by_layer: list[list[_LayerKV]] = []
        for i, block in enumerate(self.blocks):
            x, new_layer_caches = block.forward_incremental_batch(x, positions, layer_caches_by_layer[i])
            new_layer_caches_by_layer.append(new_layer_caches)
        hidden = self.ln_out(x)
        new_caches: list[_TransformerCache] = []
        for lane in range(b):
            cache = _TransformerCache(num_layers)
            parent = caches[lane]
            cache.length = (0 if parent is None else parent.length) + 1
            # `positions[lane]` (this new token's own RoPE position), not
            # `parent.next_pos`, is authoritative - a caller may pass a
            # position that doesn't equal the parent's own `next_pos`
            # (see `_replay_context_to_cache`'s per-round `active` lanes).
            cache.next_pos = int(positions[lane].item()) + 1
            for i in range(num_layers):
                cache.layers[i] = new_layer_caches_by_layer[i][lane]
            new_caches.append(cache)
        return hidden, new_caches


# ----------------------------------------------------------------------
# Heads - read off specific token positions' hidden states (see module
# docstring's "Head placement" bullet).
# ----------------------------------------------------------------------
class _Heads(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, action_space: gym.Space, support_size: int) -> None:
        super().__init__()
        self.discrete = isinstance(action_space, gym.spaces.Discrete)
        self.support_size = support_size
        self.trunk = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ELU())
        self.value_head = nn.Linear(hidden_dim, 2 * support_size + 1)
        self.reward_head = nn.Linear(hidden_dim, 2 * support_size + 1)
        # Predicted *raw* next-token embedding - fed straight back in as
        # the imagined next "obs" token during search (see
        # `_step_imagine`); trained via the consistency loss in
        # `_train_step` to actually approximate what `_Tokenizer` would
        # have produced from the real next observation.
        self.latent_head = nn.Linear(hidden_dim, embed_dim)
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

    def value_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.trunk(h))

    def value(self, h: torch.Tensor) -> torch.Tensor:
        return _logits_to_scalar(self.value_logits(h), self.support_size)

    def reward_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.reward_head(self.trunk(h))

    def reward(self, h: torch.Tensor) -> torch.Tensor:
        return _logits_to_scalar(self.reward_logits(h), self.support_size)

    def latent(self, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.latent_head(self.trunk(h)))

    def policy_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.policy_head(self.trunk(h))

    def policy_gaussian(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.trunk(h)
        mean = torch.tanh(self.mean_head(t)) * self.action_scale + self.action_bias
        log_std = self.log_std_head(t).clamp(-5.0, 1.0)
        return mean, log_std.exp()


class _Projector(nn.Module):
    """SimSiam-style projector, ported verbatim from `efficientzero.py`
    (see that file's own docstring for why `BatchNorm1d` here isn't
    decorative - stop-gradient alone still collapses in practice)."""

    def __init__(self, embed_dim: int, proj_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, proj_dim), nn.BatchNorm1d(proj_dim), nn.ELU(), nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _Predictor(nn.Module):
    def __init__(self, proj_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proj_dim, proj_dim), nn.BatchNorm1d(proj_dim), nn.ELU(), nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ----------------------------------------------------------------------
# Gumbel search tree - bookkeeping ported verbatim from `efficientzero.py`
# (Danihelka, Guez, Schrittwieser & Silver, ICLR 2022,
# https://openreview.net/pdf?id=bERaNdoegnO) with one change: a node's
# "state" is its own incremental KV-cache (`cache`, extended by two
# tokens - one action, one predicted-observation - per tree edge, see
# `_step_imagine`) instead of a fixed-size latent vector, and a node's
# reward is a plain per-edge scalar (no LSTM "value prefix" cumulative-sum
# trick - see module docstring).
# ----------------------------------------------------------------------
class _MinMaxStats:
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
    __slots__ = (
        "prior", "parent", "children", "visit_count", "value_sum",
        "reward_value", "cache", "candidate_action", "selected_children_idx",
    )

    def __init__(self, prior: float, parent: "_SearchNode | None" = None) -> None:
        self.prior = prior
        self.parent = parent
        self.children: list[_SearchNode] = []
        self.visit_count = 0
        self.value_sum = 0.0
        self.reward_value = 0.0
        self.cache: _TransformerCache | None = None  # this node's own KV-cache (see `_step_imagine`)
        self.candidate_action: Any = None  # continuous only: sampled action leading to this node
        self.selected_children_idx: list[int] = []  # root only: Sequential Halving survivors

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def reward(self) -> float:
        return self.reward_value

    def children_visit_sum(self) -> int:
        return sum(c.visit_count for c in self.children)

    def q_of_child(self, idx: int, discount: float) -> float:
        child = self.children[idx]
        return child.reward() + discount * child.value()

    def v_mix(self, discount: float) -> float:
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
        self, priors: np.ndarray, cache: _TransformerCache | None, reward_value: float,
        candidate_actions: list[Any] | None = None,
    ) -> None:
        self.cache = cache
        self.reward_value = reward_value
        self.children = [_SearchNode(float(p), self) for p in priors]
        if candidate_actions is not None:
            for child, action in zip(self.children, candidate_actions):
                child.candidate_action = action


def _transformed_completed_qs(node: _SearchNode, minmax: _MinMaxStats, discount: float, c_visit: float, c_scale: float) -> np.ndarray:
    completed = node.completed_qs(discount, minmax)
    max_visit = max((c.visit_count for c in node.children), default=0)
    return (c_visit + max_visit) * c_scale * completed


def _select_action(node: _SearchNode, minmax: _MinMaxStats, discount: float, c_visit: float, c_scale: float) -> int:
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


# ----------------------------------------------------------------------
# Replay buffer - PER + reanalyze support ported verbatim from
# `efficientzero.py`'s `_EfficientZeroBuffer` (see that class's own
# docstring), extended with `context_obs`/`context_action`/`context_valid`
# slicing so every sample also carries the *preceding* real context a
# Transformer window needs (episodes are stored as contiguous arrays, so
# this is a pure slice - no extra storage).
# ----------------------------------------------------------------------
class _UniZeroBuffer:
    _NO_SEARCH_VALUE = -1e9

    def __init__(
        self, capacity_episodes: int, obs_shape: tuple[int, ...], action_dim: int, policy_target_dim: int,
        context_length: int, num_lanes: int = 1, priority_alpha: float = 1.0, priority_beta: float = 1.0,
        min_priority: float = 1e-6,
    ) -> None:
        self.capacity = max(1, capacity_episodes)
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.policy_target_dim = policy_target_dim
        self.context_length = max(0, context_length)
        self.priority_alpha = max(0.0, priority_alpha)
        self.priority_beta = max(0.0, priority_beta)
        self.min_priority = max(1e-8, min_priority)
        self.episodes: list[dict[str, np.ndarray]] = []
        self._episode_alpha_sum: list[float] = []
        self._max_priority = 1.0
        self._cur: list[dict[str, list[Any]]] = [self._new_episode() for _ in range(max(1, num_lanes))]

    @staticmethod
    def _new_episode() -> dict[str, list[Any]]:
        return {"obs": [], "action": [], "reward": [], "next_obs": [], "policy_target": []}

    def _pad_one(self, obs_hist: list[np.ndarray], action_hist: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Right-pads (valid transitions packed at the *front*, in
        temporal order; padding, if any, trails at the end) rather than
        the more common "left-pad so the most recent is always at a fixed
        trailing slot" scheme - deliberately, so that a token's position
        (`_CausalTransformer`'s positional-embedding index) can always
        just be its plain array index with *no* separate position lookup,
        in both this class's full-sequence callers (`NativeUniZero.
        _embed_root_batch`/`_train_step`) and the incremental KV-cache
        path (`NativeUniZero._replay_context_to_cache`) - a
        short-history lane's real tokens get the exact same position
        numbers a long-history lane's corresponding real tokens would.
        Harmless either way under RoPE (module docstring's "Positional
        encoding" section - only *relative* offsets between tokens in one
        attention computation matter), but kept for the same reason it
        always was: one less thing to keep in sync between callers."""
        length = self.context_length
        ctx_obs = np.zeros((length, *self.obs_shape), dtype=np.float32)
        ctx_action = np.zeros((length, self.action_dim), dtype=np.float32)
        ctx_valid = np.zeros(length, dtype=bool)
        n = len(obs_hist)
        if n:
            ctx_obs[:n] = np.stack(obs_hist)
            ctx_action[:n] = np.stack(action_hist)
            ctx_valid[:n] = True
        return ctx_obs, ctx_action, ctx_valid

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
        ep_len = len(cur["obs"])
        episode = {
            "obs": np.stack(cur["obs"]),
            "action": np.stack(cur["action"]),
            "reward": np.asarray(cur["reward"], dtype=np.float32),
            "next_obs": np.stack(cur["next_obs"]),
            "policy_target": np.stack(cur["policy_target"]),
            "priority": np.full(ep_len, self._max_priority, dtype=np.float64),
            "search_value": np.full(ep_len, self._NO_SEARCH_VALUE, dtype=np.float32),
        }
        self.episodes.append(episode)
        self._episode_alpha_sum.append(float(np.sum(episode["priority"] ** self.priority_alpha)))
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)
            self._episode_alpha_sum.pop(0)
        self._cur[lane] = self._new_episode()

    def __len__(self) -> int:
        return sum(len(ep["reward"]) for ep in self.episodes) + sum(len(cur["reward"]) for cur in self._cur)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    def _context_for(self, ep: dict[str, np.ndarray], start: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        length = self.context_length
        lo = max(0, start - length)
        obs_hist = list(ep["obs"][lo:start])
        action_hist = list(ep["action"][lo:start])
        return self._pad_one(obs_hist, action_hist)

    def sample(self, batch_size: int, unroll_steps: int, td_steps: int, gamma: float) -> dict[str, np.ndarray]:
        length = self.context_length
        ctx_obs = np.zeros((batch_size, length, *self.obs_shape), dtype=np.float32)
        ctx_action = np.zeros((batch_size, length, self.action_dim), dtype=np.float32)
        ctx_valid = np.zeros((batch_size, length), dtype=bool)
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
        search_value = np.full((batch_size, unroll_steps), self._NO_SEARCH_VALUE, dtype=np.float32)
        episode_idx = np.zeros(batch_size, dtype=np.int64)
        timestep = np.zeros(batch_size, dtype=np.int64)
        sample_prob = np.zeros(batch_size, dtype=np.float64)

        n_episodes = len(self.episodes)
        alpha_sums = np.asarray(self._episode_alpha_sum, dtype=np.float64)
        total_alpha_sum = float(alpha_sums.sum())
        if total_alpha_sum > 0:
            ep_probs = alpha_sums / total_alpha_sum
        else:
            ep_probs = np.full(n_episodes, 1.0 / n_episodes)
        ep_choices = np.random.choice(n_episodes, size=batch_size, p=ep_probs)
        n_total_transitions = max(1, len(self))

        for b in range(batch_size):
            ep_i = int(ep_choices[b])
            ep = self.episodes[ep_i]
            t_len = len(ep["reward"])
            local_weight = ep["priority"] ** self.priority_alpha
            local_sum = float(local_weight.sum())
            if local_sum > 0:
                local_probs = local_weight / local_sum
                start = int(np.random.choice(t_len, p=local_probs))
                local_prob = float(local_probs[start])
            else:
                start = int(np.random.randint(t_len))
                local_prob = 1.0 / t_len
            episode_idx[b] = ep_i
            timestep[b] = start
            sample_prob[b] = max(ep_probs[ep_i] * local_prob, 1e-12)

            ctx_obs[b], ctx_action[b], ctx_valid[b] = self._context_for(ep, start)
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
                search_value[b, k] = ep["search_value"][idx]

                n = min(td_steps, t_len - idx)
                td_sum, discount = 0.0, 1.0
                for j in range(n):
                    td_sum += discount * float(ep["reward"][idx + j])
                    discount *= gamma
                td_reward[b, k] = td_sum
                td_discount[b, k] = discount
                landing_idx = idx + n - 1
                td_obs[b, k] = ep["next_obs"][landing_idx]
                td_bootstrap_mask[b, k] = 0.0 if landing_idx == t_len - 1 else 1.0

        is_weight = (1.0 / (n_total_transitions * sample_prob)) ** self.priority_beta
        is_weight = is_weight / max(float(is_weight.max()), 1e-12)

        return {
            "context_obs": ctx_obs, "context_action": ctx_action, "context_valid": ctx_valid,
            "obs0": obs0, "action": action, "reward": reward, "next_obs": next_obs,
            "policy_target": policy_target, "mask": mask, "td_reward": td_reward,
            "td_obs": td_obs, "td_discount": td_discount, "td_bootstrap_mask": td_bootstrap_mask,
            "search_value": search_value, "episode_idx": episode_idx, "timestep": timestep,
            "is_weight": is_weight.astype(np.float32),
        }

    def update_priorities(self, episode_idx: np.ndarray, timestep: np.ndarray, values: np.ndarray) -> None:
        values = np.maximum(np.asarray(values, dtype=np.float64), self.min_priority)
        touched = set()
        for ep_i, t, v in zip(episode_idx.tolist(), timestep.tolist(), values.tolist()):
            if not (0 <= ep_i < len(self.episodes)):
                continue
            self.episodes[ep_i]["priority"][t] = v
            touched.add(ep_i)
        self._max_priority = max(self._max_priority, float(values.max()) if len(values) else self._max_priority)
        for ep_i in touched:
            ep = self.episodes[ep_i]
            self._episode_alpha_sum[ep_i] = float(np.sum(ep["priority"] ** self.priority_alpha))

    def sample_for_reanalyze(
        self, n: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_episodes = len(self.episodes)
        n = min(n, sum(len(ep["reward"]) for ep in self.episodes))
        length = self.context_length
        if n <= 0:
            empty_obs = np.zeros((0, *self.obs_shape), dtype=np.float32)
            return (
                np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), empty_obs,
                np.zeros((0, length, *self.obs_shape), dtype=np.float32), np.zeros((0, length, self.action_dim), dtype=np.float32),
                np.zeros((0, length), dtype=bool),
            )
        lengths = np.asarray([len(ep["reward"]) for ep in self.episodes], dtype=np.float64)
        ep_choices = np.random.choice(n_episodes, size=n, p=lengths / lengths.sum())
        episode_idx = np.zeros(n, dtype=np.int64)
        timestep = np.zeros(n, dtype=np.int64)
        obs_out = np.zeros((n, *self.obs_shape), dtype=np.float32)
        ctx_obs = np.zeros((n, length, *self.obs_shape), dtype=np.float32)
        ctx_action = np.zeros((n, length, self.action_dim), dtype=np.float32)
        ctx_valid = np.zeros((n, length), dtype=bool)
        for i, ep_i in enumerate(ep_choices.tolist()):
            ep = self.episodes[ep_i]
            t = int(np.random.randint(len(ep["reward"])))
            episode_idx[i] = ep_i
            timestep[i] = t
            obs_out[i] = ep["obs"][t]
            ctx_obs[i], ctx_action[i], ctx_valid[i] = self._context_for(ep, t)
        return episode_idx, timestep, obs_out, ctx_obs, ctx_action, ctx_valid

    def update_reanalyzed_targets(
        self, episode_idx: np.ndarray, timestep: np.ndarray, policy_targets: np.ndarray, search_values: np.ndarray,
    ) -> None:
        for i, (ep_i, t) in enumerate(zip(episode_idx.tolist(), timestep.tolist())):
            if not (0 <= ep_i < len(self.episodes)):
                continue
            ep = self.episodes[ep_i]
            if t >= len(ep["reward"]):
                continue
            ep["policy_target"][t] = policy_targets[i]
            ep["search_value"][t] = search_values[i]


class NativeUniZero(CustomAlgorithm):
    def __init__(self, env: gym.Env, hyperparams: dict[str, Any], seed: int | None, device: str) -> None:
        super().__init__(env, hyperparams, seed, device)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self._obs_space = venv_obs_space(env)
        self._action_space = venv_action_space(env)
        self.discrete = isinstance(self._action_space, gym.spaces.Discrete)
        self.n_actions = int(self._action_space.n) if self.discrete else None
        self.action_dim = obs_flat_dim(self._action_space)

        self.embed_dim = int(hyperparams.get("embed_dim", 128))
        num_layers = max(1, int(hyperparams.get("num_layers", 2)))
        num_heads = max(1, int(hyperparams.get("num_heads", 4)))
        dropout = float(hyperparams.get("dropout", 0.0))
        self.context_length = max(0, int(hyperparams.get("context_length", 6)))
        self.unroll_steps = max(1, int(hyperparams.get("unroll_steps", 5)))
        self.support_size = max(1, int(hyperparams.get("value_support_size", 300)))

        self.tokenizer = _Tokenizer(self._obs_space, self.embed_dim, 2 * self.embed_dim).to(device)
        self.action_embed = _ActionEmbed(self.action_dim, self.embed_dim).to(device)
        self.transformer = _CausalTransformer(self.embed_dim, num_layers, num_heads, dropout).to(device)
        self.heads = _Heads(self.embed_dim, 2 * self.embed_dim, self._action_space, self.support_size).to(device)
        self.projector = _Projector(self.embed_dim, self.embed_dim).to(device)
        self.predictor = _Predictor(self.embed_dim).to(device)
        self._params = (
            list(self.tokenizer.parameters()) + list(self.action_embed.parameters())
            + list(self.transformer.parameters()) + list(self.heads.parameters())
            + list(self.projector.parameters()) + list(self.predictor.parameters())
        )
        self.optimizer = torch.optim.Adam(self._params, lr=float(hyperparams.get("learning_rate", 2e-4)))

        self.gamma = float(hyperparams.get("gamma", 0.99))
        self.batch_size = int(hyperparams.get("batch_size", 64))
        self.td_steps = max(1, int(hyperparams.get("td_steps", 5)))
        self.num_sampled_actions = max(2, int(hyperparams.get("num_sampled_actions", 8)))
        self.num_simulations = max(2, int(hyperparams.get("num_simulations", 32)))
        self.num_top_actions = max(2, int(hyperparams.get("num_top_actions", 8)))
        self.c_visit = float(hyperparams.get("c_visit", 50.0))
        self.c_scale = float(hyperparams.get("c_scale", 0.1))
        self.value_minmax_delta = float(hyperparams.get("value_minmax_delta", 0.01))
        self.value_loss_coef = float(hyperparams.get("value_loss_coef", 0.25))
        self.policy_loss_coef = float(hyperparams.get("policy_loss_coef", 1.0))
        self.reward_loss_coef = float(hyperparams.get("reward_loss_coef", 1.0))
        self.consistency_loss_coef = float(hyperparams.get("consistency_loss_coef", 2.0))
        self.continuous_prior_scale = float(hyperparams.get("continuous_prior_scale", 2.5))
        self.policy_target_temperature = max(1e-6, float(hyperparams.get("policy_target_temperature", 1.0)))
        self.train_freq = max(1, int(hyperparams.get("train_freq", 1)))
        self.train_steps_per_iter = max(1, int(hyperparams.get("train_steps_per_iter", 1)))
        self.learning_starts = int(hyperparams.get("learning_starts", 500))
        self.max_grad_norm = float(hyperparams.get("max_grad_norm", 5.0))
        self.priority_alpha = max(0.0, float(hyperparams.get("priority_alpha", 1.0)))
        self.priority_beta = max(0.0, float(hyperparams.get("priority_beta", 1.0)))
        self.min_priority = max(1e-8, float(hyperparams.get("min_priority", 1e-6)))
        self.reanalyze_freq = max(1, int(hyperparams.get("reanalyze_freq", 200)))
        self.reanalyze_batch_size = max(0, int(hyperparams.get("reanalyze_batch_size", 64)))

        sample_obs_arr = obs_to_array(self._obs_space.sample(), self._obs_space)
        self._obs_shape = sample_obs_arr.shape
        policy_target_dim = self.n_actions if self.discrete else self.action_dim
        self.buffer = _UniZeroBuffer(
            capacity_episodes=int(hyperparams.get("buffer_size", 2_000)),
            obs_shape=self._obs_shape, action_dim=self.action_dim, policy_target_dim=policy_target_dim,
            context_length=self.context_length, num_lanes=num_envs_of(env),
            priority_alpha=self.priority_alpha, priority_beta=self.priority_beta, min_priority=self.min_priority,
        )
        self._last_metrics: dict[str, float] = {}
        # `learn()`'s own persistent per-lane real-history cache (module
        # docstring's "Persistent, incrementally-extended per-lane root
        # cache" section) - `None` per lane means "no history yet this
        # episode" (the reference's own "first_step_flag").
        self._lane_cache: list[_TransformerCache | None] = [None] * num_envs_of(env)
        # `predict()`'s own analogous single-lane persistent cache,
        # separate from `self.buffer`'s (which only tracks lanes actually
        # being trained on) - reset on `episode_start=True` exactly like
        # every other memory-carrying algorithm in this app (see
        # `dreamer.py`'s `predict()`).
        self._eval_cache: _TransformerCache | None = None

    # ------------------------------------------------------------------
    # Token-sequence construction.
    # ------------------------------------------------------------------
    def _embed_root_batch(
        self, ctx_obs: np.ndarray, ctx_action: np.ndarray, ctx_valid: np.ndarray, current_obs: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """`ctx_*`: `(B, context_length, ...)`, right-padded (see
        `_UniZeroBuffer._pad_one`). `current_obs`: `(B, *obs_shape)`.
        Returns `(tokens (B, Lmax, E), pad_mask (B, Lmax) bool, valid_len
        (B,) int)`, `Lmax = valid_len.max()` - the raw root token
        sequence the *training* teacher-forcing window starts from (used
        only there - every other caller needing a root state, real
        steps/search/reanalyze alike, uses the cache-based
        `_replay_context_to_cache`/`_advance_lane_caches`/`_step_imagine`
        path instead, module docstring's "Persistent, incrementally-
        extended per-lane root cache" section).
        `valid_len[i] = 2*n_i + 1` is lane `i`'s *own* real token count
        (`n_i` valid context transitions, plus the current-obs token);
        since lanes can have different amounts of real history, they
        occupy different-length prefixes of this shared, batch-padded
        tensor - `_train_step` needs `valid_len` to know where each
        lane's *own* next real token belongs (there's no single shared
        offset)."""
        b, length = ctx_obs.shape[0], self.context_length
        device = self.device
        n_valid = ctx_valid.sum(axis=1).astype(np.int64) if length > 0 else np.zeros(b, dtype=np.int64)
        valid_len = 2 * n_valid + 1
        lmax = int(valid_len.max()) if b > 0 else 1
        if length > 0:
            obs_flat = torch.as_tensor(ctx_obs.reshape(b * length, *self._obs_shape), dtype=torch.float32, device=device)
            obs_emb_all = self.tokenizer(obs_flat).view(b, length, self.embed_dim)
            act_flat = torch.as_tensor(ctx_action.reshape(b * length, self.action_dim), dtype=torch.float32, device=device)
            act_emb_all = self.action_embed(act_flat).view(b, length, self.embed_dim)
        cur_emb = self.tokenizer(torch.as_tensor(current_obs, dtype=torch.float32, device=device))
        tokens = torch.zeros(b, lmax, self.embed_dim, device=device)
        pad_mask = torch.zeros(b, lmax, dtype=torch.bool, device=device)
        for i in range(b):
            n = int(n_valid[i])
            if n > 0:
                tokens[i, 0 : 2 * n : 2] = obs_emb_all[i, :n]
                tokens[i, 1 : 2 * n : 2] = act_emb_all[i, :n]
            tokens[i, 2 * n] = cur_emb[i]
            pad_mask[i, : 2 * n + 1] = True
        return tokens, pad_mask, valid_len

    def _step_imagine(
        self, parent_caches: list[_TransformerCache | None], action: torch.Tensor,
    ) -> tuple[list[_TransformerCache], torch.Tensor, torch.Tensor]:
        """One simulated tree edge: two incremental steps (module
        docstring's "Persistent, incrementally-extended per-lane root
        cache" section) - append the chosen action's token, read the
        reward off its hidden state and predict the next *imagined*
        observation token from it, append that too, and read policy/value
        off its hidden state (module docstring's "Head placement" bullet).
        Each step attends only the one new query against its parent's
        already-computed keys/values. `parent_caches[i]`: lane `i`'s
        parent node's cache - shared freely across every child expanding
        the same parent this simulation, no explicit clone needed (see
        `_TransformerCache`'s own docstring). `search()` never inspects
        `parent_caches`/the returned child cache itself, just threads it
        through `_SearchNode.cache` and back into the next simulation's
        call here. Returns `(child_caches, reward_hidden, obs_hidden)` -
        hidden states, not final head outputs, so the caller picks
        discrete/continuous/value formatting itself."""
        positions1 = torch.as_tensor(
            [0 if c is None else c.next_pos for c in parent_caches], dtype=torch.long, device=self.device,
        )
        act_emb = self.action_embed(action)  # (B, E)
        h_act, caches1 = self.transformer.forward_incremental_batch(act_emb, positions1, parent_caches)
        z_pred = self.heads.latent(h_act)
        positions2 = positions1 + 1
        h_obs, caches2 = self.transformer.forward_incremental_batch(z_pred, positions2, caches1)
        return caches2, h_act, h_obs

    def _replay_context_to_cache(
        self, ctx_obs: np.ndarray, ctx_action: np.ndarray, ctx_valid: np.ndarray,
    ) -> list[_TransformerCache | None]:
        """Cold-start cache construction for a batch of *independent*
        historical (episode, timestep) samples with no live per-lane
        cache to reuse - `_reanalyze`'s samples, specifically (real env
        steps use `_advance_lane_caches`'s persistent cache instead - see
        module docstring's "Persistent, incrementally-extended per-lane
        root cache" section). Replays each lane's own valid (right-padded,
        temporal-order) context transitions one token at a time through
        the exact same incremental machinery every other caller uses.
        `ctx_*`: `(B, context_length, ...)`."""
        b, length = ctx_obs.shape[0], self.context_length
        caches: list[_TransformerCache | None] = [None] * b
        if length == 0:
            return caches
        n_valid = ctx_valid.sum(axis=1)
        max_n = int(n_valid.max()) if b > 0 else 0
        device = self.device
        for t in range(max_n):
            active = [i for i in range(b) if t < n_valid[i]]
            if not active:
                continue
            sub_caches = [caches[i] for i in active]
            positions_obs = torch.as_tensor(
                [0 if c is None else c.next_pos for c in sub_caches], dtype=torch.long, device=device,
            )
            obs_emb = self.tokenizer(torch.as_tensor(ctx_obs[active, t], dtype=torch.float32, device=device))
            _, sub_caches = self.transformer.forward_incremental_batch(obs_emb, positions_obs, sub_caches)
            positions_act = positions_obs + 1
            act_emb = self.action_embed(torch.as_tensor(ctx_action[active, t], dtype=torch.float32, device=device))
            _, sub_caches = self.transformer.forward_incremental_batch(act_emb, positions_act, sub_caches)
            for j, i in enumerate(active):
                caches[i] = sub_caches[j]
        return caches

    def _advance_lane_caches(
        self, caches: list[_TransformerCache | None], obs_batch: np.ndarray, action_flat: np.ndarray,
    ) -> list[_TransformerCache]:
        """Extends each lane's *persistent* real-history cache by exactly
        one (obs, action) pair - the two tokens every real env step
        contributes - then trims it back down to `2 * context_length`
        tokens if it grew past that (module docstring's "Persistent,
        incrementally-extended per-lane root cache" section). Called from
        `learn()`'s collection loop and `predict()` after `search()` has
        already appended `obs_batch` on top of the *previous* call's
        returned caches to pick this step's action - appending that same
        `obs_batch` again here (rather than reusing `search()`'s own
        root-cache) keeps this method a self-contained, order-independent
        step so callers don't need to thread an extra "cache right after
        the root observation, before any action" value out of `search()`;
        the cost is one extra incremental Transformer call per lane per
        real step, negligible next to `search()`'s own `num_simulations`
        calls."""
        obs_emb = self.tokenizer(torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device))
        positions_obs = torch.as_tensor(
            [0 if c is None else c.next_pos for c in caches], dtype=torch.long, device=self.device,
        )
        _, caches1 = self.transformer.forward_incremental_batch(obs_emb, positions_obs, caches)
        act_emb = self.action_embed(torch.as_tensor(action_flat, dtype=torch.float32, device=self.device))
        positions_act = positions_obs + 1
        _, caches2 = self.transformer.forward_incremental_batch(act_emb, positions_act, caches1)
        keep = 2 * self.context_length
        for cache in caches2:
            cache.evict_front(keep)
        return caches2

    # ------------------------------------------------------------------
    # Search - a genuine batched Gumbel search tree, see module docstring.
    # ------------------------------------------------------------------
    def _sample_continuous_candidates(self, mean: torch.Tensor, std: torch.Tensor) -> tuple[list[np.ndarray], np.ndarray]:
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

    def search(
        self, obs_batch: np.ndarray, root_caches: list[_TransformerCache | None], deterministic: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Batched Gumbel search - `obs_batch`: `(B, *obs_shape)`,
        `root_caches[i]`: lane `i`'s existing real-history cache (or
        `None`) - `learn()`/`predict()` pass their own persistent
        per-lane cache directly, `_reanalyze()` passes a fresh cold-start
        replay from `_replay_context_to_cache` (module docstring's
        "Persistent, incrementally-extended per-lane root cache" section)
        - never mutated here (see `_TransformerCache`'s own docstring), so
        it's always safe for the caller to keep using its own copy
        afterward. Mirrors `efficientzero.py`'s `search()` structure; see
        module docstring for what's different (a Transformer world model
        instead of a fixed latent state)."""
        batch_size = obs_batch.shape[0]
        det = np.zeros(batch_size, dtype=bool) if deterministic is None else np.broadcast_to(deterministic, (batch_size,))
        with torch.no_grad():
            obs_emb = self.tokenizer(torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device))
            positions0 = torch.as_tensor(
                [0 if c is None else c.next_pos for c in root_caches], dtype=torch.long, device=self.device,
            )
            h_root, root_state_out = self.transformer.forward_incremental_batch(obs_emb, positions0, root_caches)
            root_value = self.heads.value(h_root)
            if self.discrete:
                root_logits = self.heads.policy_logits(h_root)
            else:
                root_mean, root_std = self.heads.policy_gaussian(h_root)

        n_slots = self.n_actions if self.discrete else self.num_sampled_actions
        num_top_actions = max(2, min(self.num_top_actions, n_slots))
        roots: list[_SearchNode] = []
        for i in range(batch_size):
            root = _SearchNode(prior=1.0)
            if self.discrete:
                priors = root_logits[i].detach().cpu().numpy().astype(np.float64)
                root.expand(priors, root_state_out[i], 0.0)
            else:
                candidates, priors = self._sample_continuous_candidates(root_mean[i : i + 1], root_std[i : i + 1])
                root.expand(priors, root_state_out[i], 0.0, candidate_actions=candidates)
            root.visit_count = 1
            root.value_sum = float(root_value[i].item())
            roots.append(root)

        minmax_list = [_MinMaxStats(self.value_minmax_delta) for _ in range(batch_size)]
        gumbel = np.random.gumbel(size=(batch_size, n_slots)) * self.policy_target_temperature
        schedule = _HalvingSchedule(self.num_simulations, num_top_actions)

        for sim_idx in range(self.num_simulations):
            leaf_nodes: list[_SearchNode] = []
            parent_states: list[Any] = []
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
                parent_states.append(parent.cache)
                chosen_actions.append(node.candidate_action if not self.discrete else parent.children.index(node))
                search_paths.append(path)

            if self.discrete:
                idx_t = torch.as_tensor(chosen_actions, dtype=torch.long, device=self.device)
                action_t = F.one_hot(idx_t, num_classes=self.n_actions).float()
            else:
                action_t = torch.as_tensor(np.stack(chosen_actions), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                child_states, h_act, h_obs = self._step_imagine(parent_states, action_t)
                reward_scalar = self.heads.reward(h_act)
                next_value = self.heads.value(h_obs)
                if self.discrete:
                    next_logits = self.heads.policy_logits(h_obs)
                else:
                    next_mean, next_std = self.heads.policy_gaussian(h_obs)

            for lane in range(batch_size):
                leaf = leaf_nodes[lane]
                rv = float(reward_scalar[lane].item())
                if self.discrete:
                    priors = next_logits[lane].detach().cpu().numpy().astype(np.float64)
                    leaf.expand(priors, child_states[lane], rv)
                else:
                    candidates, priors = self._sample_continuous_candidates(next_mean[lane : lane + 1], next_std[lane : lane + 1])
                    leaf.expand(priors, child_states[lane], rv, candidate_actions=candidates)
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
    # Reanalyze - ported wholesale from `efficientzero.py`'s `_reanalyze`.
    # ------------------------------------------------------------------
    def _reanalyze(self) -> dict[str, float] | None:
        if self.reanalyze_batch_size <= 0 or self.buffer.num_episodes == 0:
            return None
        episode_idx, timestep, obs_batch, ctx_obs, ctx_action, ctx_valid = self.buffer.sample_for_reanalyze(
            self.reanalyze_batch_size,
        )
        if obs_batch.shape[0] == 0:
            return None
        # No persistent cache exists for these arbitrary historical
        # samples - cold-start one by replaying each sample's own real
        # context (module docstring's "Persistent, incrementally-extended
        # per-lane root cache" section).
        root_caches = self._replay_context_to_cache(ctx_obs, ctx_action, ctx_valid)
        results = self.search(obs_batch, root_caches, deterministic=np.zeros(obs_batch.shape[0], dtype=bool))
        policy_targets = np.stack([r["policy_target"] for r in results])
        search_values = np.asarray([r["value_target"] for r in results], dtype=np.float32)
        self.buffer.update_reanalyzed_targets(episode_idx, timestep, policy_targets, search_values)
        return {"reanalyze_mean_search_value": float(search_values.mean())}

    # ------------------------------------------------------------------
    # Training - one forward pass over the full teacher-forced window
    # (context + real interleaved obs/action tokens), reading every step's
    # losses off that single pass (module docstring's "Full-trajectory
    # teacher forcing" bullet).
    # ------------------------------------------------------------------
    def _train_step(self) -> dict[str, float]:
        batch = self.buffer.sample(self.batch_size, self.unroll_steps, self.td_steps, self.gamma)
        device = self.device
        b, k_steps = self.batch_size, self.unroll_steps
        length = self.context_length

        obs0 = torch.as_tensor(batch["obs0"], dtype=torch.float32, device=device)
        action = torch.as_tensor(batch["action"], dtype=torch.float32, device=device)
        reward = torch.as_tensor(batch["reward"], dtype=torch.float32, device=device)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=device)
        policy_target = torch.as_tensor(batch["policy_target"], dtype=torch.float32, device=device)
        mask = torch.as_tensor(batch["mask"], dtype=torch.float32, device=device)
        td_reward = torch.as_tensor(batch["td_reward"], dtype=torch.float32, device=device)
        td_obs = torch.as_tensor(batch["td_obs"], dtype=torch.float32, device=device)
        td_discount = torch.as_tensor(batch["td_discount"], dtype=torch.float32, device=device)
        td_bootstrap_mask = torch.as_tensor(batch["td_bootstrap_mask"], dtype=torch.float32, device=device)
        # `batch["search_value"]` (`_reanalyze()`'s periodically-refreshed
        # MCTS root value, per real transition) is deliberately *not*
        # read here any more - see the value-target comment below.
        is_weight = torch.as_tensor(batch["is_weight"], dtype=torch.float32, device=device)

        # Raw token embeddings for the whole window: context (compact,
        # `2*n_i` tokens for lane `i`'s own `n_i` valid context
        # transitions - see `_embed_root_batch`), then interleaved [obs0,
        # act0, next_obs0=obs1, act1, ..., obs_{K-1}, act_{K-1},
        # next_obs_{K-1}=obs_K] - `2*k_steps` more tokens. Lanes with
        # different amounts of real context (`valid_len`) occupy
        # different-length prefixes of this shared, batch-padded tensor -
        # each lane's *own* unroll tokens are placed right after its own
        # context ends, not at one fixed shared offset (see
        # `_embed_root_batch`'s docstring for why: with compact, no-gap
        # positions - the same convention the incremental KV-cache path
        # uses - a fixed shared offset would leave a position-numbering
        # gap for every lane whose own context is shorter than the
        # batch's longest).
        root_tokens, root_pad, valid_len = self._embed_root_batch(
            batch["context_obs"], batch["context_action"], batch["context_valid"], batch["obs0"],
        )  # (B, Lmax0, E); `valid_len[i]` = lane `i`'s own real length (incl. obs0)
        act_embs = self.action_embed(action.view(b * k_steps, self.action_dim)).view(b, k_steps, self.embed_dim)
        next_obs_embs_grad = self.tokenizer(next_obs.view(b * k_steps, *self._obs_shape)).view(b, k_steps, self.embed_dim)

        lmax0 = root_tokens.shape[1]
        total_len = lmax0 + 2 * k_steps
        tokens = torch.zeros(b, total_len, self.embed_dim, device=device)
        pad_mask = torch.zeros(b, total_len, dtype=torch.bool, device=device)
        tokens[:, :lmax0] = root_tokens
        pad_mask[:, :lmax0] = root_pad
        for i in range(b):
            base_i = int(valid_len[i])
            end_i = base_i + 2 * k_steps
            tokens[i, base_i:end_i:2] = act_embs[i]
            tokens[i, base_i + 1 : end_i : 2] = next_obs_embs_grad[i]
            pad_mask[i, base_i:end_i] = True

        hidden = self.transformer(tokens, pad_mask)
        # `obs0_pos[i]` = lane `i`'s own obs0 token position (its context's
        # last real slot) - differs per lane now, so every hidden-state
        # read below gathers by index instead of slicing a shared offset.
        obs0_pos = torch.as_tensor(valid_len - 1, dtype=torch.long, device=device)
        batch_idx = torch.arange(b, device=device)

        reward_loss = torch.zeros((), device=device)
        value_loss = torch.zeros((), device=device)
        policy_loss = torch.zeros((), device=device)
        consistency_loss = torch.zeros((), device=device)
        first_step_value_pred: torch.Tensor | None = None
        first_step_value_target: torch.Tensor | None = None

        for k in range(k_steps):
            m = mask[:, k]
            wm = m * is_weight
            denom = m.sum().clamp_min(1.0)
            h_obs_k = hidden[batch_idx, obs0_pos + 2 * k]  # obs_k's own hidden state
            h_act_k = hidden[batch_idx, obs0_pos + 2 * k + 1]  # act_k's own hidden state

            value_pred_logits = self.heads.value_logits(h_obs_k)
            # Bootstrap value comes from a *fresh* root-style forward over
            # the TD-landing observation with its own (empty) context -
            # cheap and exactly analogous to `efficientzero.py`'s own
            # `self.representation(td_obs[:, k])` bootstrap call. Context
            # is empty for *every* lane here, so every lane's `valid_len`
            # is uniformly `1` and `[:, -1, :]` unambiguously means "the
            # only (td_obs's own) token" - no per-lane offset needed.
            with torch.no_grad():
                td_ctx_valid = np.zeros((b, length), dtype=bool)
                td_root_tokens, td_root_pad, _td_valid_len = self._embed_root_batch(
                    np.zeros((b, length, *self._obs_shape), dtype=np.float32),
                    np.zeros((b, length, self.action_dim), dtype=np.float32),
                    td_ctx_valid, batch["td_obs"][:, k],
                )
                bootstrap_value = self.heads.value(self.transformer(td_root_tokens, td_root_pad)[:, -1, :]) * td_bootstrap_mask[:, k]
                td_target = td_reward[:, k] + td_discount[:, k] * bootstrap_value
                # Plain n-step TD(`td_steps`) target - *not* blended with
                # `search_value` via `max()`. The reference LightZero
                # UniZero (`lzero/mcts/buffer/game_buffer_unizero.py`'s
                # `_compute_target_reward_value`, `use_root_value=False`
                # by default - `lzero/mcts/buffer/game_buffer.py`) always
                # recomputes this bootstrap value fresh from the *current*
                # network at the landing observation (which is what
                # `bootstrap_value` above already is - "value is 100%
                # reanalyzed" per that file's own docstring) and never
                # blends it with a stale MCTS root/search value. Doing the
                # `max()` blend (borrowed from `efficientzero.py`, which
                # *is* correct for EfficientZero's own MCTS-heavy target)
                # here instead one-sidedly ratchets the target upward
                # whenever a lucky/overestimated search value briefly
                # exceeds the bootstrap TD value, which self-reinforces
                # (the network is trained *towards* its own noisy search
                # overestimates) and was the likely cause of the
                # "quick reward spike, then degrades" pattern reported
                # even with a from-scratch-recompute self-play path.
                value_target = td_target
            value_two_hot = _scalar_to_two_hot(value_target, self.support_size)
            v_loss = -(value_two_hot * F.log_softmax(value_pred_logits, dim=-1)).sum(-1)
            value_loss = value_loss + (v_loss * wm).sum() / denom
            if k == 0:
                with torch.no_grad():
                    first_step_value_pred = _logits_to_scalar(value_pred_logits, self.support_size).detach()
                    first_step_value_target = value_target.detach()

            reward_logits_k = self.heads.reward_logits(h_act_k)
            reward_two_hot = _scalar_to_two_hot(reward[:, k], self.support_size)
            r_loss = -(reward_two_hot * F.log_softmax(reward_logits_k, dim=-1)).sum(-1)
            reward_loss = reward_loss + (r_loss * wm).sum() / denom

            if self.discrete:
                logp = F.log_softmax(self.heads.policy_logits(h_obs_k), dim=-1)
                p_loss = -(policy_target[:, k] * logp).sum(-1)
            else:
                mean, _std = self.heads.policy_gaussian(h_obs_k)
                p_loss = F.mse_loss(mean, policy_target[:, k], reduction="none").sum(-1)
            policy_loss = policy_loss + (p_loss * wm).sum() / denom

            # SimSiam-style consistency: `latent_head(h_act_k)` should
            # approximate the *real* next observation's own raw token
            # embedding (already computed above as `next_obs_embs_grad`,
            # with-gradient, for the main sequence - detach a copy here as
            # the stop-gradient target) - see module docstring.
            z_pred = self.heads.latent(h_act_k)
            true_next = next_obs_embs_grad[:, k].detach()
            p_true = F.normalize(self.projector(true_next), dim=-1).detach()
            p_pred = F.normalize(self.predictor(self.projector(z_pred)), dim=-1)
            c_loss = -(p_true * p_pred).sum(-1)
            consistency_loss = consistency_loss + (c_loss * wm).sum() / denom

        n = float(k_steps)
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

        assert first_step_value_pred is not None and first_step_value_target is not None
        fresh_priority = (first_step_value_pred - first_step_value_target).abs().cpu().numpy()
        self.buffer.update_priorities(batch["episode_idx"], batch["timestep"], fresh_priority)

        return {
            "reward_loss": float(reward_loss.item()),
            "value_loss": float(value_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "consistency_loss": float(consistency_loss.item()),
            "mean_priority": float(fresh_priority.mean()) if fresh_priority.size else 0.0,
            "mean_is_weight": float(is_weight.mean().item()),
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
                # `self._lane_cache[lane]` is each lane's own persistent
                # real-history cache, carried over from the *previous*
                # real step (module docstring's "Persistent,
                # incrementally-extended per-lane root cache" section) -
                # no replay from `self.buffer` needed, unlike the
                # (removed) cold-start-every-step approach this file used
                # to take.
                results = self.search(obs_arr, self._lane_cache, deterministic=None)
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
            if num_timesteps >= self.learning_starts:
                new_caches = self._advance_lane_caches(self._lane_cache, obs_arr, np.stack(action_flats))
                for lane in range(n_envs):
                    self._lane_cache[lane] = None if dones[lane] else new_caches[lane]
            obs_arr = next_obs_arr
            ep_reward += rewards
            ep_length += 1
            prev_num_timesteps = num_timesteps
            num_timesteps += n_envs

            if self.buffer.num_episodes >= 1 and num_timesteps >= self.learning_starts:
                effective_prev = max(prev_num_timesteps, self.learning_starts)
                boundaries_crossed = num_timesteps // self.train_freq - effective_prev // self.train_freq
                for _ in range(self.train_steps_per_iter * boundaries_crossed):
                    self._last_metrics = self._train_step()

                reanalyze_boundaries = num_timesteps // self.reanalyze_freq - effective_prev // self.reanalyze_freq
                if reanalyze_boundaries > 0:
                    reanalyze_metrics = self._reanalyze()
                    if reanalyze_metrics:
                        self._last_metrics = {**self._last_metrics, **reanalyze_metrics}

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

    def predict(self, obs: Any, deterministic: bool = True, episode_start: bool = False) -> tuple[Any, Any]:
        if episode_start:
            self._eval_cache = None
        obs_arr = obs_to_array(obs, self._obs_space)
        # Same persistent-cache strategy `learn()` uses for its own real
        # steps (module docstring's "Persistent, incrementally-extended
        # per-lane root cache" section) - just one lane.
        results = self.search(obs_arr[None], [self._eval_cache], deterministic=np.array([deterministic]))
        env_action = action_to_env(results[0]["env_action"], self._action_space)
        action_flat = obs_to_array(env_action, self._action_space)
        self._eval_cache = self._advance_lane_caches([self._eval_cache], obs_arr[None], action_flat[None])[0]
        return env_action, None

    def save(self, path: Path) -> None:
        torch.save(
            {
                "tokenizer_state_dict": self.tokenizer.state_dict(),
                "action_embed_state_dict": self.action_embed.state_dict(),
                "transformer_state_dict": self.transformer.state_dict(),
                "heads_state_dict": self.heads.state_dict(),
                "projector_state_dict": self.projector.state_dict(),
                "predictor_state_dict": self.predictor.state_dict(),
                "hyperparams": self.hyperparams,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, env: gym.Env, device: str = "cpu") -> "NativeUniZero":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        algo = cls(env, payload.get("hyperparams", {}), None, device)
        algo.tokenizer.load_state_dict(payload["tokenizer_state_dict"])
        algo.action_embed.load_state_dict(payload["action_embed_state_dict"])
        algo.transformer.load_state_dict(payload["transformer_state_dict"])
        algo.heads.load_state_dict(payload["heads_state_dict"])
        algo.projector.load_state_dict(payload["projector_state_dict"])
        algo.predictor.load_state_dict(payload["predictor_state_dict"])
        return algo
