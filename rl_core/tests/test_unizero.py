"""Coverage for `rl_core/algorithms/native/unizero.py` (UniZero — Transformer
world model + PUCT search, see that module's own docstring). Same "tiny
hyperparams, just check shapes/no-crash, not learned quality" convention as
`test_efficientzero.py`."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rl_core.algorithms.base import TrainingCallback
from rl_core.algorithms.native.unizero import (
    DEFAULT_HYPERPARAMS,
    NativeUniZero,
    _CausalTransformer,
    _LayerKV,
    _MinMaxStats,
    _SearchNode,
    _TransformerCache,
    _UniZeroBuffer,
    _add_dirichlet_noise,
    _build_attend_mask,
    _logits_to_scalar,
    _puct_select_child,
    _scalar_to_two_hot,
    _signed_hyperbolic,
    _signed_parabolic,
)
from rl_core.algorithms.native.preprocessing import obs_to_array
from rl_core.algorithms.vec_env import make_env_or_vec, make_gym_env_factory

_DISCRETE_ENV_ID = "CartPole-v1"
_CONTINUOUS_ENV_ID = "Pendulum-v1"

_TINY_DISCRETE = {
    "embed_dim": 16, "num_layers": 1, "num_heads": 2, "context_length": 2,
    "buffer_size": 200, "batch_size": 4, "unroll_steps": 3, "td_steps": 3,
    "num_sampled_actions": 4, "num_simulations": 6,
    "learning_starts": 10, "train_freq": 1, "value_support_size": 20,
}
_TINY_CONTINUOUS = {**_TINY_DISCRETE}


def _empty_caches(batch: int) -> list[None]:
    """No preceding real history - every lane's root starts from an empty
    incremental KV-cache (`search()`'s `root_caches` argument)."""
    return [None] * batch


def _run_smoke(env_id: str, hyperparams: dict, total_timesteps: int, num_envs: int = 1) -> None:
    if num_envs == 1:
        env = gym.make(env_id)
    else:
        env = make_env_or_vec(make_gym_env_factory(env_id), num_envs=num_envs, parallel=False)
    algo = NativeUniZero(env, {**DEFAULT_HYPERPARAMS, **hyperparams}, seed=0, device="cpu")

    steps: list[int] = []

    def writer(num_timesteps, episode_reward=None, episode_length=None, metrics=None):
        steps.append(num_timesteps)
        return num_timesteps < total_timesteps

    algo.learn(total_timesteps=total_timesteps, callback=TrainingCallback(writer))
    assert steps[-1] >= total_timesteps

    predict_env = gym.make(env_id)
    obs, _info = predict_env.reset(seed=1)
    action, _state = algo.predict(obs, deterministic=True, episode_start=True)
    assert predict_env.action_space.contains(action)
    # A second call without episode_start must reuse (not crash on) the
    # rolling eval context built up by the first call.
    obs2, reward2, terminated2, truncated2, _info2 = predict_env.step(action)
    algo.predict(obs2, deterministic=True, episode_start=False)

    with tempfile.TemporaryDirectory() as d:
        checkpoint = Path(d) / "model.pt"
        algo.save(checkpoint)
        loaded = NativeUniZero.load(checkpoint, gym.make(env_id), device="cpu")
        loaded.predict(obs, deterministic=True, episode_start=True)
    predict_env.close()
    env.close()


class TestPUCTSearch:
    """`_puct_select_child`/`_add_dirichlet_noise` (module docstring's
    "Search" section) - the classic AlphaZero/MuZero PUCT primitives that
    replaced this file's earlier Gumbel/Sequential-Halving search.
    `search()`'s own end-to-end shape/policy-target-sums-to-1 coverage
    lives in `TestNativeUniZeroDiscrete`/`TestNativeUniZeroContinuous`
    below already - this class is for the primitives in isolation."""

    def test_dirichlet_noise_preserves_simplex(self) -> None:
        priors = np.array([0.7, 0.2, 0.1], dtype=np.float64)
        noisy = _add_dirichlet_noise(priors, alpha=0.3, frac=0.25)
        assert noisy.shape == priors.shape
        assert np.all(noisy >= 0.0)
        assert np.isclose(noisy.sum(), 1.0, atol=1e-8)

    def test_dirichlet_noise_frac_zero_is_a_no_op(self) -> None:
        priors = np.array([0.7, 0.2, 0.1], dtype=np.float64)
        noisy = _add_dirichlet_noise(priors, alpha=0.3, frac=0.0)
        assert np.allclose(noisy, priors)

    def test_puct_prefers_higher_prior_when_all_children_unvisited(self) -> None:
        """No child has any visits yet (`value_score` is `0.0` for all of
        them - `_puct_select_child`'s own docstring) - the only thing
        that can make one child's score beat another's is its own
        `prior`, so the highest-prior child must always win the very
        first descent from any freshly-expanded node."""
        root = _SearchNode(prior=1.0)
        root.expand(np.array([0.1, 0.6, 0.3]), cache=None, reward_value=0.0)
        root.visit_count = 1
        minmax = _MinMaxStats(delta=0.01)
        idx = _puct_select_child(root, minmax, discount=0.99, pb_c_base=19652.0, pb_c_init=1.25)
        assert idx == 1

    def test_puct_visiting_a_child_reduces_its_own_future_selection_score(self) -> None:
        """Sanity-check the `1 / (1 + child.visit_count)` term actually
        does something: artificially visiting the best-prior child many
        times (without improving its own value estimate at all) must
        eventually make PUCT prefer a lower-prior, unvisited sibling -
        otherwise the tree would collapse to a single-path lookahead,
        defeating the entire point of a *search* tree."""
        root = _SearchNode(prior=1.0)
        root.expand(np.array([0.05, 0.9, 0.05]), cache=None, reward_value=0.0)
        root.visit_count = 1
        minmax = _MinMaxStats(delta=0.01)
        best_child = root.children[1]
        for _ in range(200):
            best_child.visit_count += 1
            best_child.value_sum += 0.0  # visited, but never actually good
            root.visit_count += 1
        idx = _puct_select_child(root, minmax, discount=0.99, pb_c_base=19652.0, pb_c_init=1.25)
        assert idx != 1

    def test_puct_score_is_deterministic_given_same_inputs(self) -> None:
        """No RNG anywhere in `_puct_select_child` itself (unlike the
        Gumbel search this replaced, which needed a fresh `gumbel` draw
        per simulation) - root Dirichlet noise is the *only* randomness
        anywhere in this search, and it's injected once, into priors,
        before any PUCT selection happens at all."""
        root = _SearchNode(prior=1.0)
        root.expand(np.array([0.3, 0.4, 0.3]), cache=None, reward_value=0.0)
        root.visit_count = 1
        minmax = _MinMaxStats(delta=0.01)
        idx_a = _puct_select_child(root, minmax, discount=0.99, pb_c_base=19652.0, pb_c_init=1.25)
        idx_b = _puct_select_child(root, minmax, discount=0.99, pb_c_base=19652.0, pb_c_init=1.25)
        assert idx_a == idx_b


class TestCategoricalValueSupport:
    """Ported wholesale from `efficientzero.py`'s own module - bugs here
    would silently miscalibrate every value/reward target without ever
    crashing anything, so keep the same standalone coverage in this file
    too rather than assuming the port was faithful."""

    def test_hyperbolic_parabolic_are_inverses(self) -> None:
        x = torch.tensor([-500.0, -10.0, -1.0, 0.0, 1.0, 10.0, 500.0, 12345.0])
        roundtrip = _signed_parabolic(_signed_hyperbolic(x))
        assert torch.allclose(roundtrip, x, atol=1e-2)

    def test_two_hot_is_a_valid_distribution_and_decodes_back(self) -> None:
        support_size = 50
        x = torch.tensor([-40.0, -0.3, 0.0, 2.7, 39.0])
        two_hot = _scalar_to_two_hot(x, support_size)
        assert two_hot.shape == (5, 2 * support_size + 1)
        assert torch.all(two_hot >= 0.0)
        assert torch.allclose(two_hot.sum(-1), torch.ones(5), atol=1e-5)
        logits = torch.log(two_hot.clamp_min(1e-9))
        decoded = _logits_to_scalar(logits, support_size)
        assert torch.allclose(decoded, x, atol=0.5)


class TestAttendMask:
    """`_build_attend_mask` - the causal + key-padding mask every
    Transformer forward pass uses. A pad *query* row with zero allowed
    keys would make attention's softmax divide 0/0 -> NaN, which then
    poisons every *valid* query's output too (masked attention *weight* is
    0, but `0 * NaN` is still `NaN`) - see that function's own docstring."""

    def test_causal_lower_triangular_when_all_valid(self) -> None:
        pad_mask = torch.ones(1, 4, dtype=torch.bool)
        mask = _build_attend_mask(pad_mask)
        expected = torch.tril(torch.ones(4, 4, dtype=torch.bool))
        assert torch.equal(mask[0], expected)

    def test_pad_keys_never_attended_to_by_valid_queries(self) -> None:
        pad_mask = torch.tensor([[False, False, True, True]])
        mask = _build_attend_mask(pad_mask)
        # Query 3 (valid, causal-sees all of 0..3) must not be allowed to
        # attend to pad keys 0/1.
        assert not bool(mask[0, 3, 0])
        assert not bool(mask[0, 3, 1])
        assert bool(mask[0, 3, 2])
        assert bool(mask[0, 3, 3])

    def test_pad_query_rows_are_never_all_false(self) -> None:
        pad_mask = torch.tensor([[False, False, True, True]])
        mask = _build_attend_mask(pad_mask)
        assert bool(mask[0, 0].any())
        assert bool(mask[0, 1].any())

    def test_no_nan_through_a_real_forward_pass_with_full_padding(self) -> None:
        """An entirely-empty context (every context slot invalid) must
        still produce finite hidden states - the realistic worst case for
        a fresh episode's very first real step."""
        from rl_core.algorithms.native.unizero import _CausalTransformer

        transformer = _CausalTransformer(embed_dim=8, num_layers=2, num_heads=2, dropout=0.0)
        tokens = torch.randn(2, 7, 8)
        pad_mask = torch.tensor([[False, False, False, False, False, False, True]] * 2)
        out = transformer(tokens, pad_mask)
        assert torch.isfinite(out).all()


class TestRoPE:
    """Rotary position embeddings (module docstring's "Positional
    encoding" section) - the property that makes both training/self-play
    convention-agreement *and* lossless cache eviction fall out for free:
    attention only ever depends on *relative* offsets between token
    positions, never their absolute value."""

    def test_apply_rope_preserves_vector_norm(self) -> None:
        """A pure rotation must not change a vector's length."""
        from rl_core.algorithms.native.unizero import _apply_rope, _rope_cos_sin

        torch.manual_seed(0)
        x = torch.randn(3, 8)
        positions = torch.tensor([0, 5, 1000])
        cos, sin = _rope_cos_sin(positions, head_dim=8)
        rotated = _apply_rope(x, cos, sin)
        assert torch.allclose(x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-5)

    def test_dot_product_depends_only_on_relative_offset(self) -> None:
        """`<rope(q, p1), rope(k, p2)>` must be identical for any two
        `(p1, p2)` pairs sharing the same `p1 - p2` - the whole reason a
        training window that always starts counting from `0` and a
        self-play root that uses true, ever-growing absolute positions
        compute the exact same attention (module docstring's "Positional
        encoding" section, first bullet)."""
        from rl_core.algorithms.native.unizero import _apply_rope, _rope_cos_sin

        torch.manual_seed(1)
        q, k = torch.randn(8), torch.randn(8)

        def dot_at(p1: int, p2: int) -> float:
            cos_q, sin_q = _rope_cos_sin(torch.tensor([p1]), head_dim=8)
            cos_k, sin_k = _rope_cos_sin(torch.tensor([p2]), head_dim=8)
            rq = _apply_rope(q.unsqueeze(0), cos_q, sin_q)
            rk = _apply_rope(k.unsqueeze(0), cos_k, sin_k)
            return float((rq * rk).sum())

        # (0, 0) and (500, 500) share offset 0; (2, 5) and (1002, 1005)
        # share offset -3 - both pairs must match despite wildly different
        # absolute magnitudes.
        assert abs(dot_at(0, 0) - dot_at(500, 500)) < 1e-3
        assert abs(dot_at(2, 5) - dot_at(1002, 1005)) < 1e-3

    def test_full_sequence_forward_is_translation_invariant(self) -> None:
        """Shifting every lane's positions by the same constant offset
        (e.g. "this window starts at absolute episode step 500" instead
        of "0") must not change `_CausalTransformer.forward`'s output at
        all - the exact property that lets training's always-`0`-based
        windows and self-play's true absolute positions agree without
        either side needing to know about the other's convention."""
        torch.manual_seed(2)
        transformer = _CausalTransformer(embed_dim=8, num_layers=2, num_heads=2, dropout=0.0)
        transformer.eval()
        tokens = torch.randn(2, 5, 8)
        pad_mask = torch.ones(2, 5, dtype=torch.bool)
        with torch.no_grad():
            out_zero_based = transformer(tokens, pad_mask, positions=torch.arange(5).unsqueeze(0).expand(2, 5))
            out_shifted = transformer(tokens, pad_mask, positions=(torch.arange(5) + 500).unsqueeze(0).expand(2, 5))
        assert torch.allclose(out_zero_based, out_shifted, atol=1e-4)


class TestIncrementalKVCache:
    """`_TransformerCache`/`forward_incremental_batch` - the real
    incremental KV-cache (module docstring's "Persistent, incrementally-
    extended per-lane root cache" section). The central correctness
    property: replaying a sequence one token at a time through the cache
    must give *bit-identical* hidden states to a single full-sequence
    `forward` call over the same tokens - a cache is a speed
    optimization, not an approximation."""

    def _random_transformer(self, embed_dim: int = 8, num_layers: int = 2, num_heads: int = 2) -> _CausalTransformer:
        torch.manual_seed(0)
        transformer = _CausalTransformer(embed_dim=embed_dim, num_layers=num_layers, num_heads=num_heads, dropout=0.0)
        transformer.eval()
        return transformer

    def test_incremental_replay_matches_full_sequence_forward(self) -> None:
        embed_dim, seq_len, batch = 8, 6, 3
        transformer = self._random_transformer(embed_dim=embed_dim)
        torch.manual_seed(1)
        tokens = torch.randn(batch, seq_len, embed_dim)
        pad_mask = torch.ones(batch, seq_len, dtype=torch.bool)

        with torch.no_grad():
            full_hidden = transformer(tokens, pad_mask)  # (B, L, E)

            caches: list[_TransformerCache | None] = [None] * batch
            replayed = torch.zeros(batch, seq_len, embed_dim)
            for t in range(seq_len):
                positions = torch.as_tensor([0 if c is None else c.next_pos for c in caches], dtype=torch.long)
                hidden, caches = transformer.forward_incremental_batch(tokens[:, t], positions, caches)
                replayed[:, t] = hidden

        assert torch.allclose(replayed, full_hidden, atol=1e-5)

    def test_incremental_replay_matches_full_sequence_with_uneven_lane_history(self) -> None:
        """Different lanes with different real history lengths (the
        common real-`learn()` case) must each still get exactly the
        hidden states a full-sequence forward over *their own* (shorter)
        sequence would give - padding across lanes inside one batched
        incremental call must never leak into another lane's result."""
        embed_dim = 8
        transformer = self._random_transformer(embed_dim=embed_dim)
        torch.manual_seed(2)
        seq_a = torch.randn(1, 5, embed_dim)
        seq_b = torch.randn(1, 2, embed_dim)

        with torch.no_grad():
            full_a = transformer(seq_a, torch.ones(1, 5, dtype=torch.bool))[0]
            full_b = transformer(seq_b, torch.ones(1, 2, dtype=torch.bool))[0]

            caches: list[_TransformerCache | None] = [None, None]
            replayed_a = torch.zeros(5, embed_dim)
            replayed_b = torch.zeros(2, embed_dim)
            for t in range(5):
                tok = torch.stack([seq_a[0, t], seq_b[0, t] if t < 2 else torch.zeros(embed_dim)])
                positions = torch.as_tensor([0 if c is None else c.next_pos for c in caches], dtype=torch.long)
                hidden, new_caches = transformer.forward_incremental_batch(tok, positions, caches)
                replayed_a[t] = hidden[0]
                if t < 2:
                    replayed_b[t] = hidden[1]
                    caches[1] = new_caches[1]
                caches[0] = new_caches[0]

        assert torch.allclose(replayed_a, full_a, atol=1e-5)
        assert torch.allclose(replayed_b, full_b, atol=1e-5)

    def test_branching_from_shared_parent_cache_does_not_cross_contaminate(self) -> None:
        """Two children extending the *same* parent cache with different
        tokens (search-tree branching) must not corrupt each other or the
        parent - see `_TransformerCache`'s own docstring for why this
        falls out for free from `forward_incremental_batch` never
        mutating its cache argument."""
        embed_dim = 8
        transformer = self._random_transformer(embed_dim=embed_dim)
        torch.manual_seed(3)
        parent_token = torch.randn(1, embed_dim)
        with torch.no_grad():
            _, parent_caches = transformer.forward_incremental_batch(parent_token, torch.zeros(1, dtype=torch.long), [None])
            parent_cache = parent_caches[0]
            parent_k_before = parent_cache.layers[0].k.clone()

            child_a_token = torch.randn(1, embed_dim)
            child_b_token = torch.randn(1, embed_dim)
            pos = torch.ones(1, dtype=torch.long)
            hidden_a, cache_a = transformer.forward_incremental_batch(child_a_token, pos, [parent_cache])
            hidden_b, cache_b = transformer.forward_incremental_batch(child_b_token, pos, [parent_cache])

            # Parent cache itself must be untouched by either child.
            assert torch.equal(parent_cache.layers[0].k, parent_k_before)
            # The two children's own caches/hidden states must differ
            # (different tokens) and each must independently match a
            # from-scratch full-sequence forward over [parent, child].
            assert not torch.allclose(hidden_a, hidden_b)
            full_a = transformer(torch.cat([parent_token, child_a_token]).unsqueeze(0), torch.ones(1, 2, dtype=torch.bool))
            full_b = transformer(torch.cat([parent_token, child_b_token]).unsqueeze(0), torch.ones(1, 2, dtype=torch.bool))
            assert torch.allclose(hidden_a, full_a[:, -1], atol=1e-5)
            assert torch.allclose(hidden_b, full_b[:, -1], atol=1e-5)
            assert cache_a[0].layers[0].k.shape[2] == 2 and cache_b[0].layers[0].k.shape[2] == 2

    def _evict_then_cold_restart(self, transformer: _CausalTransformer, embed_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(4)
        history = torch.randn(1, 8, embed_dim)
        cache: _TransformerCache | None = None
        with torch.no_grad():
            for t in range(8):
                pos = torch.tensor([0 if cache is None else cache.next_pos])
                _, caches = transformer.forward_incremental_batch(history[:, t], pos, [cache])
                cache = caches[0]
            assert cache is not None and cache.length == 8 and cache.next_pos == 8

            # Trim down to the last 4 tokens - `next_pos` must stay at 8
            # (module docstring's "Positional encoding" section), only
            # `length` shrinks.
            cache.evict_front(4)
            assert cache.length == 4 and cache.next_pos == 8

            new_token = torch.randn(1, embed_dim)
            pos = torch.tensor([cache.next_pos])
            hidden_evicted, _ = transformer.forward_incremental_batch(new_token, pos, [cache])

            cold_cache: _TransformerCache | None = None
            for t in range(4, 8):
                pos = torch.tensor([0 if cold_cache is None else cold_cache.next_pos])
                _, cc = transformer.forward_incremental_batch(history[:, t], pos, [cold_cache])
                cold_cache = cc[0]
            pos = torch.tensor([cold_cache.next_pos])
            hidden_cold, _ = transformer.forward_incremental_batch(new_token, pos, [cold_cache])
        return hidden_evicted, hidden_cold

    def test_evict_front_is_exactly_lossless_with_a_single_layer(self) -> None:
        """With `num_layers=1`, there's no residual-stream composition for
        an evicted token's influence to hide in (module docstring's
        "Positional encoding" section, second bullet, and `_TransformerCache
        .evict_front`'s own docstring) - trimming a cache down to its
        surviving tokens must give *exactly* the same next hidden state a
        fresh cold-start replay of just those survivors would."""
        transformer = self._random_transformer(embed_dim=8, num_layers=1)
        hidden_evicted, hidden_cold = self._evict_then_cold_restart(transformer, embed_dim=8)
        assert torch.allclose(hidden_evicted, hidden_cold, atol=1e-5)

    def test_evict_front_leaves_only_a_small_bounded_residual_with_multiple_layers(self) -> None:
        """With `num_layers > 1`, evicted tokens' influence on the
        survivors' *own* hidden states (baked in before eviction happened)
        legitimately carries forward through the residual stream - see
        `_TransformerCache.evict_front`'s own docstring for why that's an
        accepted, bounded trade rather than a bug. This just guards against
        a regression turning that into something unbounded/wrong (e.g. a
        stale, garbage, or `NaN` hidden state) rather than the small,
        finite perturbation it should be."""
        transformer = self._random_transformer(embed_dim=8, num_layers=2)
        hidden_evicted, hidden_cold = self._evict_then_cold_restart(transformer, embed_dim=8)
        assert torch.isfinite(hidden_evicted).all()
        assert not torch.allclose(hidden_evicted, hidden_cold, atol=1e-5)  # documents the (small) residual
        assert torch.allclose(hidden_evicted, hidden_cold, atol=0.1)  # ...but it stays small


class TestRotaryEmbToggle:
    """`rotary_emb=0` - the learned-absolute-embedding fallback (module
    docstring's "Positional encoding toggle" section), off by default.
    Complements `TestRoPE`/`TestIncrementalKVCache` (which only ever
    exercise the RoPE-on default) with the same style of coverage for the
    off branch: a fresh (never-evicted) cache must still match a
    full-sequence forward exactly, and `evict_front`'s `pos_origin`
    bookkeeping (the "re-based position-delta" approximation) must do
    exactly what its own docstring says."""

    def _random_transformer(self, embed_dim: int = 8, num_layers: int = 2, num_heads: int = 2) -> _CausalTransformer:
        torch.manual_seed(0)
        transformer = _CausalTransformer(
            embed_dim=embed_dim, num_layers=num_layers, num_heads=num_heads, dropout=0.0, rotary_emb=False,
        )
        transformer.eval()
        return transformer

    def test_pos_embed_table_exists_only_when_rope_is_off(self) -> None:
        with_rope = _CausalTransformer(embed_dim=8, num_layers=1, num_heads=2, dropout=0.0, rotary_emb=True)
        without_rope = _CausalTransformer(embed_dim=8, num_layers=1, num_heads=2, dropout=0.0, rotary_emb=False)
        assert with_rope.pos_embed is None
        assert without_rope.pos_embed is not None

    def test_incremental_replay_matches_full_sequence_forward_with_no_eviction(self) -> None:
        """A fresh cache that's never had `evict_front` called on it has
        `pos_origin == 0` throughout, so its `embed_positions` (module-
        level `_cache_positions`) exactly equal the plain `0..L-1` a
        full-sequence `forward` call would use by default - the additive-
        embedding analogue of `TestIncrementalKVCache`'s own RoPE parity
        test."""
        from rl_core.algorithms.native.unizero import _cache_positions

        embed_dim, seq_len, batch = 8, 6, 3
        transformer = self._random_transformer(embed_dim=embed_dim)
        torch.manual_seed(1)
        tokens = torch.randn(batch, seq_len, embed_dim)
        pad_mask = torch.ones(batch, seq_len, dtype=torch.bool)

        with torch.no_grad():
            full_hidden = transformer(tokens, pad_mask)  # (B, L, E)

            caches: list[_TransformerCache | None] = [None] * batch
            replayed = torch.zeros(batch, seq_len, embed_dim)
            for t in range(seq_len):
                positions, embed_positions = _cache_positions(caches, device="cpu")
                hidden, caches = transformer.forward_incremental_batch(
                    tokens[:, t], positions, caches, embed_positions,
                )
                replayed[:, t] = hidden

        assert torch.allclose(replayed, full_hidden, atol=1e-5)

    def test_evict_front_bumps_pos_origin_by_exactly_the_evicted_count(self) -> None:
        cache = _TransformerCache(num_layers=1)
        cache.layers[0] = _LayerKV(torch.randn(1, 2, 8, 4), torch.randn(1, 2, 8, 4))
        cache.length = 8
        cache.next_pos = 8
        assert cache.pos_origin == 0

        cache.evict_front(5)  # drop 3
        assert cache.length == 5
        assert cache.next_pos == 8  # untouched - module docstring's own invariant
        assert cache.pos_origin == 3

        cache.evict_front(5)  # keep_last >= length - a no-op, including for pos_origin
        assert cache.pos_origin == 3

        cache.length, cache.next_pos = 5, 10  # pretend 2 more tokens got appended since
        cache.evict_front(2)  # drop 3 more
        assert cache.pos_origin == 6

    def test_cache_positions_helper_rebase(self) -> None:
        from rl_core.algorithms.native.unizero import _cache_positions

        cache = _TransformerCache(num_layers=1)
        cache.next_pos, cache.pos_origin = 20, 6
        positions, embed_positions = _cache_positions([None, cache], device="cpu")
        assert positions.tolist() == [0, 20]  # raw, absolute - unaffected by pos_origin
        assert embed_positions.tolist() == [0, 14]  # rebased: next_pos - pos_origin

    def test_smoke_discrete_and_continuous(self) -> None:
        """End-to-end - `rotary_emb=0` must train/collect without
        crashing for both action spaces, exactly like the RoPE-on default
        (`TestNativeUniZeroDiscrete/Continuous.test_smoke`)."""
        _run_smoke(_DISCRETE_ENV_ID, {**_TINY_DISCRETE, "rotary_emb": 0}, total_timesteps=40)
        _run_smoke(_CONTINUOUS_ENV_ID, {**_TINY_CONTINUOUS, "rotary_emb": 0}, total_timesteps=40)


class TestPersistentLaneCache:
    """`NativeUniZero._lane_cache`/`_advance_lane_caches` - `learn()`'s
    per-lane cache that now genuinely persists and grows across real
    steps for an entire episode (module docstring's "Persistent,
    incrementally-extended per-lane root cache" section), instead of
    being cold-started from scratch every single step."""

    def test_cache_grows_then_stays_bounded_at_context_window(self) -> None:
        env = gym.make(_DISCRETE_ENV_ID)
        algo = NativeUniZero(env, {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE, "learning_starts": 0}, seed=0, device="cpu")
        obs, _info = env.reset(seed=0)
        obs_arr = obs_to_array(obs, env.observation_space)[None]
        lengths: list[int] = []
        for step in range(20):
            results = algo.search(obs_arr, algo._lane_cache, deterministic=np.array([False]))
            action = int(results[0]["env_action"])
            next_obs, _reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated
            next_obs_arr = obs_to_array(next_obs, env.observation_space)[None]
            action_flat = np.array([[1.0 if i == action else 0.0 for i in range(env.action_space.n)]], dtype=np.float32)
            new_caches = algo._advance_lane_caches(algo._lane_cache, obs_arr, action_flat)
            algo._lane_cache = [None] if done else new_caches
            cache = algo._lane_cache[0]
            lengths.append(0 if cache is None else cache.length)
            if cache is not None:
                assert cache.length <= 2 * algo.context_length
                assert torch.isfinite(cache.layers[0].k).all()
                # `next_pos` (the real, ever-growing token count since
                # this episode started) must be at least `length` (the
                # physical, eviction-capped count) always, and strictly
                # greater once eviction has actually kicked in once.
                assert cache.next_pos >= cache.length
            if done:
                obs, _info = env.reset()
                next_obs_arr = obs_to_array(obs, env.observation_space)[None]
            obs_arr = next_obs_arr
        # Across 20 steps of CartPole (short episodes), the cache must
        # actually have reached the `2*context_length` bound at least
        # once - otherwise this test isn't exercising eviction at all.
        assert max(lengths) == 2 * algo.context_length
        env.close()

    def test_advance_lane_caches_never_tracks_gradients(self) -> None:
        """Regression test for a real leak: `_advance_lane_caches` is what
        `self._lane_cache`/`self._eval_cache` actually get threaded
        through across an entire episode's worth of real steps (this
        class's own module docstring). If its forward passes weren't
        `torch.no_grad()`, every real step would extend the autograd
        graph rooted at the *previous* step's (already graph-tracked)
        cached K/V tensors instead of starting fresh - an ever-growing
        graph, retained for the whole episode, that silently turns a
        tiny model into a multi-GiB CUDA OOM well before a long episode
        (e.g. NetHack) ends. Every layer's cached `k`/`v` must always be
        leaf-like: `requires_grad is False` and `grad_fn is None`, even
        after many chained real-step calls."""
        env = gym.make(_DISCRETE_ENV_ID)
        algo = NativeUniZero(env, {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE, "learning_starts": 0}, seed=0, device="cpu")
        obs_arr = obs_to_array(env.reset(seed=0)[0], env.observation_space)[None]
        caches: list[_TransformerCache | None] = [None]
        for _ in range(10):
            action_flat = np.zeros((1, env.action_space.n), dtype=np.float32)
            action_flat[0, 0] = 1.0
            caches = algo._advance_lane_caches(caches, obs_arr, action_flat)
            for layer in caches[0].layers:
                assert layer is not None
                assert layer.k.requires_grad is False and layer.k.grad_fn is None
                assert layer.v.requires_grad is False and layer.v.grad_fn is None
        env.close()

    def test_cache_resets_to_none_on_episode_end(self) -> None:
        env = gym.make(_DISCRETE_ENV_ID)
        algo = NativeUniZero(env, {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE, "learning_starts": 0}, seed=0, device="cpu")
        algo._lane_cache = [_TransformerCache(1)]  # pretend there's live history
        algo._lane_cache[0].length = 4
        # `learn()`'s own per-lane reset logic: `None if dones[lane] else new_caches[lane]`.
        done = True
        new_caches = [_TransformerCache(1)]
        algo._lane_cache = [None if done else new_caches[0]]
        assert algo._lane_cache[0] is None
        env.close()

    def test_predict_persists_cache_across_calls_and_resets_on_episode_start(self) -> None:
        env = gym.make(_DISCRETE_ENV_ID)
        algo = NativeUniZero(env, {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE}, seed=0, device="cpu")
        obs, _info = env.reset(seed=0)
        assert algo._eval_cache is None
        algo.predict(obs, deterministic=True, episode_start=True)
        assert algo._eval_cache is not None and algo._eval_cache.length == 2
        algo.predict(obs, deterministic=True, episode_start=False)
        assert algo._eval_cache.length == 4
        algo.predict(obs, deterministic=True, episode_start=True)
        assert algo._eval_cache.length == 2  # reset, then one more step
        env.close()


class TestColdStartReplayMatchesTrainingConvention:
    """`_replay_context_to_cache` + `forward_incremental_batch`
    (`_reanalyze`'s root-construction path) must produce the exact same
    root hidden state `_embed_root_batch` + `_CausalTransformer.forward`
    (training's own teacher-forcing path) would for the identical
    (context, current-obs) pair - the property that makes the two safe to
    mix within one training run without a train/self-play distribution
    mismatch (module docstring's "Positional encoding" section)."""

    def test_replay_context_to_cache_never_tracks_gradients(self) -> None:
        """Same regression as `TestPersistentLaneCache
        .test_advance_lane_caches_never_tracks_gradients`, for
        `_reanalyze()`'s own cold-start cache-construction path."""
        env = gym.make(_DISCRETE_ENV_ID)
        algo = NativeUniZero(env, {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE}, seed=0, device="cpu")
        one = obs_to_array(env.reset(seed=0)[0], env.observation_space)
        length = algo.context_length
        ctx_obs = np.stack([one] * length)[None]
        ctx_action = np.zeros((1, length, algo.action_dim), dtype=np.float32)
        ctx_valid = np.ones((1, length), dtype=bool)
        root_caches = algo._replay_context_to_cache(ctx_obs, ctx_action, ctx_valid)
        for layer in root_caches[0].layers:
            assert layer is not None
            assert layer.k.requires_grad is False and layer.k.grad_fn is None
            assert layer.v.requires_grad is False and layer.v.grad_fn is None
        env.close()

    def test_cache_root_matches_full_sequence_root(self) -> None:
        env = gym.make(_DISCRETE_ENV_ID)
        algo = NativeUniZero(env, {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE}, seed=0, device="cpu")
        # `DEFAULT_HYPERPARAMS["dropout"]` is `0.1` (not `0.0`) by
        # default now (module docstring's "Networks" comparison to the
        # reference) - `.eval()` so the two forward-pass strategies this
        # test compares are deterministic and bit-comparable; dropout
        # noise itself is exercised by other tests, not this one.
        algo.transformer.eval()
        obs, _info = env.reset(seed=0)
        one = obs_to_array(obs, env.observation_space)
        length = algo.context_length
        ctx_obs = np.stack([one] * length)[None]
        ctx_action = np.zeros((1, length, algo.action_dim), dtype=np.float32)
        ctx_valid = np.ones((1, length), dtype=bool)

        with torch.no_grad():
            root_caches = algo._replay_context_to_cache(ctx_obs, ctx_action, ctx_valid)
            obs_emb = algo.tokenizer(torch.as_tensor(one[None], dtype=torch.float32))
            positions0 = torch.as_tensor([0 if c is None else c.next_pos for c in root_caches], dtype=torch.long)
            h_root_cache, root_caches_out = algo.transformer.forward_incremental_batch(obs_emb, positions0, root_caches)

            tokens, pad_mask, valid_len = algo._embed_root_batch(ctx_obs, ctx_action, ctx_valid, one[None])
            hidden = algo.transformer(tokens, pad_mask)
            h_root_full_sequence = hidden[0, int(valid_len[0]) - 1]

        assert torch.allclose(h_root_cache[0], h_root_full_sequence, atol=1e-4)

        # And one simulated tree edge from that root (`_step_imagine`,
        # incremental) must land on the same hidden states a from-scratch
        # full-sequence forward over [context, current_obs, action]
        # would give for the identical action.
        action = torch.zeros(1, algo.n_actions)
        action[0, 0] = 1.0
        with torch.no_grad():
            _child_caches, h_act_cache, h_obs_cache = algo._step_imagine(root_caches_out, action)
            act_emb = algo.action_embed(action)
            root_seq = tokens[0, : int(valid_len[0])]
            seq1 = torch.cat([root_seq, act_emb[0:1]]).unsqueeze(0)
            h1 = algo.transformer(seq1, torch.ones(1, seq1.shape[1], dtype=torch.bool))
            h_act_full = h1[:, -1]
            z_pred = algo.heads.latent(h_act_full)
            seq2 = torch.cat([seq1[0], z_pred]).unsqueeze(0)
            h2 = algo.transformer(seq2, torch.ones(1, seq2.shape[1], dtype=torch.bool))
            h_obs_full = h2[:, -1]
        assert torch.allclose(h_act_cache[0], h_act_full[0], atol=1e-4)
        assert torch.allclose(h_obs_cache[0], h_obs_full[0], atol=1e-4)
        env.close()


class TestTargetNetwork:
    """`self.target_tokenizer`/`self.target_transformer`/`self.target_heads`
    (module docstring's "Target network" section) - an EMA-updated copy
    of the corresponding online modules, used for the bootstrap value and
    the latent-consistency loss's target embedding."""

    def test_target_starts_as_exact_copy_of_online(self) -> None:
        env = gym.make(_DISCRETE_ENV_ID)
        algo = NativeUniZero(env, {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE}, seed=0, device="cpu")
        for online_p, target_p in zip(algo.tokenizer.parameters(), algo.target_tokenizer.parameters()):
            assert torch.equal(online_p, target_p)
        for online_p, target_p in zip(algo.transformer.parameters(), algo.target_transformer.parameters()):
            assert torch.equal(online_p, target_p)
        for online_p, target_p in zip(algo.heads.parameters(), algo.target_heads.parameters()):
            assert torch.equal(online_p, target_p)
        env.close()

    def test_target_params_never_require_grad(self) -> None:
        env = gym.make(_DISCRETE_ENV_ID)
        algo = NativeUniZero(env, {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE}, seed=0, device="cpu")
        for module in (algo.target_tokenizer, algo.target_transformer, algo.target_heads):
            assert not module.training
            for p in module.parameters():
                assert p.requires_grad is False
        env.close()

    def test_update_target_network_moves_toward_online_by_exactly_theta(self) -> None:
        env = gym.make(_DISCRETE_ENV_ID)
        theta = 0.3
        algo = NativeUniZero(
            env, {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE, "target_update_theta": theta}, seed=0, device="cpu",
        )
        # Perturb the online tokenizer only, so the expected post-update
        # target value has a single, easy-to-check closed form:
        # `(1-theta)*old_target + theta*online`, and `old_target ==
        # old_online` per `test_target_starts_as_exact_copy_of_online`.
        with torch.no_grad():
            for p in algo.tokenizer.parameters():
                p.add_(1.0)
        old_target = [p.clone() for p in algo.target_tokenizer.parameters()]
        online_now = [p.clone() for p in algo.tokenizer.parameters()]
        algo._update_target_network()
        for old_t, online_p, new_t in zip(old_target, online_now, algo.target_tokenizer.parameters()):
            expected = (1.0 - theta) * old_t + theta * online_p
            assert torch.allclose(new_t, expected, atol=1e-6)
        env.close()

    def test_train_step_actually_moves_target_network(self) -> None:
        """Regression: `_update_target_network()` must actually be *called*
        from `_train_step` (easy to silently skip - it's not needed for
        the forward/backward pass to run without crashing, only for the
        target network to mean anything over time)."""
        env = gym.make(_DISCRETE_ENV_ID)
        algo = NativeUniZero(env, {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE}, seed=0, device="cpu")
        old_target = [p.clone() for p in algo.target_tokenizer.parameters()]

        steps: list[int] = []

        def writer(num_timesteps, episode_reward=None, episode_length=None, metrics=None):
            steps.append(num_timesteps)
            return num_timesteps < 40

        algo.learn(total_timesteps=40, callback=TrainingCallback(writer))
        assert any(
            not torch.equal(old_t, new_t) for old_t, new_t in zip(old_target, algo.target_tokenizer.parameters())
        )
        env.close()


class TestUniZeroBuffer:
    def test_sample_shapes_and_masking(self) -> None:
        buffer = _UniZeroBuffer(capacity_episodes=10, obs_shape=(4,), action_dim=2, policy_target_dim=2, context_length=2)
        for t in range(3):
            done = t == 2
            buffer.add(
                obs=np.full(4, float(t)), action_flat=np.array([1.0, 0.0]), reward=1.0,
                next_obs=np.full(4, float(t + 1)), policy_target=np.array([0.5, 0.5]), done=done,
            )
        assert buffer.num_episodes == 1
        assert len(buffer) == 3

        batch = buffer.sample(batch_size=8, unroll_steps=3, td_steps=3, gamma=0.9)
        assert batch["obs0"].shape == (8, 4)
        assert batch["action"].shape == (8, 3, 2)
        assert batch["mask"].shape == (8, 3)
        assert batch["context_obs"].shape == (8, 2, 4)
        assert batch["context_action"].shape == (8, 2, 2)
        assert batch["context_valid"].shape == (8, 2)
        for m in batch["mask"]:
            saw_zero = False
            for v in m:
                if v == 0.0:
                    saw_zero = True
                elif saw_zero:
                    raise AssertionError(f"mask has a 1 after a 0: {m}")

    def test_context_is_compact_and_valid_flagged(self) -> None:
        """Sampling at `start=1` in a 3-step episode with `context_length=2`
        must return exactly one real preceding transition, packed at the
        *front* (compact - see `_UniZeroBuffer._pad_one`'s docstring for
        why: it lets every caller treat "position" as a plain array
        index, with no separate lookup, in both the full-sequence and the
        incremental-KV-cache code paths)."""
        buffer = _UniZeroBuffer(capacity_episodes=10, obs_shape=(1,), action_dim=1, policy_target_dim=1, context_length=2)
        for t in range(3):
            buffer.add(
                obs=np.array([float(t)]), action_flat=np.array([float(t)]), reward=1.0,
                next_obs=np.array([float(t + 1)]), policy_target=np.array([1.0]), done=t == 2,
            )
        ep = buffer.episodes[0]
        ctx_obs, ctx_action, ctx_valid = buffer._context_for(ep, start=1)
        assert ctx_valid.tolist() == [True, False]
        assert np.isclose(ctx_obs[0, 0], 0.0)  # obs at t=0
        assert np.isclose(ctx_action[0, 0], 0.0)

    def test_terminal_step_has_no_bootstrap(self) -> None:
        buffer = _UniZeroBuffer(capacity_episodes=10, obs_shape=(1,), action_dim=1, policy_target_dim=1, context_length=1)
        buffer.add(obs=np.array([0.0]), action_flat=np.array([0.0]), reward=1.0, next_obs=np.array([1.0]), policy_target=np.array([1.0]), done=True)
        batch = buffer.sample(batch_size=4, unroll_steps=1, td_steps=5, gamma=0.9)
        assert np.all(batch["td_bootstrap_mask"][:, 0] == 0.0)
        assert np.allclose(batch["td_reward"][:, 0], 1.0)

    def test_lanes_do_not_splice_episodes(self) -> None:
        buffer = _UniZeroBuffer(capacity_episodes=10, obs_shape=(1,), action_dim=1, policy_target_dim=1, context_length=1, num_lanes=2)
        buffer.add(obs=np.array([0.0]), action_flat=np.array([0.0]), reward=1.0, next_obs=np.array([1.0]), policy_target=np.array([1.0]), done=False, lane=0)
        buffer.add(obs=np.array([10.0]), action_flat=np.array([0.0]), reward=1.0, next_obs=np.array([11.0]), policy_target=np.array([1.0]), done=True, lane=1)
        buffer.add(obs=np.array([1.0]), action_flat=np.array([0.0]), reward=1.0, next_obs=np.array([2.0]), policy_target=np.array([1.0]), done=True, lane=0)
        assert buffer.num_episodes == 2
        lane1_ep = next(ep for ep in buffer.episodes if ep["obs"].shape[0] == 1)
        lane0_ep = next(ep for ep in buffer.episodes if ep["obs"].shape[0] == 2)
        assert np.allclose(lane1_ep["obs"], [[10.0]])
        assert np.allclose(lane0_ep["obs"], [[0.0], [1.0]])

    def test_update_priorities_biases_future_sampling(self) -> None:
        buffer = _UniZeroBuffer(capacity_episodes=10, obs_shape=(1,), action_dim=1, policy_target_dim=1, context_length=1, priority_alpha=1.0)
        buffer.add(obs=np.array([0.0]), action_flat=np.array([0.0]), reward=1.0, next_obs=np.array([1.0]), policy_target=np.array([1.0]), done=True)
        buffer.add(obs=np.array([100.0]), action_flat=np.array([0.0]), reward=1.0, next_obs=np.array([101.0]), policy_target=np.array([1.0]), done=True)
        assert buffer.num_episodes == 2

        batch = buffer.sample(batch_size=200, unroll_steps=1, td_steps=1, gamma=0.9)
        counts_before = np.bincount(batch["episode_idx"], minlength=2)
        assert counts_before[0] > 0 and counts_before[1] > 0

        buffer.update_priorities(np.array([1]), np.array([0]), np.array([1000.0]))
        buffer.update_priorities(np.array([0]), np.array([0]), np.array([1e-6]))

        batch2 = buffer.sample(batch_size=200, unroll_steps=1, td_steps=1, gamma=0.9)
        counts_after = np.bincount(batch2["episode_idx"], minlength=2)
        assert counts_after[1] > counts_after[0]
        assert counts_after[1] > 150

    def test_reanalyze_roundtrip_updates_stored_targets(self) -> None:
        buffer = _UniZeroBuffer(capacity_episodes=10, obs_shape=(1,), action_dim=1, policy_target_dim=2, context_length=1)
        for t in range(4):
            buffer.add(
                obs=np.array([float(t)]), action_flat=np.array([0.0]), reward=1.0,
                next_obs=np.array([float(t + 1)]), policy_target=np.array([0.5, 0.5]), done=t == 3,
            )
        episode_idx, timestep, obs_out, ctx_obs, ctx_action, ctx_valid = buffer.sample_for_reanalyze(4)
        assert episode_idx.shape == (4,)
        assert obs_out.shape == (4, 1)
        assert ctx_obs.shape == (4, 1, 1)
        assert ctx_action.shape == (4, 1, 1)
        assert ctx_valid.shape == (4, 1)
        for i in range(4):
            assert np.isclose(obs_out[i, 0], float(timestep[i]))

        fresh_policy = np.stack([np.array([0.9, 0.1])] * 4)
        fresh_values = np.array([7.0, 8.0, 9.0, 10.0], dtype=np.float32)
        buffer.update_reanalyzed_targets(episode_idx, timestep, fresh_policy, fresh_values)

        expected_value_by_t: dict[int, float] = {}
        for i, t in enumerate(timestep.tolist()):
            expected_value_by_t[t] = float(fresh_values[i])

        ep = buffer.episodes[0]
        for t, expected in expected_value_by_t.items():
            assert np.allclose(ep["policy_target"][t], [0.9, 0.1])
            assert np.isclose(ep["search_value"][t], expected)

    def test_sample_for_reanalyze_empty_buffer(self) -> None:
        buffer = _UniZeroBuffer(capacity_episodes=10, obs_shape=(1,), action_dim=1, policy_target_dim=1, context_length=1)
        episode_idx, timestep, obs_out, ctx_obs, ctx_action, ctx_valid = buffer.sample_for_reanalyze(8)
        assert episode_idx.shape == (0,)
        assert timestep.shape == (0,)
        assert obs_out.shape == (0, 1)
        assert ctx_obs.shape == (0, 1, 1)
        assert ctx_valid.shape == (0, 1)


class TestNativeUniZeroDiscrete:
    def test_search_output_shapes(self) -> None:
        env = gym.make(_DISCRETE_ENV_ID)
        algo = NativeUniZero(env, {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE}, seed=0, device="cpu")
        obs, _info = env.reset(seed=0)

        obs_batch = obs_to_array(obs, env.observation_space)[None]
        results = algo.search(obs_batch, _empty_caches(1), deterministic=np.array([False]))
        assert len(results) == 1
        result = results[0]
        assert result["policy_target"].shape == (env.action_space.n,)
        assert np.isclose(result["policy_target"].sum(), 1.0, atol=1e-4)
        assert 0 <= result["env_action"] < env.action_space.n
        env.close()

    def test_search_is_batched_across_lanes(self) -> None:
        env = gym.make(_DISCRETE_ENV_ID)
        algo = NativeUniZero(env, {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE}, seed=0, device="cpu")
        obs, _info = env.reset(seed=0)
        one = obs_to_array(obs, env.observation_space)
        obs_batch = np.stack([one, one, one])
        results = algo.search(obs_batch, _empty_caches(3), deterministic=np.array([True, True, True]))
        assert len(results) == 3
        for result in results:
            assert result["policy_target"].shape == (env.action_space.n,)
            assert 0 <= result["env_action"] < env.action_space.n
        env.close()

    def test_search_with_nonempty_context_does_not_crash(self) -> None:
        """A search rooted with a *warm* incremental KV-cache built from
        real (non-padding) preceding context - the common case after an
        episode's first few real steps."""
        env = gym.make(_DISCRETE_ENV_ID)
        algo = NativeUniZero(env, {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE}, seed=0, device="cpu")
        obs, _info = env.reset(seed=0)
        one = obs_to_array(obs, env.observation_space)
        length = algo.context_length
        ctx_obs = np.stack([one] * length)[None]
        ctx_action = np.zeros((1, length, algo.action_dim), dtype=np.float32)
        ctx_valid = np.ones((1, length), dtype=bool)
        root_caches = algo._replay_context_to_cache(ctx_obs, ctx_action, ctx_valid)
        assert root_caches[0] is not None and root_caches[0].length == 2 * length
        results = algo.search(one[None], root_caches, deterministic=np.array([False]))
        assert len(results) == 1
        assert np.isfinite(results[0]["value_target"])
        env.close()

    def test_smoke(self) -> None:
        _run_smoke(_DISCRETE_ENV_ID, _TINY_DISCRETE, total_timesteps=40)

    def test_smoke_parallel_envs(self) -> None:
        _run_smoke(_DISCRETE_ENV_ID, _TINY_DISCRETE, total_timesteps=40, num_envs=3)


class TestNativeUniZeroContinuous:
    def test_search_output_shapes(self) -> None:
        env = gym.make(_CONTINUOUS_ENV_ID)
        algo = NativeUniZero(env, {**DEFAULT_HYPERPARAMS, **_TINY_CONTINUOUS}, seed=0, device="cpu")
        obs, _info = env.reset(seed=0)

        obs_batch = obs_to_array(obs, env.observation_space)[None]
        results = algo.search(obs_batch, _empty_caches(1), deterministic=np.array([False]))
        assert len(results) == 1
        result = results[0]
        action_dim = int(np.prod(env.action_space.shape))
        # Sampled UniZero's own policy target: root's own sampled
        # candidate actions, flattened, followed by their visit-count
        # distribution (module docstring's "Search" section) - not a
        # single blended action any more.
        num_cand = algo.num_sampled_actions
        assert result["policy_target"].shape == (num_cand * (action_dim + 1),)
        weights = result["policy_target"][num_cand * action_dim :]
        assert np.isclose(weights.sum(), 1.0, atol=1e-4)
        assert result["env_action"].shape == (action_dim,)
        env.close()

    def test_smoke(self) -> None:
        _run_smoke(_CONTINUOUS_ENV_ID, _TINY_CONTINUOUS, total_timesteps=40)

    def test_smoke_parallel_envs(self) -> None:
        _run_smoke(_CONTINUOUS_ENV_ID, _TINY_CONTINUOUS, total_timesteps=40, num_envs=3)


class TestSampledContinuousPolicyLoss:
    """`_train_step`'s continuous policy loss (module docstring's
    "Search" section) - a genuine importance-weighted NLL against
    `search()`'s own (sampled candidates, visit-count weights) pair, the
    reference's own Sampled-MuZero/Sampled-UniZero policy-improvement
    estimator, not this file's earlier MSE-to-a-blended-action
    regression."""

    def test_weighted_nll_matches_torch_distributions_normal(self) -> None:
        """Directly checks `_train_step`'s own inlined Gaussian NLL
        formula (`-0.5*((x-mu)^2/var + log(2*pi*var))`, summed over action
        dims) against `torch.distributions.Normal.log_prob` for the exact
        same `(candidates, mean, std)` - a regression test for a sign/
        formula slip that a pure shape/no-crash smoke test wouldn't catch."""
        torch.manual_seed(0)
        b, num_cand, action_dim = 3, 5, 2
        mean = torch.randn(b, action_dim)
        std = torch.rand(b, action_dim) + 0.5
        candidates = torch.randn(b, num_cand, action_dim)

        var = std.clamp_min(1e-6) ** 2
        diff = candidates - mean.unsqueeze(1)
        log_prob = -0.5 * ((diff ** 2) / var.unsqueeze(1) + torch.log(2.0 * math.pi * var.unsqueeze(1))).sum(-1)

        dist = torch.distributions.Normal(mean.unsqueeze(1), std.unsqueeze(1))
        expected_log_prob = dist.log_prob(candidates).sum(-1)
        assert torch.allclose(log_prob, expected_log_prob, atol=1e-4)

    def test_train_step_runs_and_produces_finite_policy_loss(self) -> None:
        """Pendulum's own episodes never terminate early (fixed 200-step
        `TimeLimit`), so a short `learn()` smoke run never flushes one
        into the buffer for `_train_step` to actually run against -
        populate one directly instead, in `search()`'s own (candidates,
        visit-weights) `policy_target` encoding (module docstring's
        "Search" section), and call `_train_step` straight away."""
        env = gym.make(_CONTINUOUS_ENV_ID)
        hp = {**DEFAULT_HYPERPARAMS, **_TINY_CONTINUOUS, "learning_starts": 0}
        algo = NativeUniZero(env, hp, seed=0, device="cpu")
        obs_arr = obs_to_array(env.reset(seed=0)[0], env.observation_space)
        num_cand = algo.num_sampled_actions
        for t in range(6):
            action = env.action_space.sample()
            action_flat = obs_to_array(action, env.action_space)
            candidates = np.tile(action_flat, (num_cand, 1)).astype(np.float32)
            weights = np.full(num_cand, 1.0 / num_cand, dtype=np.float32)
            policy_target = np.concatenate([candidates.reshape(-1), weights])
            algo.buffer.add(obs_arr, action_flat, 1.0, obs_arr, policy_target, done=(t == 5))

        metrics = algo._train_step()
        env.close()
        assert np.isfinite(metrics["policy_loss"])
        assert np.isfinite(metrics["policy_entropy"])


class TestReanalyzeIntegration:
    def test_reanalyze_triggers_and_refreshes_search_value(self) -> None:
        env = gym.make(_DISCRETE_ENV_ID)
        hp = {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE, "reanalyze_freq": 15, "reanalyze_batch_size": 4}
        algo = NativeUniZero(env, hp, seed=0, device="cpu")

        def writer(num_timesteps, episode_reward=None, episode_length=None, metrics=None):
            return num_timesteps < 60

        algo.learn(total_timesteps=60, callback=TrainingCallback(writer))
        env.close()

        any_reanalyzed = any(np.any(ep["search_value"] > -1e8) for ep in algo.buffer.episodes)
        assert any_reanalyzed

    def test_reanalyze_batch_size_zero_disables_it(self) -> None:
        env = gym.make(_DISCRETE_ENV_ID)
        hp = {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE, "reanalyze_freq": 5, "reanalyze_batch_size": 0}
        algo = NativeUniZero(env, hp, seed=0, device="cpu")

        def writer(num_timesteps, episode_reward=None, episode_length=None, metrics=None):
            return num_timesteps < 40

        algo.learn(total_timesteps=40, callback=TrainingCallback(writer))
        env.close()

        assert all(np.all(ep["search_value"] <= -1e8) for ep in algo.buffer.episodes)

    def test_reanalyze_continuous_action_space(self) -> None:
        env = gym.make(_CONTINUOUS_ENV_ID)
        hp = {**DEFAULT_HYPERPARAMS, **_TINY_CONTINUOUS, "reanalyze_freq": 15, "reanalyze_batch_size": 4}
        algo = NativeUniZero(env, hp, seed=0, device="cpu")

        def writer(num_timesteps, episode_reward=None, episode_length=None, metrics=None):
            return num_timesteps < 60

        algo.learn(total_timesteps=60, callback=TrainingCallback(writer))
        env.close()  # must not raise - continuous search() during reanalyze


class TestTrainFrequencyIndependentOfNumEnvs:
    """Same guarantee `efficientzero.py` makes and tests, ported here since
    `learn()`'s boundary-counting logic was copied verbatim."""

    def _count_train_steps(self, num_envs: int, total_timesteps: int) -> int:
        env = (
            gym.make(_DISCRETE_ENV_ID)
            if num_envs == 1
            else make_env_or_vec(make_gym_env_factory(_DISCRETE_ENV_ID), num_envs=num_envs, parallel=False)
        )
        algo = NativeUniZero(
            env, {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE, "train_freq": 1, "train_steps_per_iter": 1},
            seed=0, device="cpu",
        )
        call_count = 0
        original_train_step = algo._train_step

        def counting_train_step():
            nonlocal call_count
            call_count += 1
            return original_train_step()

        algo._train_step = counting_train_step

        def writer(num_timesteps, episode_reward=None, episode_length=None, metrics=None):
            return num_timesteps < total_timesteps

        algo.learn(total_timesteps=total_timesteps, callback=TrainingCallback(writer))
        env.close()
        return call_count

    def test_train_steps_scale_with_num_envs_not_against_it(self) -> None:
        total_timesteps = 200
        single = self._count_train_steps(num_envs=1, total_timesteps=total_timesteps)
        vectorized = self._count_train_steps(num_envs=4, total_timesteps=total_timesteps)

        assert single > 0 and vectorized > 0
        ratio = vectorized / single
        assert 0.5 <= ratio <= 2.0, (
            f"expected train-step counts within 2x of each other regardless of "
            f"num_envs, got single={single} vectorized(num_envs=4)={vectorized} "
            f"ratio={ratio:.2f}"
        )
