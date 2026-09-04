"""Coverage for `rl_core/algorithms/native/researchimzero.py` (ResearchImZero
- UniZero's Transformer-world-model architecture + `efficientzero.py`'s own
training recipe, see that module's own docstring for why). Same "tiny
hyperparams, just check shapes/no-crash, not learned quality" convention as
`test_unizero.py`/`test_efficientzero.py`."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rl_core.algorithms.base import TrainingCallback
from rl_core.algorithms.native.researchimzero import (
    DEFAULT_HYPERPARAMS,
    NativeResearchImZero,
    _HalvingSchedule,
    _MinMaxStats,
    _Predictor,
    _Projector,
    _ResearchImZeroBuffer,
    _SearchNode,
    _select_action,
    _sequential_halving,
)
from rl_core.algorithms.vec_env import make_env_or_vec, make_gym_env_factory

_DISCRETE_ENV_ID = "CartPole-v1"
_CONTINUOUS_ENV_ID = "Pendulum-v1"

_TINY_DISCRETE = {
    "embed_dim": 16, "num_layers": 1, "num_heads": 2, "context_length": 2,
    "buffer_size": 200, "batch_size": 4, "unroll_steps": 3, "td_steps": 3,
    "num_sampled_actions": 4, "num_simulations": 6, "num_top_actions": 4,
    "learning_starts": 5, "train_freq": 1, "value_support_size": 20,
    "proj_dim": 8,
}
_TINY_CONTINUOUS = {**_TINY_DISCRETE}


class _OneStepResearchEnv(gym.Env):
    observation_space = gym.spaces.Box(-1000, 1000, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def __init__(self) -> None:
        self.episode = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        return np.full(4, 10 * self.episode, dtype=np.float32), {}

    def step(self, action):
        del action
        return np.full(4, 10 * self.episode + 1, dtype=np.float32), 1.0, True, False, {}


def _run_smoke(env_id: str, hyperparams: dict, total_timesteps: int, num_envs: int = 1) -> None:
    if num_envs == 1:
        env = gym.make(env_id)
    else:
        env = make_env_or_vec(make_gym_env_factory(env_id), num_envs=num_envs, parallel=False)
    algo = NativeResearchImZero(env, {**DEFAULT_HYPERPARAMS, **hyperparams}, seed=0, device="cpu")

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
    obs2, reward2, terminated2, truncated2, _info2 = predict_env.step(action)
    algo.predict(obs2, deterministic=True, episode_start=False)

    with tempfile.TemporaryDirectory() as d:
        checkpoint = Path(d) / "model.pt"
        algo.save(checkpoint)
        loaded = NativeResearchImZero.load(checkpoint, gym.make(env_id), device="cpu")
        loaded.predict(obs, deterministic=True, episode_start=True)
    predict_env.close()
    env.close()


class TestNativeResearchImZeroDiscrete:
    def test_smoke_single_env(self) -> None:
        _run_smoke(_DISCRETE_ENV_ID, _TINY_DISCRETE, total_timesteps=40, num_envs=1)

    def test_smoke_multi_env(self) -> None:
        _run_smoke(_DISCRETE_ENV_ID, _TINY_DISCRETE, total_timesteps=40, num_envs=3)

    def test_search_output_shapes(self) -> None:
        env = gym.make(_DISCRETE_ENV_ID)
        algo = NativeResearchImZero(env, {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE}, seed=0, device="cpu")
        obs_arr = np.stack([np.asarray(env.observation_space.sample(), dtype=np.float32) for _ in range(2)])
        results = algo.search(obs_arr, [None, None])
        assert len(results) == 2
        for r in results:
            assert r["policy_target"].shape == (algo.n_actions,)
            assert np.isclose(float(r["policy_target"].sum()), 1.0, atol=1e-4)
            assert isinstance(r["env_action"], (int, np.integer))
        env.close()


class TestVectorCollectionRegressions:
    def test_default_training_cadence_is_one_update_per_vector_iteration(self) -> None:
        env = make_env_or_vec(_OneStepResearchEnv, num_envs=24, parallel=False)
        algo = NativeResearchImZero(
            env,
            {
                **DEFAULT_HYPERPARAMS,
                **_TINY_DISCRETE,
                "learning_starts": 0,
                "auto_scale_replay_ratio": 0,
            },
            seed=0,
            device="cpu",
        )
        train_calls = 0

        def fake_search(obs_batch, root_caches, deterministic=None):
            del root_caches, deterministic
            return [
                {
                    "env_action": 0,
                    "policy_target": np.full(algo.n_actions, 1.0 / algo.n_actions, dtype=np.float32),
                    "value_target": 0.0,
                }
                for _ in range(obs_batch.shape[0])
            ]

        def fake_train_step():
            nonlocal train_calls
            train_calls += 1
            return {}

        algo.search = fake_search
        algo._train_step = fake_train_step
        algo._advance_lane_caches = lambda caches, obs, actions: [None] * len(caches)
        algo.learn(48, TrainingCallback(lambda *args, **kwargs: True))
        assert train_calls == 2
        assert algo.train_steps_per_iter == 1
        env.close()

    def test_autoscale_is_explicit_and_not_multiplied_by_crossed_boundaries(self) -> None:
        env = make_env_or_vec(_OneStepResearchEnv, num_envs=24, parallel=False)
        algo = NativeResearchImZero(
            env,
            {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE, "auto_scale_replay_ratio": 1},
            seed=0,
            device="cpu",
        )
        assert algo.train_steps_per_iter == 6
        env.close()

    def test_terminal_replay_observation_is_not_same_step_reset_observation(self) -> None:
        env = make_env_or_vec(_OneStepResearchEnv, num_envs=2, parallel=False)
        algo = NativeResearchImZero(
            env,
            {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE, "learning_starts": 1000},
            seed=0,
            device="cpu",
        )
        algo.learn(2, TrainingCallback(lambda *args, **kwargs: True))
        assert algo.buffer.num_episodes == 2
        for episode in algo.buffer.episodes:
            # Episode 1 starts at 10, truly terminates at 11, while the
            # SAME_STEP reset observation for episode 2 is 20.
            assert np.allclose(episode["obs"][0], 10.0)
            assert np.allclose(episode["next_obs"][0], 11.0)
        assert all(cache is None for cache in algo._lane_cache)
        env.close()


class TestNativeResearchImZeroContinuous:
    def test_smoke_single_env(self) -> None:
        _run_smoke(_CONTINUOUS_ENV_ID, _TINY_CONTINUOUS, total_timesteps=40, num_envs=1)

    def test_search_output_shapes(self) -> None:
        env = gym.make(_CONTINUOUS_ENV_ID)
        algo = NativeResearchImZero(env, {**DEFAULT_HYPERPARAMS, **_TINY_CONTINUOUS}, seed=0, device="cpu")
        obs_arr = np.stack([np.asarray(env.observation_space.sample(), dtype=np.float32) for _ in range(2)])
        results = algo.search(obs_arr, [None, None])
        assert len(results) == 2
        for r in results:
            # `policy_target` here is the visit-weighted *average*
            # candidate action (`efficientzero.py`'s own, simpler choice -
            # module docstring's "Search" section) - shape `(action_dim,)`,
            # not the flattened candidates+weights `unizero.py` uses.
            assert r["policy_target"].shape == (algo.action_dim,)
            assert env.action_space.contains(np.asarray(r["env_action"], dtype=np.float32))
        env.close()


class TestGumbelSearchPrimitives:
    """`_select_action`/`_sequential_halving`/`_HalvingSchedule` - the
    Gumbel-Top-k + Sequential Halving primitives ported from
    `efficientzero.py` (module docstring's "Search" section)."""

    def test_halving_schedule_never_exceeds_budget(self) -> None:
        schedule = _HalvingSchedule(num_simulations=32, num_top_actions=8)
        cutoffs = [schedule.next_cutoff]
        for sim_idx in range(32):
            if schedule.maybe_advance(sim_idx):
                cutoffs.append(schedule.next_cutoff)
        assert all(c <= 32 for c in cutoffs)
        assert schedule.current_top == 1

    def test_halving_schedule_first_phase_is_multiple_of_m(self) -> None:
        schedule = _HalvingSchedule(num_simulations=32, num_top_actions=8)
        assert schedule.next_cutoff % 8 == 0

    def test_select_action_at_root_does_equal_visit_round_robin(self) -> None:
        root = _SearchNode(prior=1.0)
        root.expand(np.array([0.5, 0.3, 0.2]), cache=None, reward_value=0.0)
        root.selected_children_idx = [0, 1, 2]
        minmax = _MinMaxStats(delta=0.01)
        # All children start with 0 visits - the earliest-ranked survivor
        # (index 0) must be picked first.
        idx = _select_action(root, minmax, discount=0.99, c_visit=50.0, c_scale=0.1)
        assert idx == 0
        root.children[0].visit_count = 1
        idx2 = _select_action(root, minmax, discount=0.99, c_visit=50.0, c_scale=0.1)
        assert idx2 == 1

    def test_sequential_halving_keeps_top_scoring_survivors(self) -> None:
        root = _SearchNode(prior=1.0)
        root.expand(np.array([0.0, 0.0, 0.0, 0.0]), cache=None, reward_value=0.0)
        root.selected_children_idx = [0, 1, 2, 3]
        minmax = _MinMaxStats(delta=0.01)
        gumbel = np.array([10.0, 0.0, 5.0, -10.0])
        _sequential_halving(root, gumbel, minmax, keep=2, discount=0.99, c_visit=50.0, c_scale=0.1)
        assert set(root.selected_children_idx) == {0, 2}

    def test_sequential_halving_is_a_no_op_with_one_survivor(self) -> None:
        root = _SearchNode(prior=1.0)
        root.expand(np.array([1.0]), cache=None, reward_value=0.0)
        root.selected_children_idx = [0]
        minmax = _MinMaxStats(delta=0.01)
        _sequential_halving(root, np.array([0.0]), minmax, keep=1, discount=0.99, c_visit=50.0, c_scale=0.1)
        assert root.selected_children_idx == [0]


class TestSimSiamModules:
    """`_Projector`/`_Predictor` (module docstring's "no target network"
    section) - `efficientzero.py`'s own SimSiam consistency-loss modules,
    ported verbatim."""

    def test_projector_output_shape(self) -> None:
        proj = _Projector(embed_dim=16, proj_dim=8)
        z = torch.randn(4, 16)
        out = proj(z)
        assert out.shape == (4, 8)

    def test_predictor_output_shape(self) -> None:
        pred = _Predictor(proj_dim=8)
        z = torch.randn(4, 8)
        out = pred(z)
        assert out.shape == (4, 8)

    def test_train_step_produces_finite_consistency_loss(self) -> None:
        env = gym.make(_DISCRETE_ENV_ID)
        algo = NativeResearchImZero(env, {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE}, seed=0, device="cpu")
        obs, _info = env.reset(seed=0)
        for step in range(20):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated) or step == 19  # force at least one finalized episode
            policy_target = np.full(algo.n_actions, 1.0 / algo.n_actions, dtype=np.float32)
            action_flat = np.zeros(algo.n_actions, dtype=np.float32)
            action_flat[action] = 1.0
            algo.buffer.add(obs, action_flat, float(reward), next_obs, policy_target, done, lane=0)
            obs = next_obs if not done else env.reset()[0]
        metrics = algo._train_step()
        assert np.isfinite(metrics["consistency_loss"])
        assert np.isfinite(metrics["value_loss"])
        assert np.isfinite(metrics["policy_loss"])
        env.close()


class TestValueTargetMaxBlend:
    """`_train_step`'s `value_target = max(td_target, search_value)`
    (module docstring's "no target network" section, `efficientzero.py`'s
    own `value_target: 'max'` mode) - `_ResearchImZeroBuffer` must
    actually carry a `search_value` array per transition for this to have
    anything to blend with."""

    def test_buffer_sample_carries_search_value_field(self) -> None:
        env = gym.make(_DISCRETE_ENV_ID)
        algo = NativeResearchImZero(env, {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE}, seed=0, device="cpu")
        obs, _info = env.reset(seed=0)
        for step in range(20):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated) or step == 19  # force at least one finalized episode
            policy_target = np.full(algo.n_actions, 1.0 / algo.n_actions, dtype=np.float32)
            action_flat = np.zeros(algo.n_actions, dtype=np.float32)
            action_flat[action] = 1.0
            algo.buffer.add(obs, action_flat, float(reward), next_obs, policy_target, done, lane=0)
            obs = next_obs if not done else env.reset()[0]
        batch = algo.buffer.sample(algo.batch_size, algo.unroll_steps, algo.td_steps, algo.gamma)
        assert "search_value" in batch
        assert batch["search_value"].shape == (algo.batch_size, algo.unroll_steps)
        # Never-reanalyzed transitions keep the buffer's own sentinel.
        assert np.all(batch["search_value"] <= _ResearchImZeroBuffer._NO_SEARCH_VALUE + 1.0)
        env.close()


class TestAdaptiveLossWeights:
    """`adaptive_loss_weights`'s Kendall-et-al. `precision*loss + log_var`
    total loss - regression coverage for the exact real-run failure this
    file's own postmortem found: `consistency_loss`'s old `-cos_sim`
    formulation (`[-1, 1]`, can go negative) had no interior minimum
    under this weighting scheme, so `loss_weight_consistency` climbed
    unbounded (measured 2 -> 9896 in under 7k real NetHack steps,
    `total_loss` to `-13486`) instead of converging, and effectively
    starved every other task's gradient. Fixed by shifting consistency
    to `1 - cos_sim` (`[0, 2]`, always non-negative) plus a defense-in-
    depth `log_var` clamp - this test runs enough real `_train_step`
    calls that the old bug would already show clear divergence, and
    asserts every `loss_weight_*` stays inside the clamp's own bound."""

    def test_consistency_loss_is_never_negative(self) -> None:
        """The actual root-cause regression check: `adaptive_loss_
        weights`'s `precision*loss + log_var` term only has an interior
        minimum (so `precision` converges instead of running away to
        `+inf`) if `loss >= 0`. `consistency_loss` is a cosine-similarity-
        based loss, the one task loss here not automatically non-negative
        by construction (unlike the categorical cross-entropies/MSE the
        other three tasks use) - it must be `1 - cos_sim` (`[0, 2]`), not
        `-cos_sim` (`[-1, 1]`, and reliably negative once the model is any
        good at it - a real NetHack run measured `loss_weight_consistency`
        climbing `2 -> 9896` in under 7k steps off exactly this sign bug,
        `total_loss` diverging to `-13486`, training effectively stalling
        since every other task's gradient got swamped). A short, high-
        quality-consistency-prediction run is enough to catch a regression
        here directly, immediately - no need to wait out however many
        `_train_step` calls an actual unbounded blowup would take to
        become numerically obvious (which, empirically, is *not* fast
        with tiny test-sized networks and short buffers - the divergence
        speed depends on how negative `consistency_loss` gets, which
        depends on how good the SimSiam prediction already is, not on the
        bug's presence alone)."""
        env = gym.make(_DISCRETE_ENV_ID)
        algo = NativeResearchImZero(
            env, {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE, "adaptive_loss_weights": 1}, seed=0, device="cpu",
        )
        obs, _info = env.reset(seed=0)
        for step in range(60):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated) or step == 59
            policy_target = np.full(algo.n_actions, 1.0 / algo.n_actions, dtype=np.float32)
            action_flat = np.zeros(algo.n_actions, dtype=np.float32)
            action_flat[action] = 1.0
            algo.buffer.add(obs, action_flat, float(reward), next_obs, policy_target, done, lane=0)
            obs = next_obs if not done else env.reset(seed=step)[0]
        for _ in range(10):
            metrics = algo._train_step()
            assert metrics["consistency_loss"] >= -1e-5, (
                f"consistency_loss={metrics['consistency_loss']} went negative - "
                "adaptive_loss_weights's precision term has no interior minimum here anymore"
            )
        env.close()

    def test_loss_weights_stay_within_their_own_clamp(self) -> None:
        env = gym.make(_DISCRETE_ENV_ID)
        algo = NativeResearchImZero(
            env, {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE, "adaptive_loss_weights": 1}, seed=0, device="cpu",
        )
        obs, _info = env.reset(seed=0)
        for step in range(60):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated) or step == 59
            policy_target = np.full(algo.n_actions, 1.0 / algo.n_actions, dtype=np.float32)
            action_flat = np.zeros(algo.n_actions, dtype=np.float32)
            action_flat[action] = 1.0
            algo.buffer.add(obs, action_flat, float(reward), next_obs, policy_target, done, lane=0)
            obs = next_obs if not done else env.reset(seed=step)[0]

        # `exp(5) ~= 148.4` - `_train_step`'s own `log_var.clamp(-5.0, 5.0)`
        # ceiling (defense-in-depth on top of the sign fix above); a small
        # margin above it catches an off-by-one without re-allowing the
        # kind of unbounded blowup this regresses against.
        max_allowed_weight = math.exp(5.0) * 1.05
        weight_keys = ["loss_weight_reward", "loss_weight_value", "loss_weight_policy"]
        for _ in range(30):
            metrics = algo._train_step()
            assert np.isfinite(metrics["total_loss"])
            # `consistency_loss` deliberately isn't adaptively weighted
            # at all anymore (see `__init__`'s own `self.loss_log_vars`
            # comment: no genuine noise floor -> it just camps at this
            # exact clamp ceiling instead, ~200x `policy`'s concurrent
            # weight, real-run-measured to stall `episode_reward_mean`)
            # - there's no `loss_weight_consistency` metric key to check.
            assert "loss_weight_consistency" not in metrics
            for key in weight_keys:
                assert 0.0 < metrics[key] <= max_allowed_weight, f"{key}={metrics[key]} escaped its own clamp"
        env.close()
