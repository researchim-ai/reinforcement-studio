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
import pytest
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
            assert r["root_cache"] is not None
        env.close()


class TestVectorCollectionRegressions:
    def test_default_training_cadence_is_two_updates_per_vector_iteration(self) -> None:
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
        assert train_calls == 4
        assert algo.train_steps_per_iter == 2
        env.close()

    def test_autoscale_is_explicit_and_not_multiplied_by_crossed_boundaries(self) -> None:
        env = make_env_or_vec(_OneStepResearchEnv, num_envs=24, parallel=False)
        algo = NativeResearchImZero(
            env,
            {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE, "auto_scale_replay_ratio": 1},
            seed=0,
            device="cpu",
        )
        assert algo.train_steps_per_iter == 12
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

    def test_halving_schedule_never_eliminates_unvisited_custom_top_actions(self) -> None:
        schedule = _HalvingSchedule(num_simulations=32, num_top_actions=16)
        assert schedule.current_top == 16
        assert schedule.next_cutoff == 16
        for sim_idx in range(15):
            assert schedule.maybe_advance(sim_idx) is False
            assert schedule.current_top == 16
        assert schedule.maybe_advance(15) is True
        assert schedule.current_top == 8

    def test_halving_schedule_phases_consume_exact_total_budget(self) -> None:
        schedule = _HalvingSchedule(num_simulations=32, num_top_actions=16)
        phase_allocations: dict[int, int] = {}
        for sim_idx in range(32):
            top = schedule.current_top
            phase_allocations[top] = phase_allocations.get(top, 0) + 1
            schedule.maybe_advance(sim_idx)
        assert sum(phase_allocations.values()) == 32
        assert phase_allocations[16] == 16
        assert all(
            allocated % top == 0
            for top, allocated in phase_allocations.items()
            if top > 1
        )
        assert schedule.next_cutoff == 32

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
        assert np.isfinite(metrics["closed_loop_loss"])
        env.close()


class TestStreamingReplayAndTargets:
    """Active replay, context targets and bounded reanalysis blend."""

    def test_active_lane_is_sampleable_before_episode_end(self) -> None:
        env = gym.make(_DISCRETE_ENV_ID)
        algo = NativeResearchImZero(env, {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE}, seed=0, device="cpu")
        obs, _info = env.reset(seed=0)
        for _ in range(4):
            action = env.action_space.sample()
            next_obs, reward, _terminated, _truncated, _info = env.step(action)
            action_flat = np.eye(algo.n_actions, dtype=np.float32)[action]
            policy_target = np.full(algo.n_actions, 1.0 / algo.n_actions, dtype=np.float32)
            algo.buffer.add(obs, action_flat, float(reward), next_obs, policy_target, False, lane=0)
            obs = next_obs

        assert algo.buffer.num_episodes == 0
        assert algo.buffer.num_replay_transitions == 4
        batch = algo.buffer.sample(8, algo.unroll_steps, algo.td_steps, algo.gamma)
        assert np.all(batch["episode_idx"] < 0)
        assert np.any(batch["td_bootstrap_mask"] > 0)
        assert batch["bootstrap_action"].shape[1] == algo.unroll_steps + algo.td_steps
        fresh_priorities = np.full(8, 0.25, dtype=np.float32)
        algo.buffer.update_priorities(batch["episode_idx"], batch["timestep"], fresh_priorities)
        for timestep in np.unique(batch["timestep"]):
            assert np.isclose(algo.buffer._cur[0]["priority"][int(timestep)], 0.25)

        ep_idx, timesteps, *_ = algo.buffer.sample_for_reanalyze(4)
        assert np.all(ep_idx < 0)
        refreshed_policy = np.full((4, algo.n_actions), 1.0 / algo.n_actions, dtype=np.float32)
        refreshed_values = np.arange(4, dtype=np.float32)
        algo.buffer.update_reanalyzed_targets(ep_idx, timesteps, refreshed_policy, refreshed_values)
        assert any(gen > 0 for gen in algo.buffer._cur[0]["reanalyzed_gen"])
        active_priorities = np.asarray(algo.buffer._cur[0]["priority"]).copy()
        active_generations = np.asarray(algo.buffer._cur[0]["reanalyzed_gen"]).copy()
        algo.buffer._flush_episode(0)
        assert np.allclose(algo.buffer.episodes[0]["priority"], active_priorities)
        assert np.array_equal(algo.buffer.episodes[0]["reanalyzed_gen"], active_generations)
        env.close()

    def test_recent_replay_samples_only_configured_active_tail(self) -> None:
        env = gym.make(_DISCRETE_ENV_ID)
        algo = NativeResearchImZero(env, {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE}, seed=0, device="cpu")
        algo.buffer.recent_fraction = 1.0
        algo.buffer.recent_window = 1
        obs, _info = env.reset(seed=0)
        for _ in range(4):
            action = env.action_space.sample()
            next_obs, reward, _terminated, _truncated, _info = env.step(action)
            algo.buffer.add(
                obs,
                np.eye(algo.n_actions, dtype=np.float32)[action],
                float(reward),
                next_obs,
                np.full(algo.n_actions, 1.0 / algo.n_actions, dtype=np.float32),
                False,
                lane=0,
            )
            obs = next_obs
        batch = algo.buffer.sample(16, algo.unroll_steps, algo.td_steps, algo.gamma)
        assert np.all(batch["timestep"] == 3)
        assert np.allclose(batch["is_weight"], 1.0)
        env.close()

    def test_adaptive_search_budget_reaches_configured_maximum(self) -> None:
        env = gym.make(_DISCRETE_ENV_ID)
        algo = NativeResearchImZero(
            env,
            {
                **DEFAULT_HYPERPARAMS,
                **_TINY_DISCRETE,
                "num_simulations_initial": 2,
                "num_simulations": 10,
                "search_ramp_steps": 100,
            },
            seed=0,
            device="cpu",
        )
        algo._num_timesteps = 0
        assert algo._current_num_simulations() == 2
        algo._num_timesteps = 100
        assert algo._current_num_simulations() == 2
        algo._model_error_ema = algo.search_model_error_low
        algo._num_timesteps = 50
        assert algo._current_num_simulations() == 6
        algo._num_timesteps = 100
        assert algo._current_num_simulations() == 10
        env.close()

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

    def test_checkpoint_roundtrips_finalized_and_active_replay(
        self,
        tmp_path: Path,
    ) -> None:
        env = gym.make(_DISCRETE_ENV_ID)
        params = {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE}
        algo = NativeResearchImZero(env, params, seed=0, device="cpu")
        policy = np.full(algo.n_actions, 1.0 / algo.n_actions, dtype=np.float32)
        action = np.eye(algo.n_actions, dtype=np.float32)[0]
        metadata = {
            "policy_target_valid": True,
            "search_budget": 4,
            "model_expansions": 6,
            "behavior_policy": policy,
            "audit_type": "targeted",
            "stage_index": 2,
            "stage_id": "medium",
            "nominal_stage_budget": 4,
            "requested_budget": 4,
            "predicted_stage_index": 1,
            "predicted_stage_id": "small",
            "predicted_requested_budget": 2,
            "executed_search_expansions": 4,
            "common_evaluator_expansions": 2,
            "total_model_calls": 6,
            "termination_reason": "controller_stop",
        }
        algo.buffer.add(
            np.zeros(4, dtype=np.float32),
            action,
            1.5,
            np.ones(4, dtype=np.float32),
            policy,
            True,
            **metadata,
        )
        algo.buffer.add(
            np.ones(4, dtype=np.float32),
            action,
            2.5,
            np.full(4, 2.0, dtype=np.float32),
            policy,
            False,
            **metadata,
        )
        indices = np.asarray([0, -1])
        timesteps = np.asarray([0, 0])
        algo.buffer.update_priorities(indices, timesteps, np.asarray([1.7, 1.8]))
        algo.buffer.update_raw_learning_errors(
            indices, timesteps, np.asarray([0.4, 0.6]),
        )
        algo.buffer.update_reanalyzed_targets(
            indices,
            timesteps,
            np.stack([policy, policy]),
            np.asarray([3.0, 4.0]),
            np.asarray([True, True]),
        )
        checkpoint = tmp_path / "research-replay.pt"
        algo.save(checkpoint)
        loaded = NativeResearchImZero.load(
            checkpoint, gym.make(_DISCRETE_ENV_ID), device="cpu",
        )

        assert len(loaded.buffer.episodes) == 1
        assert len(loaded.buffer._cur[0]["reward"]) == 1
        assert loaded.buffer._reanalyze_generation == 1
        assert loaded.buffer._max_priority == 1.8
        assert loaded.buffer._episode_last_reanalyzed_gen == [1.0]
        assert loaded.buffer._episode_returns == [1.5]
        assert loaded.buffer._episode_alpha_sum == pytest.approx([1.7])
        assert loaded.buffer.raw_learning_errors_for(
            indices, timesteps,
        ).tolist() == pytest.approx([0.4, 0.6])
        assert loaded.buffer.episodes[0]["priority"].tolist() == [1.7]
        assert loaded.buffer._cur[0]["priority"] == [1.8]
        assert loaded.buffer.episodes[0]["search_value"].tolist() == [3.0]
        assert loaded.buffer._cur[0]["search_value"] == [4.0]
        assert loaded.buffer.episodes[0]["reanalyzed_gen"].tolist() == [1]
        assert loaded.buffer._cur[0]["reanalyzed_gen"] == [1]

        np.random.seed(5)
        sample = loaded.buffer.sample(16, 1, 1, loaded.gamma)
        assert np.all(sample["stage_index"][:, 0] == 2)
        assert np.all(sample["requested_budget"][:, 0] == 4)
        assert np.all(sample["executed_search_expansions"][:, 0] == 4)
        assert np.all(sample["common_evaluator_expansions"][:, 0] == 2)
        assert np.all(sample["total_model_calls"][:, 0] == 6)
        assert set(sample["termination_reason"][:, 0]) == {"controller_stop"}

        legacy_payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False,
        )
        legacy_payload.pop("replay_buffer_state")
        legacy_checkpoint = tmp_path / "research-legacy-empty-replay.pt"
        torch.save(legacy_payload, legacy_checkpoint)
        legacy_loaded = NativeResearchImZero.load(
            legacy_checkpoint, gym.make(_DISCRETE_ENV_ID), device="cpu",
        )
        assert len(legacy_loaded.buffer) == 0
        env.close()
        loaded.env.close()
        legacy_loaded.env.close()


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
