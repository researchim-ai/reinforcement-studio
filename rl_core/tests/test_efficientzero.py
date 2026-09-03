"""Coverage for `rl_core/algorithms/native/efficientzero.py` (EfficientZero
V2 — MuZero-family model-based planning, see that module's own docstring).
Same "tiny hyperparams, just check shapes/no-crash, not learned quality"
convention as `test_world_models.py`."""
from __future__ import annotations

import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np

import torch

from rl_core.algorithms.base import TrainingCallback
from rl_core.algorithms.native.efficientzero import (
    DEFAULT_HYPERPARAMS,
    NativeEfficientZero,
    _EfficientZeroBuffer,
    _logits_to_scalar,
    _scalar_to_two_hot,
    _signed_hyperbolic,
    _signed_parabolic,
)
from rl_core.algorithms.native.preprocessing import obs_to_array
from rl_core.algorithms.vec_env import make_env_or_vec, make_gym_env_factory

_DISCRETE_ENV_ID = "CartPole-v1"
_CONTINUOUS_ENV_ID = "Pendulum-v1"

_TINY_DISCRETE = {
    "latent_dim": 8, "hidden_dim": 16, "proj_dim": 8, "buffer_size": 200,
    "batch_size": 4, "unroll_steps": 3, "td_steps": 3, "num_sampled_actions": 4,
    "num_simulations": 6, "num_top_actions": 4, "learning_starts": 10, "train_freq": 1,
    "value_support_size": 20,
}
_TINY_CONTINUOUS = {**_TINY_DISCRETE}


def _run_smoke(env_id: str, hyperparams: dict, total_timesteps: int, num_envs: int = 1) -> None:
    if num_envs == 1:
        env = gym.make(env_id)
    else:
        env = make_env_or_vec(make_gym_env_factory(env_id), num_envs=num_envs, parallel=False)
    algo = NativeEfficientZero(env, {**DEFAULT_HYPERPARAMS, **hyperparams}, seed=0, device="cpu")

    steps: list[int] = []

    def writer(num_timesteps, episode_reward=None, episode_length=None, metrics=None):
        steps.append(num_timesteps)
        return num_timesteps < total_timesteps

    algo.learn(total_timesteps=total_timesteps, callback=TrainingCallback(writer))
    assert steps[-1] >= total_timesteps

    predict_env = gym.make(env_id)
    obs, _info = predict_env.reset(seed=1)
    action, _state = algo.predict(obs, deterministic=True)
    assert predict_env.action_space.contains(action)

    with tempfile.TemporaryDirectory() as d:
        checkpoint = Path(d) / "model.pt"
        algo.save(checkpoint)
        loaded = NativeEfficientZero.load(checkpoint, gym.make(env_id), device="cpu")
        loaded.predict(obs, deterministic=True)
    predict_env.close()
    env.close()


class TestCategoricalValueSupport:
    """`_signed_hyperbolic`/`_signed_parabolic`/`_scalar_to_two_hot` -
    see `efficientzero.py`'s "Categorical value/reward" module docstring
    bullet. Bugs here would silently miscalibrate every value/reward
    target in training without ever crashing anything."""

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
        # Decoding a two-hot's own logits (log of the target, since softmax
        # of a log-distribution recovers it) must land back near `x`.
        logits = torch.log(two_hot.clamp_min(1e-9))
        decoded = _logits_to_scalar(logits, support_size)
        assert torch.allclose(decoded, x, atol=0.5)

    def test_large_target_does_not_escape_the_support(self) -> None:
        # A target far outside the support must still produce a valid
        # (clamped-at-the-edge) two-hot rather than an out-of-bounds index.
        support_size = 10
        two_hot = _scalar_to_two_hot(torch.tensor([1e6]), support_size)
        assert torch.allclose(two_hot.sum(-1), torch.ones(1), atol=1e-5)
        assert torch.all(two_hot >= 0.0)


class TestEfficientZeroBuffer:
    def test_sample_shapes_and_masking(self) -> None:
        buffer = _EfficientZeroBuffer(capacity_episodes=10, obs_shape=(4,), action_dim=2, policy_target_dim=2)
        # One 3-step episode: sampling from the last step with unroll_steps=3
        # must mask out steps 1 and 2 (nothing left in the episode there).
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
        # Every sampled start index is one of {0, 1, 2}; whatever mask[b]
        # looks like, it must be a monotonically non-increasing run of 1s
        # then 0s (no "hole" of a valid step after an invalid one).
        for m in batch["mask"]:
            saw_zero = False
            for v in m:
                if v == 0.0:
                    saw_zero = True
                elif saw_zero:
                    raise AssertionError(f"mask has a 1 after a 0: {m}")

    def test_terminal_step_has_no_bootstrap(self) -> None:
        buffer = _EfficientZeroBuffer(capacity_episodes=10, obs_shape=(1,), action_dim=1, policy_target_dim=1)
        buffer.add(obs=np.array([0.0]), action_flat=np.array([0.0]), reward=1.0, next_obs=np.array([1.0]), policy_target=np.array([1.0]), done=True)
        batch = buffer.sample(batch_size=4, unroll_steps=1, td_steps=5, gamma=0.9)
        # The only real step is terminal -> no bootstrap continues past it.
        assert np.all(batch["td_bootstrap_mask"][:, 0] == 0.0)
        assert np.allclose(batch["td_reward"][:, 0], 1.0)

    def test_lanes_do_not_splice_episodes(self) -> None:
        # `num_lanes=2` — each lane's in-progress episode must stay separate
        # until *that* lane's own `done=True`, even when interleaved.
        buffer = _EfficientZeroBuffer(capacity_episodes=10, obs_shape=(1,), action_dim=1, policy_target_dim=1, num_lanes=2)
        buffer.add(obs=np.array([0.0]), action_flat=np.array([0.0]), reward=1.0, next_obs=np.array([1.0]), policy_target=np.array([1.0]), done=False, lane=0)
        buffer.add(obs=np.array([10.0]), action_flat=np.array([0.0]), reward=1.0, next_obs=np.array([11.0]), policy_target=np.array([1.0]), done=True, lane=1)
        buffer.add(obs=np.array([1.0]), action_flat=np.array([0.0]), reward=1.0, next_obs=np.array([2.0]), policy_target=np.array([1.0]), done=True, lane=0)
        assert buffer.num_episodes == 2
        lane1_ep = next(ep for ep in buffer.episodes if ep["obs"].shape[0] == 1)
        lane0_ep = next(ep for ep in buffer.episodes if ep["obs"].shape[0] == 2)
        assert np.allclose(lane1_ep["obs"], [[10.0]])
        assert np.allclose(lane0_ep["obs"], [[0.0], [1.0]])

    def test_sample_returns_per_transition_metadata(self) -> None:
        """`episode_idx`/`timestep`/`is_weight`/`search_value` - the extra
        keys `_train_step` needs for prioritized replay + reanalyze."""
        buffer = _EfficientZeroBuffer(capacity_episodes=10, obs_shape=(1,), action_dim=1, policy_target_dim=1)
        for t in range(5):
            buffer.add(
                obs=np.array([float(t)]), action_flat=np.array([0.0]), reward=1.0,
                next_obs=np.array([float(t + 1)]), policy_target=np.array([1.0]), done=t == 4,
            )
        batch = buffer.sample(batch_size=6, unroll_steps=2, td_steps=2, gamma=0.9)
        assert batch["episode_idx"].shape == (6,)
        assert batch["timestep"].shape == (6,)
        assert batch["is_weight"].shape == (6,)
        assert batch["search_value"].shape == (6, 2)
        # Never-reanalyzed transitions carry the "no search value" sentinel.
        assert np.all(batch["search_value"] <= -1e8)
        # All episode indices point at the one episode that exists.
        assert np.all(batch["episode_idx"] == 0)
        assert np.all(batch["timestep"] < 5)
        # IS weights are finite, positive, and the largest in the batch is 1
        # (Schaul et al., 2016's own per-batch normalization convention).
        assert np.all(np.isfinite(batch["is_weight"]))
        assert np.all(batch["is_weight"] > 0)
        assert np.isclose(batch["is_weight"].max(), 1.0)

    def test_update_priorities_biases_future_sampling(self) -> None:
        """Two single-step episodes; give one a much higher priority than
        the other and confirm it gets sampled far more often - the whole
        point of prioritized replay."""
        buffer = _EfficientZeroBuffer(capacity_episodes=10, obs_shape=(1,), action_dim=1, policy_target_dim=1, priority_alpha=1.0)
        buffer.add(obs=np.array([0.0]), action_flat=np.array([0.0]), reward=1.0, next_obs=np.array([1.0]), policy_target=np.array([1.0]), done=True)
        buffer.add(obs=np.array([100.0]), action_flat=np.array([0.0]), reward=1.0, next_obs=np.array([101.0]), policy_target=np.array([1.0]), done=True)
        assert buffer.num_episodes == 2

        # Both start at the same default priority -> roughly even split.
        batch = buffer.sample(batch_size=200, unroll_steps=1, td_steps=1, gamma=0.9)
        counts_before = np.bincount(batch["episode_idx"], minlength=2)
        assert counts_before[0] > 0 and counts_before[1] > 0

        # Crank episode 1's priority way up relative to episode 0's.
        buffer.update_priorities(np.array([1]), np.array([0]), np.array([1000.0]))
        buffer.update_priorities(np.array([0]), np.array([0]), np.array([1e-6]))

        batch2 = buffer.sample(batch_size=200, unroll_steps=1, td_steps=1, gamma=0.9)
        counts_after = np.bincount(batch2["episode_idx"], minlength=2)
        assert counts_after[1] > counts_after[0]
        assert counts_after[1] > 150  # heavily skewed towards episode 1

    def test_reanalyze_roundtrip_updates_stored_targets(self) -> None:
        """`sample_for_reanalyze` -> `update_reanalyzed_targets` must
        overwrite exactly the sampled transitions' stored policy_target/
        search_value, visible in a subsequent `sample()` call."""
        buffer = _EfficientZeroBuffer(capacity_episodes=10, obs_shape=(1,), action_dim=1, policy_target_dim=2)
        for t in range(4):
            buffer.add(
                obs=np.array([float(t)]), action_flat=np.array([0.0]), reward=1.0,
                next_obs=np.array([float(t + 1)]), policy_target=np.array([0.5, 0.5]), done=t == 3,
            )
        episode_idx, timestep, obs_out = buffer.sample_for_reanalyze(4)
        assert episode_idx.shape == (4,)
        assert obs_out.shape == (4, 1)
        # obs returned must actually match what's stored at that index.
        for i in range(4):
            assert np.isclose(obs_out[i, 0], float(timestep[i]))

        fresh_policy = np.stack([np.array([0.9, 0.1])] * 4)
        fresh_values = np.array([7.0, 8.0, 9.0, 10.0], dtype=np.float32)
        buffer.update_reanalyzed_targets(episode_idx, timestep, fresh_policy, fresh_values)

        # `sample_for_reanalyze` samples *with replacement*, so a timestep
        # can appear more than once in the batch - the last write for that
        # timestep (in call order) wins, same as any other last-write-wins
        # batched update. Build that expectation explicitly rather than
        # assuming each `timestep[i]` is unique.
        expected_value_by_t: dict[int, float] = {}
        for i, t in enumerate(timestep.tolist()):
            expected_value_by_t[t] = float(fresh_values[i])

        ep = buffer.episodes[0]
        for t, expected in expected_value_by_t.items():
            assert np.allclose(ep["policy_target"][t], [0.9, 0.1])
            assert np.isclose(ep["search_value"][t], expected)

    def test_sample_for_reanalyze_empty_buffer(self) -> None:
        buffer = _EfficientZeroBuffer(capacity_episodes=10, obs_shape=(1,), action_dim=1, policy_target_dim=1)
        episode_idx, timestep, obs_out = buffer.sample_for_reanalyze(8)
        assert episode_idx.shape == (0,)
        assert timestep.shape == (0,)
        assert obs_out.shape == (0, 1)


class TestNativeEfficientZeroDiscrete:
    def test_search_output_shapes(self) -> None:
        env = gym.make(_DISCRETE_ENV_ID)
        algo = NativeEfficientZero(env, {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE}, seed=0, device="cpu")
        obs, _info = env.reset(seed=0)

        obs_batch = obs_to_array(obs, env.observation_space)[None]
        results = algo.search(obs_batch, deterministic=np.array([False]))
        assert len(results) == 1
        result = results[0]
        assert result["policy_target"].shape == (env.action_space.n,)
        assert np.isclose(result["policy_target"].sum(), 1.0, atol=1e-4)
        assert 0 <= result["env_action"] < env.action_space.n
        env.close()

    def test_search_is_batched_across_lanes(self) -> None:
        env = gym.make(_DISCRETE_ENV_ID)
        algo = NativeEfficientZero(env, {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE}, seed=0, device="cpu")
        obs, _info = env.reset(seed=0)
        one = obs_to_array(obs, env.observation_space)
        results = algo.search(np.stack([one, one, one]), deterministic=np.array([True, True, True]))
        assert len(results) == 3
        for result in results:
            assert result["policy_target"].shape == (env.action_space.n,)
            assert 0 <= result["env_action"] < env.action_space.n
        env.close()

    def test_smoke(self) -> None:
        _run_smoke(_DISCRETE_ENV_ID, _TINY_DISCRETE, total_timesteps=40)

    def test_smoke_parallel_envs(self) -> None:
        _run_smoke(_DISCRETE_ENV_ID, _TINY_DISCRETE, total_timesteps=40, num_envs=3)


class TestNativeEfficientZeroContinuous:
    def test_search_output_shapes(self) -> None:
        env = gym.make(_CONTINUOUS_ENV_ID)
        algo = NativeEfficientZero(env, {**DEFAULT_HYPERPARAMS, **_TINY_CONTINUOUS}, seed=0, device="cpu")
        obs, _info = env.reset(seed=0)

        obs_batch = obs_to_array(obs, env.observation_space)[None]
        results = algo.search(obs_batch, deterministic=np.array([False]))
        assert len(results) == 1
        result = results[0]
        action_dim = int(np.prod(env.action_space.shape))
        assert result["policy_target"].shape == (action_dim,)
        assert result["env_action"].shape == (action_dim,)
        env.close()

    def test_smoke(self) -> None:
        _run_smoke(_CONTINUOUS_ENV_ID, _TINY_CONTINUOUS, total_timesteps=40)

    def test_smoke_parallel_envs(self) -> None:
        _run_smoke(_CONTINUOUS_ENV_ID, _TINY_CONTINUOUS, total_timesteps=40, num_envs=3)


class TestReanalyzeIntegration:
    """`_reanalyze()` end-to-end through `learn()` - a small enough
    `reanalyze_freq` that it actually triggers within the smoke run's
    `total_timesteps`, on both discrete and continuous action spaces (the
    reanalyzing `search()` call has to handle both)."""

    def test_reanalyze_triggers_and_refreshes_search_value(self) -> None:
        env = gym.make(_DISCRETE_ENV_ID)
        hp = {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE, "reanalyze_freq": 15, "reanalyze_batch_size": 4}
        algo = NativeEfficientZero(env, hp, seed=0, device="cpu")

        steps: list[int] = []

        def writer(num_timesteps, episode_reward=None, episode_length=None, metrics=None):
            steps.append(num_timesteps)
            return num_timesteps < 60

        algo.learn(total_timesteps=60, callback=TrainingCallback(writer))
        env.close()

        # At least one stored transition must have had its search_value
        # refreshed away from the "never reanalyzed" sentinel.
        any_reanalyzed = any(
            np.any(ep["search_value"] > -1e8) for ep in algo.buffer.episodes
        )
        assert any_reanalyzed

    def test_reanalyze_batch_size_zero_disables_it(self) -> None:
        """`reanalyze_batch_size=0` must be a clean no-op, not a crash -
        the buffer's `search_value` stays at the sentinel throughout."""
        env = gym.make(_DISCRETE_ENV_ID)
        hp = {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE, "reanalyze_freq": 5, "reanalyze_batch_size": 0}
        algo = NativeEfficientZero(env, hp, seed=0, device="cpu")

        def writer(num_timesteps, episode_reward=None, episode_length=None, metrics=None):
            return num_timesteps < 40

        algo.learn(total_timesteps=40, callback=TrainingCallback(writer))
        env.close()

        assert all(np.all(ep["search_value"] <= -1e8) for ep in algo.buffer.episodes)

    def test_reanalyze_continuous_action_space(self) -> None:
        env = gym.make(_CONTINUOUS_ENV_ID)
        hp = {**DEFAULT_HYPERPARAMS, **_TINY_CONTINUOUS, "reanalyze_freq": 15, "reanalyze_batch_size": 4}
        algo = NativeEfficientZero(env, hp, seed=0, device="cpu")

        def writer(num_timesteps, episode_reward=None, episode_length=None, metrics=None):
            return num_timesteps < 60

        algo.learn(total_timesteps=60, callback=TrainingCallback(writer))
        env.close()  # must not raise - continuous search() during reanalyze


class TestTrainFrequencyIndependentOfNumEnvs:
    """`num_timesteps` jumps by `n_envs` per iteration (all lanes step in
    lockstep). A naive "did we cross a train_freq boundary?" check fires at
    most once per iteration regardless of how many boundaries the jump
    actually spans, which silently divides the gradient-steps-per-env-step
    ratio by `n_envs` as num_envs grows - i.e. vectorizing environments would
    make training *less* sample-efficient purely as a side effect, not by
    design. `learn()` must instead run `train_steps_per_iter` updates for
    *each* train_freq boundary crossed in a jump, so the update-to-data
    ratio is independent of num_envs."""

    def _count_train_steps(self, num_envs: int, total_timesteps: int) -> int:
        env = (
            gym.make(_DISCRETE_ENV_ID)
            if num_envs == 1
            else make_env_or_vec(make_gym_env_factory(_DISCRETE_ENV_ID), num_envs=num_envs, parallel=False)
        )
        algo = NativeEfficientZero(
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
        # learning_starts=10 (from _TINY_DISCRETE) real steps before training
        # starts; use a large-enough total_timesteps that both configurations
        # get well past learning_starts regardless of n_envs granularity.
        total_timesteps = 200
        single = self._count_train_steps(num_envs=1, total_timesteps=total_timesteps)
        vectorized = self._count_train_steps(num_envs=4, total_timesteps=total_timesteps)

        # Roughly the same number of *post-learning_starts* real steps were
        # collected in both cases (bounded by total_timesteps), so with
        # train_freq=1/train_steps_per_iter=1 the gradient-step count should
        # be roughly equal too (a fixed ratio of ~1 grad step per real step),
        # not off by a factor of ~n_envs=4x in either direction.
        assert single > 0 and vectorized > 0
        ratio = vectorized / single
        assert 0.5 <= ratio <= 2.0, (
            f"expected train-step counts within 2x of each other regardless of "
            f"num_envs, got single={single} vectorized(num_envs=4)={vectorized} "
            f"ratio={ratio:.2f}"
        )
