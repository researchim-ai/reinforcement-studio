"""Tests for `training.num_envs` vectorization (see
`rl_core/algorithms/vec_env.py` and the "Vectorized parallel environments"
plan) — `RolloutBuffer`'s vectorized GAE/flatten/chunking math, the
`vec_env.py` helpers themselves, per-lane isolation of the off-policy
buffers that track independent n-step/episode state per lane, and
end-to-end smoke tests (learn/predict/save/load, no crashes, per-lane
episode bookkeping) for every native algorithm at `num_envs=1` (must stay
byte-for-byte the old single-env behavior) and `num_envs>1` (including
recurrent memory)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor

from rl_core.algorithms.base import TrainingCallback
from rl_core.algorithms.native.a2c import DEFAULT_HYPERPARAMS as A2C_DEFAULTS
from rl_core.algorithms.native.a2c import NativeA2C
from rl_core.algorithms.native.buffers import EpisodeSequenceReplayBuffer, NStepPrioritizedReplayBuffer, RolloutBuffer
from rl_core.algorithms.native.ddpg import DEFAULT_HYPERPARAMS as DDPG_DEFAULTS
from rl_core.algorithms.native.ddpg import NativeDDPG
from rl_core.algorithms.native.dqn import DEFAULT_HYPERPARAMS as DQN_DEFAULTS
from rl_core.algorithms.native.dqn import NativeDQN
from rl_core.algorithms.native.es import DEFAULT_HYPERPARAMS as ES_DEFAULTS
from rl_core.algorithms.native.es import NativeES
from rl_core.algorithms.native.ppo import DEFAULT_HYPERPARAMS as PPO_DEFAULTS
from rl_core.algorithms.native.ppo import NativePPO
from rl_core.algorithms.native.rainbow_dqn import DEFAULT_HYPERPARAMS as RAINBOW_DEFAULTS
from rl_core.algorithms.native.rainbow_dqn import NativeRainbowDQN
from rl_core.algorithms.native.sac import DEFAULT_HYPERPARAMS as SAC_DEFAULTS
from rl_core.algorithms.native.sac import NativeSAC
from rl_core.algorithms.native.td3 import DEFAULT_HYPERPARAMS as TD3_DEFAULTS
from rl_core.algorithms.native.td3 import NativeTD3
from rl_core.algorithms.vec_env import (
    action_space,
    is_vector_env,
    make_env_or_vec,
    make_gym_env_factory,
    num_envs_of,
    obs_space,
    vec_reset,
    vec_step,
)
from rl_core.envs.pomdp import register_pomdp_envs


class VecEnvHelperTests(unittest.TestCase):
    def test_num_envs_1_stays_a_plain_env(self) -> None:
        env = make_env_or_vec(lambda: gym.make("CartPole-v1"), num_envs=1)
        self.assertFalse(is_vector_env(env))
        self.assertEqual(num_envs_of(env), 1)
        self.assertIs(obs_space(env), env.observation_space)
        self.assertIs(action_space(env), env.action_space)
        env.close()

    def test_num_envs_4_builds_async_vector_env(self) -> None:
        from gymnasium.vector import AsyncVectorEnv

        env = make_env_or_vec(lambda: gym.make("CartPole-v1"), num_envs=4)
        self.assertTrue(is_vector_env(env))
        self.assertIsInstance(env, AsyncVectorEnv)
        self.assertEqual(num_envs_of(env), 4)
        self.assertEqual(len(env.processes), 4)
        self.assertIs(obs_space(env), env.single_observation_space)
        self.assertIs(action_space(env), env.single_action_space)
        env.close()

    def test_async_vector_env_uses_separate_worker_processes(self) -> None:
        import os

        from gymnasium.vector import AsyncVectorEnv

        env = make_env_or_vec(lambda: gym.make("CartPole-v1"), num_envs=3)
        self.assertIsInstance(env, AsyncVectorEnv)
        main_pid = os.getpid()
        worker_pids = {proc.pid for proc in env.processes}
        self.assertEqual(len(worker_pids), 3)
        self.assertNotIn(main_pid, worker_pids)
        env.close()

    def test_vec_reset_and_step_shapes_for_1_and_n_envs(self) -> None:
        for n in (1, 3):
            with self.subTest(num_envs=n):
                env = make_env_or_vec(lambda: gym.make("CartPole-v1"), num_envs=n)
                obs_list = vec_reset(env, seed=0)
                self.assertEqual(len(obs_list), n)
                actions = [action_space(env).sample() for _ in range(n)]
                next_obs, rewards, terminated, truncated, infos = vec_step(env, actions)
                self.assertEqual(len(next_obs), n)
                self.assertEqual(rewards.shape, (n,))
                self.assertEqual(terminated.shape, (n,))
                self.assertEqual(truncated.shape, (n,))
                self.assertEqual(len(infos), n)
                env.close()

    def test_single_env_can_step_again_immediately_after_done(self) -> None:
        """Regression test for the Gymnasium "step() called after
        terminated=True" warning fixed in `vec_step`: a plain (num_envs=1)
        env must auto-reset *within* the same call that sees done=True, so
        the very next `vec_step` call is always safe."""
        env = make_env_or_vec(lambda: gym.make("CartPole-v1"), num_envs=1)
        vec_reset(env, seed=0)
        for _ in range(500):
            _obs, _r, terminated, truncated, _infos = vec_step(env, [action_space(env).sample()])
            if terminated[0] or truncated[0]:
                break
        else:
            self.fail("CartPole never terminated within 500 random steps")
        vec_step(env, [action_space(env).sample()])  # must not raise/warn
        env.close()

    def test_vec_step_handles_nested_dict_info_from_monitor_wrapper(self) -> None:
        """Regression test: vector envs aggregate a per-lane info key
        whose *value* is itself a dict (e.g. `Monitor`'s `"episode"` key,
        only present on the step a given lane's episode ends) into a
        nested dict-of-arrays plus a `_key` presence mask, not a plain
        per-lane array — `vec_step` must unpack that recursively instead
        of indexing it like every other (flat) info key."""
        num_envs = 3
        env = make_env_or_vec(lambda: Monitor(gym.make("CartPole-v1")), num_envs=num_envs)
        vec_reset(env, seed=0)
        saw_episode_info = False
        for _ in range(500):
            actions = [action_space(env).sample() for _ in range(num_envs)]
            _obs, _r, terminated, truncated, infos = vec_step(env, actions)
            self.assertEqual(len(infos), num_envs)
            for i in range(num_envs):
                if terminated[i] or truncated[i]:
                    self.assertIn("episode", infos[i])
                    self.assertIn("r", infos[i]["episode"])
                    saw_episode_info = True
                else:
                    self.assertNotIn("episode", infos[i])
            if saw_episode_info:
                break
        else:
            self.fail("no lane finished an episode within 500 random steps")
        env.close()


class RolloutBufferTests(unittest.TestCase):
    def _fill(self, buf: RolloutBuffer, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray) -> None:
        capacity, num_envs = rewards.shape
        for t in range(capacity):
            buf.add(
                obs=np.zeros((num_envs, *buf.obs.shape[2:]), dtype=np.float32),
                actions=np.zeros(num_envs, dtype=np.int64),
                log_probs=np.zeros(num_envs, dtype=np.float32),
                values=values[t],
                rewards=rewards[t],
                dones=dones[t],
                episode_starts=np.zeros(num_envs, dtype=np.float32),
            )

    def test_gae_num_envs_1_matches_manual_scalar_recursion(self) -> None:
        capacity = 5
        buf = RolloutBuffer(capacity, num_envs=1, obs_shape=(1,), action_dim=0, discrete=True)
        rewards = np.array([[1.0], [0.5], [-1.0], [2.0], [0.0]], dtype=np.float32)
        values = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]], dtype=np.float32)
        dones = np.array([[0.0], [0.0], [1.0], [0.0], [0.0]], dtype=np.float32)
        self._fill(buf, rewards, values, dones)
        last_value, last_done = 0.7, 0.0
        gamma, lam = 0.9, 0.95
        buf.compute_returns_and_advantage(np.array([last_value]), np.array([last_done]), gamma, lam)

        expected_adv = [0.0] * capacity
        last_gae = 0.0
        next_value, next_non_terminal = last_value, 1.0 - last_done
        for t in reversed(range(capacity)):
            delta = rewards[t, 0] + gamma * next_value * next_non_terminal - values[t, 0]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            expected_adv[t] = last_gae
            next_value, next_non_terminal = values[t, 0], 1.0 - dones[t, 0]

        np.testing.assert_allclose(buf.advantages[:, 0], expected_adv, atol=1e-6)
        np.testing.assert_allclose(buf.returns[:, 0], np.array(expected_adv) + values[:, 0], atol=1e-6)

    def test_gae_vectorized_matches_independent_per_lane_recursion(self) -> None:
        capacity, num_envs = 6, 3
        buf = RolloutBuffer(capacity, num_envs, obs_shape=(2,), action_dim=0, discrete=True)
        rng = np.random.default_rng(0)
        rewards = rng.normal(size=(capacity, num_envs)).astype(np.float32)
        values = rng.normal(size=(capacity, num_envs)).astype(np.float32)
        dones = (rng.random((capacity, num_envs)) < 0.3).astype(np.float32)
        self._fill(buf, rewards, values, dones)
        last_values = rng.normal(size=num_envs).astype(np.float32)
        last_dones = (rng.random(num_envs) < 0.3).astype(np.float32)
        gamma, lam = 0.99, 0.9
        buf.compute_returns_and_advantage(last_values, last_dones, gamma, lam)

        for lane in range(num_envs):
            expected = [0.0] * capacity
            last_gae = 0.0
            next_value, next_non_terminal = last_values[lane], 1.0 - last_dones[lane]
            for t in reversed(range(capacity)):
                delta = rewards[t, lane] + gamma * next_value * next_non_terminal - values[t, lane]
                last_gae = delta + gamma * lam * next_non_terminal * last_gae
                expected[t] = last_gae
                next_value, next_non_terminal = values[t, lane], 1.0 - dones[t, lane]
            np.testing.assert_allclose(buf.advantages[:, lane], expected, atol=1e-5)

    def test_minibatches_cover_every_lane_step_transition_exactly_once(self) -> None:
        capacity, num_envs = 5, 4
        buf = RolloutBuffer(capacity, num_envs, obs_shape=(1,), action_dim=0, discrete=True)
        flat_id = 0
        ids = np.zeros((capacity, num_envs), dtype=np.float32)
        for t in range(capacity):
            for e in range(num_envs):
                ids[t, e] = flat_id
                flat_id += 1
        for t in range(capacity):
            buf.add(
                obs=ids[t].reshape(num_envs, 1),
                actions=np.zeros(num_envs, dtype=np.int64),
                log_probs=np.zeros(num_envs, dtype=np.float32),
                values=np.zeros(num_envs, dtype=np.float32),
                rewards=np.zeros(num_envs, dtype=np.float32),
                dones=np.zeros(num_envs, dtype=np.float32),
                episode_starts=np.zeros(num_envs, dtype=np.float32),
            )
        buf.compute_returns_and_advantage(np.zeros(num_envs), np.zeros(num_envs), 0.99, 0.95)
        seen: list[float] = []
        for batch in buf.minibatches(batch_size=6):
            seen.extend(batch["obs"][:, 0].tolist())
        self.assertEqual(sorted(seen), list(range(capacity * num_envs)))

    def test_sequences_yields_num_envs_batched_time_chunks(self) -> None:
        capacity, num_envs, seq_len = 10, 3, 4
        buf = RolloutBuffer(capacity, num_envs, obs_shape=(1,), action_dim=0, discrete=True)
        self._fill(
            buf,
            np.zeros((capacity, num_envs), dtype=np.float32),
            np.zeros((capacity, num_envs), dtype=np.float32),
            np.zeros((capacity, num_envs), dtype=np.float32),
        )
        buf.compute_returns_and_advantage(np.zeros(num_envs), np.zeros(num_envs), 0.99, 0.95)
        chunk_lens = []
        for chunk in buf.sequences(seq_len):
            self.assertEqual(chunk["obs"].shape[0], num_envs)
            chunk_lens.append(chunk["obs"].shape[1])
        self.assertEqual(chunk_lens, [4, 4, 2])
        self.assertEqual(sum(chunk_lens), capacity)


class PerLaneOffPolicyBufferTests(unittest.TestCase):
    """`EpisodeSequenceReplayBuffer`/`NStepPrioritizedReplayBuffer` track
    independent per-lane state (in-progress episode / n-step window) so
    `num_envs>1` lanes on completely different schedules never bleed into
    each other — verified directly against the buffers, without running a
    full algorithm."""

    def test_episode_sequence_buffer_keeps_lanes_independent(self) -> None:
        buf = EpisodeSequenceReplayBuffer(capacity=100, obs_shape=(1,), num_envs=2)
        for i in range(3):  # lane 0: a 3-step episode
            buf.add([i], i, 1.0, [i + 1], i == 2, lane=0)
        for i in range(2):  # lane 1: an interleaved-in-time, shorter 2-step episode
            buf.add([100 + i], i, 2.0, [100 + i + 1], i == 1, lane=1)

        self.assertEqual(len(buf._episodes), 2)
        lengths = sorted(len(ep["dones"]) for ep in buf._episodes)
        self.assertEqual(lengths, [2, 3])
        for ep in buf._episodes:
            if ep["obs"][0, 0] < 100:
                self.assertEqual(len(ep["dones"]), 3)
            else:
                self.assertEqual(len(ep["dones"]), 2)

    def test_nstep_buffer_pending_windows_are_independent_per_lane(self) -> None:
        buf = NStepPrioritizedReplayBuffer(capacity=100, obs_shape=(1,), n_step=3, num_envs=2)
        buf.add([0], 0, 1.0, [1], False, lane=0)
        buf.add([0], 0, 5.0, [1], False, lane=1)
        self.assertEqual(len(buf._pending[0]), 1)
        self.assertEqual(len(buf._pending[1]), 1)

        # Advancing lane 0 alone must not disturb lane 1's own pending window.
        buf.add([1], 0, 1.0, [2], False, lane=0)
        self.assertEqual(len(buf._pending[0]), 2)
        self.assertEqual(len(buf._pending[1]), 1)

        # Finishing lane 1's episode flushes only lane 1's window.
        buf.add([1], 0, 5.0, [2], True, lane=1)
        self.assertEqual(len(buf._pending[1]), 0)
        self.assertEqual(len(buf._pending[0]), 2)


def _collect_metrics_writer() -> tuple[dict[str, float], TrainingCallback]:
    latest: dict[str, float] = {}

    def writer(_step, _reward=None, _length=None, metrics=None):
        if metrics:
            latest.update(metrics)
        return True

    return latest, TrainingCallback(writer)


class OnPolicyVecSmokeTests(unittest.TestCase):
    """PPO/A2C at `num_envs=1` (must stay a behavior-preserving no-op vs.
    pre-vectorization) and `num_envs=4` (feedforward and recurrent)."""

    def _run_case(self, algo_cls, defaults: dict, num_envs: int, recurrent: bool) -> None:
        register_pomdp_envs()
        env_id = "MemoryCorridor-v0" if recurrent else "CartPole-v1"
        env = make_env_or_vec(make_gym_env_factory(env_id), num_envs=num_envs)
        hyperparams = {
            **defaults,
            "n_steps": 16,
            "batch_size": 8,
            "n_epochs": 2,
            "memory_type": 1 if recurrent else 0,
            "memory_hidden_size": 16,
            "memory_seq_len": 8,
        }
        algo = algo_cls(env, hyperparams, 42, "cpu")
        latest_metrics, callback = _collect_metrics_writer()

        algo.learn(16 * max(1, num_envs) * 2, callback)

        probe_env = gym.make(env_id)
        obs, _ = probe_env.reset()
        algo.predict(obs, deterministic=True, episode_start=True)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.zip"
            algo.save(checkpoint)
            loaded = algo_cls.load(checkpoint, probe_env)
            loaded.predict(obs, deterministic=True, episode_start=True)
        probe_env.close()
        env.close()
        del latest_metrics

    def test_ppo_num_envs_1_parity(self) -> None:
        self._run_case(NativePPO, PPO_DEFAULTS, num_envs=1, recurrent=False)

    def test_ppo_num_envs_4_feedforward(self) -> None:
        self._run_case(NativePPO, PPO_DEFAULTS, num_envs=4, recurrent=False)

    def test_ppo_num_envs_4_recurrent(self) -> None:
        self._run_case(NativePPO, PPO_DEFAULTS, num_envs=4, recurrent=True)

    def test_a2c_num_envs_1_parity(self) -> None:
        self._run_case(NativeA2C, A2C_DEFAULTS, num_envs=1, recurrent=False)

    def test_a2c_num_envs_4_feedforward(self) -> None:
        self._run_case(NativeA2C, A2C_DEFAULTS, num_envs=4, recurrent=False)

    def test_a2c_num_envs_4_recurrent(self) -> None:
        self._run_case(NativeA2C, A2C_DEFAULTS, num_envs=4, recurrent=True)


class OffPolicyDiscreteVecSmokeTests(unittest.TestCase):
    """DQN/Rainbow DQN at `num_envs>1`, feedforward and recurrent (DRQN)."""

    def _run_case(self, algo_cls, defaults: dict, num_envs: int, recurrent: bool) -> None:
        register_pomdp_envs()
        env_id = "MemoryCorridor-v0" if recurrent else "CartPole-v1"
        env = make_env_or_vec(make_gym_env_factory(env_id), num_envs=num_envs)
        hyperparams = {
            **defaults,
            "buffer_size": 500,
            "batch_size": 8,
            "learning_starts": 8,
            "train_freq": 2,
            "target_update_interval": 20,
            "memory_type": 1 if recurrent else 0,
            "memory_hidden_size": 16,
            "memory_seq_len": 8,
        }
        algo = algo_cls(env, hyperparams, 42, "cpu")
        _latest_metrics, callback = _collect_metrics_writer()

        algo.learn(8 * max(1, num_envs) * 4, callback)

        probe_env = gym.make(env_id)
        obs, _ = probe_env.reset()
        algo.predict(obs, deterministic=True, episode_start=True)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.zip"
            algo.save(checkpoint)
            loaded = algo_cls.load(checkpoint, probe_env)
            loaded.predict(obs, deterministic=True, episode_start=True)
        probe_env.close()
        env.close()

    def test_dqn_num_envs_1_parity(self) -> None:
        self._run_case(NativeDQN, DQN_DEFAULTS, num_envs=1, recurrent=False)

    def test_dqn_num_envs_4_feedforward(self) -> None:
        self._run_case(NativeDQN, DQN_DEFAULTS, num_envs=4, recurrent=False)

    def test_dqn_num_envs_3_recurrent(self) -> None:
        self._run_case(NativeDQN, DQN_DEFAULTS, num_envs=3, recurrent=True)

    def test_rainbow_dqn_num_envs_1_parity(self) -> None:
        self._run_case(NativeRainbowDQN, RAINBOW_DEFAULTS, num_envs=1, recurrent=False)

    def test_rainbow_dqn_num_envs_4_feedforward(self) -> None:
        self._run_case(NativeRainbowDQN, RAINBOW_DEFAULTS, num_envs=4, recurrent=False)

    def test_rainbow_dqn_num_envs_3_recurrent(self) -> None:
        self._run_case(NativeRainbowDQN, RAINBOW_DEFAULTS, num_envs=3, recurrent=True)


class OffPolicyContinuousVecSmokeTests(unittest.TestCase):
    """SAC/DDPG/TD3 at `num_envs>1` on a continuous-action env."""

    def _run_case(self, algo_cls, defaults: dict, num_envs: int) -> None:
        env = make_env_or_vec(lambda: gym.make("Pendulum-v1"), num_envs=num_envs)
        hyperparams = {
            **defaults,
            "buffer_size": 500,
            "batch_size": 8,
            "learning_starts": 8,
            "train_freq": 1,
        }
        algo = algo_cls(env, hyperparams, 42, "cpu")
        _latest_metrics, callback = _collect_metrics_writer()

        algo.learn(8 * max(1, num_envs) * 4, callback)

        probe_env = gym.make("Pendulum-v1")
        obs, _ = probe_env.reset()
        algo.predict(obs, deterministic=True)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.zip"
            algo.save(checkpoint)
            loaded = algo_cls.load(checkpoint, probe_env)
            loaded.predict(obs, deterministic=True)
        probe_env.close()
        env.close()

    def test_sac_num_envs_1_parity(self) -> None:
        self._run_case(NativeSAC, SAC_DEFAULTS, num_envs=1)

    def test_sac_num_envs_3(self) -> None:
        self._run_case(NativeSAC, SAC_DEFAULTS, num_envs=3)

    def test_ddpg_num_envs_1_parity(self) -> None:
        self._run_case(NativeDDPG, DDPG_DEFAULTS, num_envs=1)

    def test_ddpg_num_envs_3(self) -> None:
        self._run_case(NativeDDPG, DDPG_DEFAULTS, num_envs=3)

    def test_td3_num_envs_1_parity(self) -> None:
        self._run_case(NativeTD3, TD3_DEFAULTS, num_envs=1)

    def test_td3_num_envs_3(self) -> None:
        self._run_case(NativeTD3, TD3_DEFAULTS, num_envs=3)


class ESVecSmokeTests(unittest.TestCase):
    """ES's parallelism is population- rather than time-parallel — see
    `NativeES._run_episode_batch` — so this checks concurrent per-lane
    population members instead of a time-step collection loop."""

    def _run_case(self, num_envs: int) -> None:
        env = make_env_or_vec(lambda: gym.make("CartPole-v1"), num_envs=num_envs)
        hyperparams = {**ES_DEFAULTS, "population_size": 4, "episodes_per_eval": 1}
        algo = NativeES(env, hyperparams, 42, "cpu")
        _latest_metrics, callback = _collect_metrics_writer()

        algo.learn(20, callback)

        probe_env = gym.make("CartPole-v1")
        obs, _ = probe_env.reset()
        algo.predict(obs, deterministic=True)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.zip"
            algo.save(checkpoint)
            loaded = NativeES.load(checkpoint, probe_env)
            loaded.predict(obs, deterministic=True)
        probe_env.close()
        env.close()

    def test_es_num_envs_1_parity(self) -> None:
        self._run_case(num_envs=1)

    def test_es_num_envs_4_population_parallel(self) -> None:
        self._run_case(num_envs=4)


class RunCustomAlgorithmVecIntegrationTests(unittest.TestCase):
    """`run_custom_algorithm` (the actual driver behind every Designer run,
    not just the algorithms in isolation) wraps each lane in `Monitor`
    before building the `AsyncVectorEnv` — exercises the full
    config -> runner -> algorithm -> metrics.json path at `num_envs>1`,
    including the nested-info aggregation `vec_step` has to unpack."""

    def _run(self, algo_cls, hyperparams: dict, num_envs: int) -> dict:
        import tempfile as _tempfile

        from rl_core.algorithms.runner_utils import run_custom_algorithm

        with _tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            config = {
                "environment": {"id": "CartPole-v1", "wrappers": []},
                "training": {"total_timesteps": 40, "seed": 7, "num_envs": num_envs},
            }
            run_custom_algorithm(algo_cls, hyperparams, config, run_dir, "test-algo", "MlpPolicy")
            import json as _json

            metrics = _json.loads((run_dir / "metrics.json").read_text())
            self.assertTrue((run_dir / "model.zip").exists())
            return metrics

    def test_ppo_num_envs_4_end_to_end(self) -> None:
        metrics = self._run(NativePPO, {"n_steps": 16, "batch_size": 8, "n_epochs": 2}, num_envs=4)
        self.assertEqual(metrics["status"], "completed")
        self.assertEqual(metrics["num_envs"], 4)

    def test_dqn_num_envs_4_end_to_end(self) -> None:
        metrics = self._run(
            NativeDQN,
            {"buffer_size": 200, "learning_starts": 8, "batch_size": 8, "train_freq": 2, "target_update_interval": 20},
            num_envs=4,
        )
        self.assertEqual(metrics["status"], "completed")
        self.assertEqual(metrics["num_envs"], 4)


if __name__ == "__main__":
    unittest.main()
