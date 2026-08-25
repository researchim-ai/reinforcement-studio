from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor

from rl_core.algorithms.base import TrainingCallback
from rl_core.algorithms.native.dqn import DEFAULT_HYPERPARAMS as DQN_DEFAULTS
from rl_core.algorithms.native.dqn import NativeDQN
from rl_core.algorithms.native.exploration import NoisyLinear, RNDModule, linear_schedule
from rl_core.algorithms.native.rainbow_dqn import DEFAULT_HYPERPARAMS as RAINBOW_DEFAULTS
from rl_core.algorithms.native.rainbow_dqn import NativeRainbowDQN
from rl_core.envs.pomdp import register_pomdp_envs


class ScheduleTests(unittest.TestCase):
    def test_linear_epsilon_schedule_boundaries(self) -> None:
        self.assertAlmostEqual(linear_schedule(0, 1_000, 0.2, 1.0, 0.05), 1.0)
        self.assertAlmostEqual(linear_schedule(200, 1_000, 0.2, 1.0, 0.05), 0.05)
        self.assertAlmostEqual(linear_schedule(1_000, 1_000, 0.2, 1.0, 0.05), 0.05)
        self.assertAlmostEqual(linear_schedule(0, 1_000, 0.0, 1.0, 0.05), 0.05)

    def test_schedule_is_monotonic(self) -> None:
        values = [linear_schedule(step, 100, 0.5, 1.0, 0.1) for step in range(101)]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))

    def test_training_callback_remains_backward_compatible(self) -> None:
        calls: list[int] = []

        def legacy_writer(step, _reward=None, _length=None):
            calls.append(step)
            return True

        callback = TrainingCallback(legacy_writer)
        self.assertTrue(callback.on_step(7, metrics={"exploration_epsilon": 0.5}))
        self.assertEqual(calls, [7])


class NoisyNetTests(unittest.TestCase):
    def test_noise_changes_training_forward_but_not_eval(self) -> None:
        torch.manual_seed(7)
        layer = NoisyLinear(4, 3)
        x = torch.ones(2, 4)
        before = layer(x).detach().clone()
        layer.reset_noise()
        after = layer(x).detach().clone()
        self.assertFalse(torch.equal(before, after))

        layer.eval()
        deterministic_before = layer(x).detach().clone()
        layer.reset_noise()
        deterministic_after = layer(x).detach().clone()
        self.assertTrue(torch.equal(deterministic_before, deterministic_after))


class RNDTests(unittest.TestCase):
    def test_target_is_frozen_and_novelty_decreases(self) -> None:
        torch.manual_seed(3)
        rnd = RNDModule(
            gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            "cpu",
            feature_dim=8,
            hidden_dim=16,
            learning_rate=1e-3,
        )
        self.assertTrue(all(not parameter.requires_grad for parameter in rnd.target.parameters()))
        observations = np.repeat(np.asarray([[0.25, -0.5]], dtype=np.float32), 32, axis=0)
        before = rnd.bonus(observations[0])
        for _ in range(100):
            rnd.update_predictor(observations)
        after = rnd.bonus(observations[0], update_stats=False)
        self.assertLess(after, before)


class NativeExplorationSmokeTests(unittest.TestCase):
    def _run_case(self, algo_cls, defaults: dict, noisy: int, rnd: int, memory: int) -> None:
        register_pomdp_envs()
        # Recurrent replay only exposes completed episodes to training.
        # MemoryCorridor terminates deterministically after eight steps,
        # avoiding a flaky "first CartPole episode outlived the smoke run".
        env_id = "MemoryCorridor-v0" if memory else "CartPole-v1"
        env = Monitor(gym.make(env_id))
        hyperparams = {
            **defaults,
            "action_exploration": noisy,
            "intrinsic_exploration": rnd,
            "memory_type": memory,
            "memory_hidden_size": 16,
            "memory_seq_len": 8,
            "learning_starts": 8,
            "batch_size": 4,
            "buffer_size": 200,
            "train_freq": 2,
            "target_update_interval": 20,
            "rnd_feature_dim": 8,
            "rnd_hidden_dim": 16,
        }
        algo = algo_cls(env, hyperparams, 42, "cpu")
        latest_metrics: dict[str, float] = {}

        def writer(_step, _reward=None, _length=None, metrics=None):
            if metrics:
                latest_metrics.update(metrics)
            return True

        algo.learn(64, TrainingCallback(writer))
        obs, _ = env.reset()
        algo.predict(obs, deterministic=True, episode_start=True)
        self.assertIn("exploration_epsilon", latest_metrics)
        if rnd:
            self.assertGreater(latest_metrics["rnd_predictor_loss"], 0.0)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.zip"
            algo.save(checkpoint)
            loaded = algo_cls.load(checkpoint, env)
            loaded.predict(obs, deterministic=True, episode_start=True)
            if rnd:
                self.assertIsNotNone(loaded.rnd)
                self.assertGreater(loaded.rnd.obs_rms.count, 1.0)
        env.close()

    def test_dqn_exploration_matrix(self) -> None:
        for noisy, rnd, memory in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 1)):
            with self.subTest(noisy=noisy, rnd=rnd, memory=memory):
                self._run_case(NativeDQN, DQN_DEFAULTS, noisy, rnd, memory)

    def test_rainbow_exploration_matrix(self) -> None:
        for noisy, rnd, memory in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 1)):
            with self.subTest(noisy=noisy, rnd=rnd, memory=memory):
                self._run_case(NativeRainbowDQN, RAINBOW_DEFAULTS, noisy, rnd, memory)


if __name__ == "__main__":
    unittest.main()
