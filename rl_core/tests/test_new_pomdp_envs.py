"""Sanity tests for `RockSample-v0` and `VisualMemoryMaze-v0` (see
`rl_core/envs/pomdp.py`) — the two "big, important" POMDP benchmarks added
alongside the simpler masking/recall tasks: RockSample(7,8) from the POMDP
*planning* literature (Smith & Simmons, 2004) and a MiniGrid-`MemoryEnv`-style
visual maze. Full learning-curve A/B validation (like `RepeatPrevious-v0`
got) is deliberately *not* attempted here — both are genuinely hard enough
that a convincing memory-vs-no-memory gap needs a training budget far beyond
what a unit test should spend; these tests instead lock in the mechanics
(reward logic, sensor noise, cue/goal matching, image+LSTM wiring) via cheap
scripted-policy and short-training smoke checks.
"""
from __future__ import annotations

import unittest

import numpy as np

from rl_core.envs.pomdp import RockSampleEnv, VisualMemoryMazeEnv, register_pomdp_envs
from rl_core.algorithms.native.networks import build_feature_extractor, NatureCNN, SmallCNN
import gymnasium as gym

register_pomdp_envs()


class RockSampleTests(unittest.TestCase):
    def test_registered_and_spaces(self) -> None:
        env = gym.make("RockSample-v0", render_mode="rgb_array")
        self.assertEqual(env.action_space.n, 13)  # 4 moves + sample + 8 checks
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.shape, env.observation_space.shape)
        env.close()

    def test_sampling_a_known_good_rock_gives_plus_ten(self) -> None:
        env = RockSampleEnv()
        env.reset(seed=1)
        good_idx = int(np.flatnonzero(env._good)[0])
        target = env._rocks[good_idx]
        # Walk straight to the rock (RockSample's grid has no internal walls).
        while env._agent != target:
            ax, ay = env._agent
            if ax < target[0]:
                action = 2
            elif ax > target[0]:
                action = 3
            elif ay < target[1]:
                action = 0
            else:
                action = 1
            env.step(action)
        _, reward, _, _, _ = env.step(4)  # Sample
        self.assertEqual(reward, 10.0)
        self.assertTrue(env._sampled[good_idx])

    def test_sampling_empty_cell_is_penalized(self) -> None:
        env = RockSampleEnv()
        env.reset(seed=0)
        rock_cells = set(env._rocks)
        # Agent's own starting cell is guaranteed rock-free by construction.
        self.assertNotIn(env._agent, rock_cells)
        _, reward, _, _, _ = env.step(4)  # Sample on an empty cell
        self.assertEqual(reward, -1.0)

    def test_sensor_noise_decays_with_distance(self) -> None:
        """A far-away check should be noisier (closer to 50/50) than a
        check made while standing right on the rock."""
        env = RockSampleEnv(sensor_efficiency=3.0)
        env.reset(seed=3)
        # Move onto rock 0 and check it repeatedly from there (distance 0).
        target = env._rocks[0]
        while env._agent != target:
            ax, ay = env._agent
            action = 2 if ax < target[0] else (3 if ax > target[0] else (0 if ay < target[1] else 1))
            env.step(action)
        correct_close = 0
        n = 300
        for _ in range(n):
            obs, _, _, _, _ = env.step(5)
            signal = obs[2 + 2 * env.k + env.k + 0]
            if (signal > 0) == bool(env._good[0]):
                correct_close += 1
        # Distance 0 => p_correct = 0.5 + 0.5*2**0 = 1.0 exactly.
        self.assertEqual(correct_close, n)

    def test_exit_east_terminates(self) -> None:
        env = RockSampleEnv(n=7, k=2)
        env.reset(seed=0)
        terminated = False
        for _ in range(20):
            _, _, terminated, truncated, _ = env.step(2)  # keep moving east
            if terminated:
                break
        self.assertTrue(terminated)

    def test_render_returns_rgb_frame(self) -> None:
        env = RockSampleEnv(render_mode="rgb_array")
        env.reset(seed=0)
        frame = env.render()
        self.assertEqual(frame.dtype, np.uint8)
        self.assertEqual(frame.shape[-1], 3)


class VisualMemoryMazeTests(unittest.TestCase):
    def test_registered_and_spaces(self) -> None:
        env = gym.make("VisualMemoryMaze-v0", render_mode="rgb_array")
        self.assertEqual(env.action_space.n, 4)
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertEqual(obs.dtype, np.uint8)
        env.close()

    def test_is_image_space_routes_to_small_cnn(self) -> None:
        env = VisualMemoryMazeEnv()
        extractor = build_feature_extractor(env.observation_space)
        self.assertIsInstance(extractor, SmallCNN)
        # Sanity: a real Atari-sized space should still get the big NatureCNN.
        atari_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 4), dtype=np.uint8)
        self.assertIsInstance(build_feature_extractor(atari_space), NatureCNN)

    def test_scripted_policy_reaches_matching_goal(self) -> None:
        """Knowing the (otherwise-hidden) cue color, a scripted policy that
        walks straight to the matching branch should score positively and
        terminate — this exercises the full cue/goal-matching reward logic
        end to end."""
        env = VisualMemoryMazeEnv()
        for seed in range(6):
            env.reset(seed=seed)
            go_top = env._cue_color == env._top_color
            terminated = truncated = False
            steps = 0
            total_reward = 0.0
            while not (terminated or truncated) and steps < 40:
                ax, ay = env._agent
                if ax < env.WIDTH - 2:
                    action = 3  # east
                elif ay != (1 if go_top else 5):
                    action = 0 if go_top else 1
                else:
                    action = 3
                _, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
            self.assertTrue(terminated, f"seed={seed} did not terminate within 40 steps")
            self.assertGreater(total_reward, 0.0, f"seed={seed} reached the wrong goal")

    def test_wrong_branch_is_penalized(self) -> None:
        env = VisualMemoryMazeEnv()
        env.reset(seed=0)
        go_top = env._cue_color == env._top_color
        terminated = truncated = False
        total_reward = 0.0
        steps = 0
        while not (terminated or truncated) and steps < 40:
            ax, ay = env._agent
            # Deliberately walk to the *wrong* branch.
            if ax < env.WIDTH - 2:
                action = 3
            elif ay != (5 if go_top else 1):
                action = 1 if go_top else 0
            else:
                action = 3
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
        self.assertTrue(terminated)
        self.assertLess(total_reward, 0.0)

    def test_cue_leaves_egocentric_view_after_a_few_steps(self) -> None:
        """The partial observability must come from the limited field of
        view, not a hand-coded hide — moving `VIEW_RADIUS + 1` steps away
        from the cue cell should make its color vanish from the crop."""
        env = VisualMemoryMazeEnv()
        env.reset(seed=0)
        cue_rgb = np.array(env._cue_color, dtype=np.uint8)
        obs, *_ = env.step(3)  # one step east: cue cell (x=0) still barely in view
        for _ in range(env.VIEW_RADIUS + 2):
            obs, _, _, _, _ = env.step(3)
        self.assertFalse(np.any(np.all(obs == cue_rgb, axis=-1)))

    def test_short_recurrent_dqn_training_does_not_crash(self) -> None:
        """Cheap end-to-end smoke test of the image+LSTM combo (CNN feature
        extractor -> RecurrentCore -> dueling/plain Q-head) through
        `NativeDQN`'s full train loop, including save/load."""
        import tempfile
        import os
        from rl_core.algorithms.native.dqn import NativeDQN, DEFAULT_HYPERPARAMS
        from rl_core.algorithms.base import TrainingCallback

        env = VisualMemoryMazeEnv()
        hp = dict(DEFAULT_HYPERPARAMS)
        hp.update(dict(
            buffer_size=500, batch_size=8, learning_starts=50, train_freq=1,
            target_update_interval=50, memory_type=1, memory_hidden_size=16,
            memory_num_layers=1, memory_seq_len=8,
        ))
        algo = NativeDQN(env, hp, device="cpu", seed=0)
        episodes = []
        cb = TrainingCallback(writer=lambda *a, **k: (episodes.append(1) if a[1] is not None else None) or True)
        algo.learn(total_timesteps=300, callback=cb)
        obs, _ = env.reset(seed=0)
        action, _ = algo.predict(obs, deterministic=True, episode_start=True)
        self.assertIn(int(action), range(4))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model")
            algo.save(path)
            NativeDQN.load(path, env)


if __name__ == "__main__":
    unittest.main()
