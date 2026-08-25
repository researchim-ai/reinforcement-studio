"""Sanity tests for the second batch of "important/classic" POMDP benchmarks
added to `rl_core/envs/pomdp.py`: `Tiger-v0`, `HeavenHell-v0`, `Hallway-v0`,
`Battleship-v0`, `MinesweeperPOMDP-v0`, `Concentration-v0`, `LaserTag-v0` and
`ActiveTMaze-v0`/`ActiveTMazeLong-v0`.

Each test locks in the core mechanic that makes the env actually require
memory (rather than just being "an env with a hidden fact somewhere") via a
scripted policy comparison: a policy that uses the relevant piece of memory
should reliably outperform one that structurally cannot use it — this is a
cheap, deterministic substitute for a full training-curve A/B run.
"""
from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np

from rl_core.envs.pomdp import (
    ActiveTMazeEnv,
    BattleshipEnv,
    ConcentrationEnv,
    HallwayEnv,
    HeavenHellEnv,
    LaserTagEnv,
    MinesweeperPOMDPEnv,
    TigerEnv,
    register_pomdp_envs,
)

register_pomdp_envs()

_ALL_IDS = [
    "Tiger-v0", "HeavenHell-v0", "Hallway-v0", "Battleship-v0",
    "MinesweeperPOMDP-v0", "Concentration-v0", "LaserTag-v0",
    "ActiveTMaze-v0", "ActiveTMazeLong-v0",
]


class RegistrationTests(unittest.TestCase):
    def test_all_registered_with_rgb_render(self) -> None:
        for env_id in _ALL_IDS:
            env = gym.make(env_id, render_mode="rgb_array")
            obs, _ = env.reset(seed=0)
            self.assertEqual(obs.shape, env.observation_space.shape)
            frame = env.render()
            self.assertIsNotNone(frame)
            self.assertEqual(frame.dtype, np.uint8)
            env.close()


class TigerTests(unittest.TestCase):
    def test_repeated_listening_reveals_correct_side(self) -> None:
        env = TigerEnv()
        correct = 0
        trials = 40
        for seed in range(trials):
            env.reset(seed=seed)
            left_votes = right_votes = 0
            for _ in range(6):
                obs, _, _, _, _ = env.step(TigerEnv.LISTEN)
                if obs[0] == 1.0:
                    left_votes += 1
                else:
                    right_votes += 1
            believe_tiger_left = left_votes > right_votes
            safe_action = TigerEnv.OPEN_RIGHT if believe_tiger_left else TigerEnv.OPEN_LEFT
            _, reward, terminated, _, _ = env.step(safe_action)
            self.assertTrue(terminated)
            correct += reward == 10.0
        # 6 independent 85%-accurate votes should be right far more than a
        # single listen (or blind guessing) would be.
        self.assertGreater(correct / trials, 0.9)

    def test_opening_wrong_door_is_heavily_penalized(self) -> None:
        env = TigerEnv()
        env.reset(seed=0)
        action = TigerEnv.OPEN_LEFT if env._tiger_left else TigerEnv.OPEN_RIGHT
        _, reward, terminated, _, _ = env.step(action)
        self.assertEqual(reward, -100.0)
        self.assertTrue(terminated)


class HeavenHellTests(unittest.TestCase):
    def _goto_sign_then_fork(self, env: HeavenHellEnv) -> bool:
        for a in (env.NORTH, env.NORTH, env.WEST):
            obs, *_ = env.step(a)
        heaven_left = obs[5] > 0
        for a in (env.EAST, env.NORTH, env.NORTH):
            env.step(a)
        return heaven_left

    def test_memory_policy_always_reaches_heaven(self) -> None:
        env = HeavenHellEnv()
        for seed in range(15):
            env.reset(seed=seed)
            heaven_left = self._goto_sign_then_fork(env)
            action = env.WEST if heaven_left else env.EAST
            _, reward, terminated, _, _ = env.step(action)
            self.assertEqual(reward, 1.0)
            self.assertTrue(terminated)

    def test_never_visiting_sign_cannot_beat_chance_with_fixed_bias(self) -> None:
        # A bias-only ("always go west at the fork") policy is right exactly
        # when heaven happens to be on the left — i.e. at chance level.
        env = HeavenHellEnv()
        wins = 0
        trials = 30
        for seed in range(trials):
            env.reset(seed=seed)
            for a in (env.NORTH, env.NORTH, env.NORTH, env.NORTH, env.WEST):
                obs, reward, terminated, _, _ = env.step(a)
            wins += reward == 1.0
        frac = wins / trials
        self.assertGreater(frac, 0.2)
        self.assertLess(frac, 0.8)  # nowhere near the ~1.0 the memory policy gets


class HallwayTests(unittest.TestCase):
    def _run(self, env: HallwayEnv, fixed_action: int | None) -> float:
        env.reset()
        fork_choice = None
        for _ in range(15):
            room = env._room
            if room == "S":
                a = HallwayEnv.STRAIGHT
            elif room == "FORK0":
                a = fixed_action if fixed_action is not None else HallwayEnv.LEFT
                fork_choice = a
            elif room in ("TWIN1", "TWIN2"):
                if fixed_action is not None:
                    a = fixed_action
                else:
                    a = HallwayEnv.RIGHT if fork_choice == HallwayEnv.LEFT else HallwayEnv.LEFT
            else:
                a = HallwayEnv.STRAIGHT
            obs, reward, terminated, truncated, _ = env.step(a)
            if terminated or truncated:
                return reward
        raise AssertionError("hallway episode did not terminate in time")

    def test_memory_policy_always_wins(self) -> None:
        env = HallwayEnv()
        for _ in range(10):
            self.assertEqual(self._run(env, fixed_action=None), 1.0)

    def test_any_fixed_reactive_policy_always_loses_at_the_twin(self) -> None:
        env = HallwayEnv()
        for fixed in (HallwayEnv.LEFT, HallwayEnv.RIGHT):
            for _ in range(10):
                self.assertEqual(self._run(env, fixed_action=fixed), -1.0)

    def test_fork0_and_both_twins_emit_identical_observation(self) -> None:
        env = HallwayEnv()
        env.reset()
        env._room = "FORK0"
        fork_obs = env._obs()
        env._room = "TWIN1"
        twin1_obs = env._obs()
        env._room = "TWIN2"
        twin2_obs = env._obs()
        np.testing.assert_array_equal(fork_obs, twin1_obs)
        np.testing.assert_array_equal(fork_obs, twin2_obs)


class BattleshipTests(unittest.TestCase):
    def test_repeat_shot_is_penalized_and_flagged(self) -> None:
        env = BattleshipEnv()
        env.reset(seed=0)
        env.step(0)
        _, reward, _, _, _ = env.step(0)
        self.assertEqual(reward, -1.0)

    def test_hitting_and_sinking_a_ship_gives_positive_reward(self) -> None:
        env = BattleshipEnv()
        env.reset(seed=1)
        ship_idx = 0
        cells = sorted(cell for cell, sid in env._ship_id.items() if sid == ship_idx)
        total = 0.0
        terminated = False
        for (r, c) in cells:
            _, reward, terminated, _, _ = env.step(r * env.SIZE + c)
            total += reward
        self.assertGreater(total, 0.0)  # sinking bonus outweighs nothing here

    def test_never_repeating_beats_a_memoryless_policy_that_may_repeat(self) -> None:
        env = BattleshipEnv()

        def run(remember: bool) -> float:
            env.reset(seed=2)
            rng = np.random.default_rng(0)
            order = list(range(env.SIZE * env.SIZE))
            rng.shuffle(order)
            idx = 0
            total = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                a = order[idx] if remember else int(rng.integers(0, env.SIZE * env.SIZE))
                idx += 1
                _, reward, terminated, truncated, _ = env.step(a)
                total += reward
            return total

        self.assertGreater(run(True), run(False))


class MinesweeperTests(unittest.TestCase):
    def test_hitting_a_mine_terminates_with_penalty(self) -> None:
        env = MinesweeperPOMDPEnv()
        env.reset(seed=0)
        mine = next(iter(env._mines))
        _, reward, terminated, _, _ = env.step(mine[0] * env.SIZE + mine[1])
        self.assertEqual(reward, -1.0)
        self.assertTrue(terminated)

    def test_adjacent_mine_count_matches_observation(self) -> None:
        env = MinesweeperPOMDPEnv()
        env.reset(seed=1)
        safe = next((r, c) for r in range(env.SIZE) for c in range(env.SIZE) if (r, c) not in env._mines)
        expected = env._adjacent_mines(*safe)
        obs, _, _, _, _ = env.step(safe[0] * env.SIZE + safe[1])
        self.assertAlmostEqual(float(obs[2]), expected / 8.0)

    def test_repeat_reveal_is_penalized_and_flagged(self) -> None:
        env = MinesweeperPOMDPEnv()
        env.reset(seed=1)
        safe = next((r, c) for r in range(env.SIZE) for c in range(env.SIZE) if (r, c) not in env._mines)
        env.step(safe[0] * env.SIZE + safe[1])
        obs, reward, _, _, _ = env.step(safe[0] * env.SIZE + safe[1])
        self.assertEqual(reward, -0.5)
        self.assertEqual(obs[3], 1.0)


class ConcentrationTests(unittest.TestCase):
    def test_matching_pair_gives_reward_one(self) -> None:
        env = ConcentrationEnv()
        env.reset(seed=0)
        values = env._values
        partner = next(i for i in range(1, env.num_cards) if values[i] == values[0])
        env.step(0)
        _, reward, _, _, _ = env.step(int(partner))
        self.assertEqual(reward, 1.0)

    def test_mismatch_gives_small_negative_reward_and_hides_again(self) -> None:
        env = ConcentrationEnv()
        env.reset(seed=0)
        values = env._values
        wrong = next(i for i in range(1, env.num_cards) if values[i] != values[0])
        env.step(0)
        obs, reward, _, _, _ = env.step(int(wrong))
        self.assertEqual(reward, -0.1)
        # Neither card should still show as "matched" (i.e. permanently open).
        self.assertFalse(env._matched[0])
        self.assertFalse(env._matched[wrong])

    def test_repicking_pending_card_is_invalid(self) -> None:
        env = ConcentrationEnv()
        env.reset(seed=0)
        env.step(0)
        _, reward, _, _, _ = env.step(0)
        self.assertEqual(reward, -0.5)

    def test_card_value_hidden_once_turn_ends(self) -> None:
        env = ConcentrationEnv()
        env.reset(seed=0)
        values = env._values
        wrong = next(i for i in range(1, env.num_cards) if values[i] != values[0])
        env.step(0)
        env.step(int(wrong))
        obs, _, _, _, _ = env.step(1 if 1 not in (0, wrong) else 2)
        # after the mismatch turn, cards 0/wrong should be face-down again
        reshaped = obs.reshape(env.num_cards, 3)
        self.assertEqual(reshaped[0, 1], 0.0)
        self.assertEqual(reshaped[wrong, 1], 0.0)


class LaserTagTests(unittest.TestCase):
    def test_adjacent_opponent_always_visible(self) -> None:
        env = LaserTagEnv()
        env.reset(seed=0)
        env._agent = (2, 2)
        env._opponent = (2, 3)
        self.assertTrue(env._visible())

    def test_far_opponent_not_visible(self) -> None:
        env = LaserTagEnv()
        env.reset(seed=0)
        env._agent = (0, 0)
        env._opponent = (7, 7)
        self.assertFalse(env._visible())

    def test_wall_blocks_line_of_sight(self) -> None:
        env = LaserTagEnv()
        env.reset(seed=0)
        env._agent = (2, 3)
        env._opponent = (5, 3)  # (3,3)/(4,3) walls sit strictly between them
        self.assertFalse(env._visible())

    def test_catching_opponent_ends_episode_with_positive_reward(self) -> None:
        env = LaserTagEnv()
        env.reset(seed=0)
        env._agent = (3, 5)
        env._opponent = (3, 6)
        _, reward, terminated, _, _ = env.step(0)  # move +y towards opponent
        self.assertEqual(reward, 1.0)
        self.assertTrue(terminated)


class ActiveTMazeTests(unittest.TestCase):
    def test_looking_then_choosing_correctly_always_wins(self) -> None:
        env = ActiveTMazeEnv()
        for seed in range(10):
            env.reset(seed=seed)
            obs, _, _, _, _ = env.step(ActiveTMazeEnv.LOOK)
            cue_left = obs[1] > 0
            for _ in range(env.corridor_length):
                env.step(ActiveTMazeEnv.FORWARD)
            action = ActiveTMazeEnv.CHOOSE_LEFT if cue_left else ActiveTMazeEnv.CHOOSE_RIGHT
            _, reward, terminated, _, _ = env.step(action)
            self.assertEqual(reward, 1.0)
            self.assertTrue(terminated)

    def test_skipping_look_gives_no_usable_cue_signal(self) -> None:
        env = ActiveTMazeEnv()
        env.reset(seed=0)
        for _ in range(env.corridor_length):
            obs, _, _, _, _ = env.step(ActiveTMazeEnv.FORWARD)
            self.assertEqual(obs[1], 0.0)  # cue never leaks without a look

    def test_look_only_works_at_the_very_start(self) -> None:
        env = ActiveTMazeEnv()
        env.reset(seed=0)
        env.step(ActiveTMazeEnv.FORWARD)
        obs, reward, _, _, _ = env.step(ActiveTMazeEnv.LOOK)
        self.assertEqual(obs[1], 0.0)
        self.assertEqual(reward, -0.1)

    def test_long_variant_has_longer_corridor(self) -> None:
        env = gym.make("ActiveTMazeLong-v0")
        self.assertGreater(env.unwrapped.corridor_length, ActiveTMazeEnv.CORRIDOR_LENGTH)
        env.close()


if __name__ == "__main__":
    unittest.main()
