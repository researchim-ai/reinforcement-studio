"""Tests for the from-scratch industrial (Job Shop Scheduling, Bin
Packing) and trading environments (see rl_core/envs/industrial_envs.py,
trading_envs.py, registry.py) — no third-party dependency, so unlike every
other new-environment test file in this directory, nothing here is
conditionally skipped."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from rl_core.envs import registry
from rl_core.envs.industrial_envs import BinPackingEnv, JobShopEnv
from rl_core.envs.trading_envs import TradingEnv


def test_new_envs_listed_and_available():
    envs = {e["id"]: e for e in registry.list_environments()}
    for eid, category, action_kind in [
        ("JobShop-6x6-v0", "industrial", "discrete"),
        ("JobShop-10x10-v0", "industrial", "discrete"),
        ("BinPacking-v0", "industrial", "discrete"),
        ("Trading-Discrete-v0", "trading", "discrete"),
        ("Trading-Continuous-v0", "trading", "continuous"),
    ]:
        assert envs[eid]["available"] is True, eid
        assert envs[eid]["category"] == category, eid
        assert envs[eid]["action_kind"] == action_kind, eid


class TestJobShopEnv:
    def test_reset_and_step_shapes(self):
        env = JobShopEnv(num_jobs=4, num_machines=3)
        obs, info = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        assert env.action_space == gym.spaces.Discrete(4)
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        env.close()

    def test_invalid_action_is_penalized_without_state_change(self):
        env = JobShopEnv(num_jobs=2, num_machines=2)
        env.reset(seed=0)
        # Finish job 0 entirely (2 machines -> 2 valid dispatches).
        env.step(0)
        env.step(0)
        makespan_before = env._makespan
        obs, reward, terminated, truncated, info = env.step(0)  # job 0 already done -> invalid
        assert reward == -1.0
        assert env._makespan == makespan_before
        env.close()

    def test_episode_terminates_when_all_operations_scheduled(self):
        env = JobShopEnv(num_jobs=3, num_machines=3)
        env.reset(seed=0)
        terminated = False
        # Round-robin dispatch always eventually finishes every job's every
        # operation — deterministic upper bound on steps needed.
        for _ in range(3 * 3 * 3):
            for job in range(3):
                obs, reward, terminated, truncated, info = env.step(job)
                if terminated:
                    break
            if terminated:
                break
        assert terminated
        assert "makespan" in info
        assert info["makespan"] > 0
        assert "total_weighted_tardiness" in info
        assert "tardy_jobs" in info
        env.close()

    def test_makespan_term_of_reward_telescopes_to_negative_final_makespan(self):
        # The reward has two telescoping components (makespan + weighted
        # tardiness, see class docstring) — isolate the makespan one by
        # zeroing out the tardiness contribution (due dates loose enough
        # that nothing is ever late) rather than asserting on the combined
        # total.
        env = JobShopEnv(num_jobs=3, num_machines=3, due_date_tightness=(50.0, 50.0), proc_time_noise_std=0.0)
        env.reset(seed=1)
        total_reward = 0.0
        terminated = False
        for _ in range(50):
            for job in range(3):
                obs, reward, terminated, truncated, info = env.step(job)
                total_reward += reward
                if terminated:
                    break
            if terminated:
                break
        assert terminated
        assert info["total_weighted_tardiness"] == pytest.approx(0.0, abs=1e-6)
        assert total_reward == pytest.approx(-info["makespan"] / env.max_duration, abs=1e-4)
        env.close()

    def test_tight_due_dates_produce_tardiness(self):
        # Due dates far tighter than any feasible schedule guarantees at
        # least some lateness, whatever dispatch order is used.
        env = JobShopEnv(num_jobs=3, num_machines=3, due_date_tightness=(0.2, 0.2), proc_time_noise_std=0.0)
        env.reset(seed=2)
        terminated = False
        for _ in range(50):
            for job in range(3):
                obs, reward, terminated, truncated, info = env.step(job)
                if terminated:
                    break
            if terminated:
                break
        assert terminated
        assert info["total_weighted_tardiness"] > 0.0
        assert info["tardy_jobs"] > 0
        env.close()

    def test_switching_families_on_same_machine_incurs_setup_time(self):
        env = JobShopEnv(num_jobs=2, num_machines=2, num_families=2, setup_time=7.0, proc_time_noise_std=0.0)
        env.reset(seed=0)
        # Force the two jobs into different families so machine 0's second
        # dispatch always needs a setup, regardless of the random draw.
        env._families = np.array([0, 1])
        env.step(0)  # job 0's first op — whichever machine that routes to, no setup yet (machine untouched)
        machine_before = int(env._machine_order[1, 0])
        free_before = float(env._machine_free_time[machine_before])
        env.step(1)  # job 1's first op — same or different machine depending on routing
        # Only assert the invariant that actually holds regardless of
        # routing: if job 1's first op landed on the *same* machine job 0's
        # first op did, the gap between "machine became free" and "next op
        # actually started" must be at least `setup_time` (a different
        # family just ran there).
        machine0_first = int(env._machine_order[0, 0])
        machine1_first = int(env._machine_order[1, 0])
        if machine0_first == machine1_first:
            setup_start, proc_start = env._schedule[1][4], env._schedule[1][5]
            assert proc_start - setup_start == pytest.approx(7.0)
        env.close()

    def test_stochastic_processing_time_noise_changes_actual_duration(self):
        env = JobShopEnv(num_jobs=2, num_machines=2, proc_time_noise_std=0.5, setup_time=0.0)
        env.reset(seed=0)
        nominal = float(env._proc_times[0, int(env._op_idx[0])])
        env.step(0)
        _job, _machine, _op, _family, _setup_start, proc_start, end = env._schedule[-1]
        actual = end - proc_start
        # With sigma=0.5 the draw essentially never lands exactly on the
        # nominal duration — flags a regression to "noise silently
        # disabled" without pinning an exact (seed-fragile) value.
        assert actual != pytest.approx(nominal)
        env.close()

    def test_render_returns_rgb_array(self):
        env = JobShopEnv(num_jobs=3, num_machines=3, render_mode="rgb_array")
        env.reset(seed=0)
        env.step(0)
        img = env.render()
        assert img is not None
        assert img.dtype == np.uint8
        assert img.ndim == 3
        assert img.shape[2] == 3
        env.close()

    def test_gym_make_registered_ids(self):
        for eid, jobs, machines in [("JobShop-6x6-v0", 6, 6), ("JobShop-10x10-v0", 10, 10)]:
            env = gym.make(eid)
            assert env.action_space.n == jobs
            env.reset(seed=0)
            env.close()


class TestBinPackingEnv:
    def test_reset_and_step_shapes(self):
        env = BinPackingEnv(max_bins=5, num_items=10)
        obs, info = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        obs, reward, terminated, truncated, info = env.step(env.max_bins)  # always "open new bin"
        assert obs.shape == env.observation_space.shape
        env.close()

    def test_fitting_into_open_bin_is_free(self):
        env = BinPackingEnv(bin_capacity=100, num_items=5, min_item_size=10, max_item_size=10, max_bins=5)
        env.reset(seed=0)
        # First item always opens a new bin (no bins open yet).
        obs, reward, terminated, truncated, info = env.step(0)
        assert reward == -1.0
        assert env._num_open == 1
        # Second (size-10) item easily fits the same 90%-empty bin 0.
        obs, reward, terminated, truncated, info = env.step(0)
        assert reward == 0.0
        assert env._num_open == 1
        env.close()

    def test_episode_terminates_after_num_items(self):
        env = BinPackingEnv(max_bins=10, num_items=8)
        env.reset(seed=0)
        terminated = False
        for _ in range(8):
            obs, reward, terminated, truncated, info = env.step(env.max_bins)
        assert terminated
        assert info["num_bins_used"] == env._num_open
        env.close()

    def test_overflow_beyond_max_bins_is_penalized(self):
        env = BinPackingEnv(bin_capacity=10, num_items=3, min_item_size=10, max_item_size=10, max_bins=1)
        env.reset(seed=0)
        env.step(1)  # opens the only bin slot, fills it completely
        obs, reward, terminated, truncated, info = env.step(1)  # can't fit, no more slots
        assert reward == -3.0
        env.close()

    def test_gym_make_registered_id(self):
        env = gym.make("BinPacking-v0")
        env.reset(seed=0)
        env.close()


class TestTradingEnv:
    def test_discrete_action_space_and_obs_shape(self):
        env = TradingEnv(continuous=False, window=10)
        obs, info = env.reset(seed=0)
        assert env.action_space == gym.spaces.Discrete(3)
        assert obs.shape == (16,)  # window (10) + 6 engineered features
        env.close()

    def test_continuous_action_space(self):
        env = TradingEnv(continuous=True, window=10)
        env.reset(seed=0)
        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (1,)
        obs, reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
        assert env._position == pytest.approx(0.5)
        env.close()

    def test_flat_position_has_zero_pnl_and_only_pays_switching_cost(self):
        env = TradingEnv(continuous=False, window=5)
        env.reset(seed=0)
        # Staying flat (action=1) should never incur pnl or any cost —
        # there's nothing to switch away from after the first flat step.
        env.step(1)
        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(1)
            assert reward == 0.0
        env.close()

    def test_switching_position_incurs_cost(self):
        env = TradingEnv(continuous=False, window=5)
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(2)  # flat -> long: commission + spread + slippage > 0
        assert info["commission_cost"] > 0.0
        assert info["spread_cost"] > 0.0
        env.close()

    def test_bars_are_valid_ohlc(self):
        env = TradingEnv(continuous=False, window=5)
        env.reset(seed=0)
        for _ in range(20):
            env.step(1)
        for o, h, low, c in env._bars:
            assert low <= o <= h
            assert low <= c <= h
            assert low <= h

    def test_episode_truncates_at_max_steps(self):
        env = TradingEnv(continuous=False, window=5, max_steps=20)
        env.reset(seed=0)
        truncated = False
        for _ in range(20):
            obs, reward, terminated, truncated, info = env.step(1)
            if terminated or truncated:
                break
        assert truncated or terminated
        env.close()

    def test_render_returns_rgb_array(self):
        env = TradingEnv(continuous=False, window=5, render_mode="rgb_array")
        env.reset(seed=0)
        rng = np.random.default_rng(0)
        for _ in range(10):
            env.step(int(rng.integers(0, 3)))
        img = env.render()
        assert img is not None
        assert img.dtype == np.uint8
        assert img.shape[2] == 3
        env.close()

    def test_gym_make_registered_ids(self):
        for eid in ["Trading-Discrete-v0", "Trading-Continuous-v0"]:
            env = gym.make(eid)
            env.reset(seed=0)
            env.step(env.action_space.sample())
            env.close()

    def test_ppo_smoke_on_trading_continuous(self):
        from rl_core.algorithms.native_runner import run

        config = {
            "kind": "gym",
            "environment": {"id": "Trading-Continuous-v0", "wrappers": []},
            "algorithm": {"id": "ppo", "hyperparams": {"n_steps": 32, "batch_size": 16, "n_epochs": 1, "use_beta": 1}},
            "training": {"total_timesteps": 96, "seed": 0},
        }
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d)
            run(config, run_dir)
            metrics = json.loads((run_dir / "metrics.json").read_text())
            assert metrics["status"] == "completed"
            assert metrics["step"] >= 96

    def test_dqn_smoke_on_trading_discrete(self):
        from rl_core.algorithms.native_runner import run

        config = {
            "kind": "gym",
            "environment": {"id": "Trading-Discrete-v0", "wrappers": []},
            "algorithm": {"id": "dqn", "hyperparams": {"learning_rate": 1e-3, "buffer_size": 500}},
            "training": {"total_timesteps": 96, "seed": 0},
        }
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d)
            run(config, run_dir)
            metrics = json.loads((run_dir / "metrics.json").read_text())
            assert metrics["status"] == "completed"
            assert metrics["step"] >= 96
