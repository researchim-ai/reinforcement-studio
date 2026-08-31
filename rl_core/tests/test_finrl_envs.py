"""Tests for the FinRL-style multi-asset portfolio environments (see
rl_core/envs/finrl_envs.py) — no third-party dependency, so unlike every
`nle`/`minihack`/`gymnasium_robotics`-backed test file in this directory,
nothing here is conditionally skipped."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from rl_core.envs import registry
from rl_core.envs.finrl_envs import FinRLPortfolioAllocationEnv, FinRLStockTradingEnv


def test_new_envs_listed_and_available():
    envs = {e["id"]: e for e in registry.list_environments()}
    for eid in ["FinRL-StockTrading-v0", "FinRL-PortfolioAllocation-v0"]:
        assert envs[eid]["available"] is True, eid
        assert envs[eid]["category"] == "trading", eid
        assert envs[eid]["action_kind"] == "continuous", eid


class TestFinRLStockTradingEnv:
    def test_reset_and_step_shapes(self):
        env = FinRLStockTradingEnv(num_stocks=4)
        obs, info = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        assert env.action_space.shape == (4,)
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs.shape == env.observation_space.shape
        assert np.all(np.isfinite(obs))
        env.close()

    def test_buying_reduces_cash_and_increases_shares(self):
        env = FinRLStockTradingEnv(num_stocks=3, initial_amount=100_000.0, hmax=10)
        env.reset(seed=0)
        cash_before = env._cash
        action = np.ones(3, dtype=np.float32)  # buy hmax shares of every stock
        env.step(action)
        assert env._cash < cash_before
        assert np.all(env._shares > 0)
        env.close()

    def test_selling_more_than_held_is_clamped_not_negative(self):
        env = FinRLStockTradingEnv(num_stocks=2, hmax=10)
        env.reset(seed=0)
        # Never bought anything — a full "sell everything" action must not
        # push holdings negative (no shorting in this env).
        obs, reward, terminated, truncated, info = env.step(np.array([-1.0, -1.0], dtype=np.float32))
        assert np.all(env._shares >= 0.0)
        env.close()

    def test_portfolio_value_matches_cash_plus_holdings(self):
        env = FinRLStockTradingEnv(num_stocks=3, hmax=5)
        env.reset(seed=0)
        env.step(env.action_space.sample())
        expected = env._cash + float(np.sum(env._shares * env._market.prices))
        assert env._total_value() == pytest.approx(expected)
        env.close()

    def test_reward_equals_scaled_change_in_total_value(self):
        env = FinRLStockTradingEnv(num_stocks=3, initial_amount=100_000.0, hmax=5)
        env.reset(seed=0)
        prev_value = env._total_value()
        obs, reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        new_value = env._total_value()
        expected_reward = (new_value - prev_value) / env.initial_amount * 100.0
        assert reward == pytest.approx(expected_reward, abs=1e-6)
        env.close()

    def test_episode_truncates_at_max_steps(self):
        env = FinRLStockTradingEnv(num_stocks=2, max_steps=15)
        env.reset(seed=0)
        terminated = truncated = False
        for _ in range(15):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                break
        assert terminated or truncated
        env.close()

    def test_render_returns_rgb_array(self):
        env = FinRLStockTradingEnv(num_stocks=3, render_mode="rgb_array")
        env.reset(seed=0)
        env.step(env.action_space.sample())
        img = env.render()
        assert img is not None
        assert img.dtype == np.uint8
        assert img.shape[2] == 3
        env.close()

    def test_gym_make_registered_id(self):
        env = gym.make("FinRL-StockTrading-v0")
        env.reset(seed=0)
        env.step(env.action_space.sample())
        env.close()


class TestFinRLPortfolioAllocationEnv:
    def test_reset_and_step_shapes(self):
        env = FinRLPortfolioAllocationEnv(num_stocks=4, window=10)
        obs, info = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        assert obs.shape == (4 * 4 + 4 * 2,)
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs))
        env.close()

    def test_weights_are_always_normalized_simplex(self):
        env = FinRLPortfolioAllocationEnv(num_stocks=5, window=10)
        env.reset(seed=0)
        for _ in range(10):
            env.step(env.action_space.sample())
            assert env._weights.sum() == pytest.approx(1.0, abs=1e-5)
            assert np.all(env._weights >= 0.0)
        env.close()

    def test_no_turnover_when_action_repeats_uniform_weights(self):
        env = FinRLPortfolioAllocationEnv(num_stocks=4, window=10, turnover_cost_rate=0.01)
        env.reset(seed=0)
        # A uniform raw-score action softmaxes to uniform weights — same as
        # the initial `_weights`, so turnover (and its cost) must be ~0.
        uniform_action = np.zeros(4, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(uniform_action)
        assert info["turnover"] == pytest.approx(0.0, abs=1e-6)
        env.close()

    def test_reward_telescopes_to_log_equity_change(self):
        env = FinRLPortfolioAllocationEnv(num_stocks=3, window=10, turnover_cost_rate=0.0)
        env.reset(seed=0)
        total_reward = 0.0
        for _ in range(20):
            obs, reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
            total_reward += reward
            if terminated or truncated:
                break
        assert total_reward == pytest.approx(np.log(env._equity) * 100.0, abs=1e-3)
        env.close()

    def test_episode_truncates_at_max_steps(self):
        env = FinRLPortfolioAllocationEnv(num_stocks=3, window=5, max_steps=15)
        env.reset(seed=0)
        terminated = truncated = False
        for _ in range(15):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                break
        assert terminated or truncated
        env.close()

    def test_render_returns_rgb_array(self):
        env = FinRLPortfolioAllocationEnv(num_stocks=3, render_mode="rgb_array")
        env.reset(seed=0)
        env.step(env.action_space.sample())
        img = env.render()
        assert img is not None
        assert img.dtype == np.uint8
        assert img.shape[2] == 3
        env.close()

    def test_gym_make_registered_id(self):
        env = gym.make("FinRL-PortfolioAllocation-v0")
        env.reset(seed=0)
        env.step(env.action_space.sample())
        env.close()


def test_ppo_smoke_on_finrl_stock_trading():
    from rl_core.algorithms.native_runner import run

    config = {
        "kind": "gym",
        "environment": {"id": "FinRL-StockTrading-v0", "wrappers": []},
        "algorithm": {"id": "ppo", "hyperparams": {"n_steps": 32, "batch_size": 16, "n_epochs": 1, "use_beta": 1}},
        "training": {"total_timesteps": 96, "seed": 0},
    }
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        run(config, run_dir)
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["status"] == "completed"
        assert metrics["step"] >= 96


def test_ppo_smoke_on_finrl_portfolio_allocation():
    from rl_core.algorithms.native_runner import run

    config = {
        "kind": "gym",
        "environment": {"id": "FinRL-PortfolioAllocation-v0", "wrappers": []},
        "algorithm": {"id": "ppo", "hyperparams": {"n_steps": 32, "batch_size": 16, "n_epochs": 1}},
        "training": {"total_timesteps": 96, "seed": 0},
    }
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        run(config, run_dir)
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["status"] == "completed"
        assert metrics["step"] >= 96
