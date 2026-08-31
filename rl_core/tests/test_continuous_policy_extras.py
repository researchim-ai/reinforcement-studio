"""Regression tests for the continuous-action-space "boosters" (gSDE via
`use_sde`, Beta-distribution policy head via `use_beta`, extra per-head
hidden layer via `head_hidden_size` — see `ActorCriticNet` in
rl_core/algorithms/native/networks.py) across every algorithm that now
wires them through, not just `NativePPO` (which is where they were first
added, for CarRacing). None of the three had a single direct test before
this file — CarRacing's own coverage is env-registry-level, not a unit
test exercising these hyperparams — so this also closes that gap for
`NativePPO` itself.

`NativeA2C` shares `ActorCriticNet` and `OnPolicyAlgorithm`'s collection
loop with `NativePPO`, so wiring these through was almost a copy of
`NativePPO.__init__`; `MultiAgentPPO` (`ippo`) already had `use_beta`/
`head_hidden_size` per-team and only gained `use_sde` here, with its own
hand-rolled step loop (it isn't built on `OnPolicyAlgorithm`) needing its
own periodic `reset_noise()` call."""
from __future__ import annotations

import tempfile
from pathlib import Path

import gymnasium as gym

from rl_core import scene_store
from rl_core.algorithms.base import TrainingCallback
from rl_core.algorithms.native.a2c import DEFAULT_HYPERPARAMS as A2C_DEFAULTS
from rl_core.algorithms.native.a2c import NativeA2C
from rl_core.algorithms.native.marl_ppo import DEFAULT_HYPERPARAMS as IPPO_DEFAULTS
from rl_core.algorithms.native.marl_ppo import MultiAgentPPO
from rl_core.algorithms.native.networks import ActorCriticNet
from rl_core.algorithms.native.ppo import DEFAULT_HYPERPARAMS as PPO_DEFAULTS
from rl_core.algorithms.native.ppo import NativePPO
from rl_core.envs.scene_env import SceneMultiAgentEnv

_CONTINUOUS_ENV_ID = "MountainCarContinuous-v0"  # Box(-1, 1) — finite bounds, required for use_beta


def _run_on_policy_case(algo_cls, defaults: dict, extra_hyperparams: dict) -> None:
    env = gym.make(_CONTINUOUS_ENV_ID)
    hyperparams = {**defaults, "n_steps": 16, "batch_size": 8, "n_epochs": 2, **extra_hyperparams}
    algo = algo_cls(env, hyperparams, seed=0, device="cpu")

    steps: list[int] = []

    def writer(num_timesteps, episode_reward=None, episode_length=None, metrics=None):
        steps.append(num_timesteps)
        return num_timesteps < 64

    algo.learn(total_timesteps=64, callback=TrainingCallback(writer))
    assert steps[-1] >= 64

    obs, _ = env.reset(seed=1)
    action, _ = algo.predict(obs, deterministic=True)
    assert env.action_space.contains(action)

    with tempfile.TemporaryDirectory() as d:
        checkpoint = Path(d) / "model.zip"
        algo.save(checkpoint)
        loaded = algo_cls.load(checkpoint, env, device="cpu")
        loaded.predict(obs, deterministic=True)
    env.close()


class TestNativePPOContinuousExtras:
    def test_use_sde(self) -> None:
        _run_on_policy_case(NativePPO, PPO_DEFAULTS, {"use_sde": 1, "sde_sample_freq": 2})

    def test_use_beta(self) -> None:
        _run_on_policy_case(NativePPO, PPO_DEFAULTS, {"use_beta": 1})

    def test_head_hidden_size(self) -> None:
        _run_on_policy_case(NativePPO, PPO_DEFAULTS, {"head_hidden_size": 32})

    def test_use_beta_wins_over_use_sde(self) -> None:
        env = gym.make(_CONTINUOUS_ENV_ID)
        algo = NativePPO(env, {**PPO_DEFAULTS, "use_sde": 1, "use_beta": 1}, seed=0, device="cpu")
        assert isinstance(algo.net, ActorCriticNet)
        assert algo.net.use_beta is True
        assert algo.net.use_sde is False
        env.close()


class TestNativeA2CContinuousExtras:
    """A2C never had these three hyperparams before — this is the whole
    point of extending the same "boosters" beyond PPO."""

    def test_use_sde(self) -> None:
        _run_on_policy_case(NativeA2C, A2C_DEFAULTS, {"use_sde": 1, "sde_sample_freq": 2})

    def test_use_beta(self) -> None:
        _run_on_policy_case(NativeA2C, A2C_DEFAULTS, {"use_beta": 1})

    def test_head_hidden_size(self) -> None:
        _run_on_policy_case(NativeA2C, A2C_DEFAULTS, {"head_hidden_size": 32})

    def test_defaults_are_off_and_match_ppo_shape(self) -> None:
        assert A2C_DEFAULTS["use_sde"] == 0
        assert A2C_DEFAULTS["use_beta"] == 0
        assert A2C_DEFAULTS["head_hidden_size"] == 0


def _continuous_team_battle_spec() -> dict:
    """`default_team_battle_spec()` with both groups switched to continuous
    movement — the scene-wide `movement`/`sensors` schema is shared by
    every lane (see `SceneMultiAgentEnv.__init__`'s docstring), so only the
    first group's `movement` actually matters, but both are set for
    clarity."""
    spec = scene_store.default_team_battle_spec()
    for group in spec["agents"]:
        group["movement"] = {"type": "continuous", "speed": 0.55}
    return spec


class TestMultiAgentPPOContinuousExtras:
    """`ippo` already had `use_beta`/`head_hidden_size` per-team; `use_sde`
    is the new addition here, with its own `reset_noise()` call inside
    `MultiAgentPPO.learn`'s hand-rolled step loop (not built on
    `OnPolicyAlgorithm`, so no shared collection loop to piggyback on)."""

    def test_use_sde_trains_and_saves_roundtrip(self) -> None:
        spec = _continuous_team_battle_spec()
        env = SceneMultiAgentEnv(spec)
        hp = {**IPPO_DEFAULTS, "n_steps": 16, "batch_size": 8, "n_epochs": 2, "use_sde": 1, "sde_sample_freq": 2}
        algo = MultiAgentPPO(env, hp, seed=0, device="cpu")
        for policy in algo.policies.values():
            assert policy.use_sde is True

        steps: list[int] = []

        def writer(num_timesteps, episode_reward=None, episode_length=None, metrics=None):
            steps.append(num_timesteps)
            return num_timesteps < 64

        algo.learn(total_timesteps=64, callback=TrainingCallback(writer))
        assert steps[-1] >= 64

        obs, _ = env.reset(seed=1)
        action, _ = algo.predict(obs[0], deterministic=True)
        assert env.single_action_space.contains(action)

        with tempfile.TemporaryDirectory() as d:
            model_path = Path(d) / "model.zip"
            algo.save(model_path)
            loaded = MultiAgentPPO.load(model_path, env, device="cpu")
            assert set(loaded.teams) == {"red", "blue"}
        env.close()

    def test_use_beta_still_works_per_team(self) -> None:
        spec = _continuous_team_battle_spec()
        env = SceneMultiAgentEnv(spec)
        hp = {**IPPO_DEFAULTS, "n_steps": 16, "batch_size": 8, "n_epochs": 1, "use_beta": 1}
        algo = MultiAgentPPO(env, hp, seed=0, device="cpu")
        for policy in algo.policies.values():
            assert policy.net.use_beta is True
        env.close()
