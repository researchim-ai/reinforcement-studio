from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from backend.routes.environments import ALGORITHM_CATALOG
import rl_core.algorithms.native.latentimzero as latentimzero_module
from rl_core.algorithms.native.latentimzero import (
    DEFAULT_HYPERPARAMS,
    LATENTIMZERO_CHECKPOINT_VERSION,
    NativeLatentImZero,
    _SearchUncertaintyProbe,
)
from rl_core.algorithms.native.researchimzero import (
    DEFAULT_HYPERPARAMS as RESEARCH_DEFAULTS,
    NativeResearchImZero,
)
from rl_core.algorithms.native_runner import ALGO_CLASSES, DEFAULT_HYPERPARAMS as RUNNER_DEFAULTS


TINY = {
    **RESEARCH_DEFAULTS,
    "embed_dim": 16,
    "num_layers": 1,
    "num_heads": 2,
    "context_length": 1,
    "buffer_size": 40,
    "batch_size": 2,
    "unroll_steps": 2,
    "td_steps": 2,
    "num_sampled_actions": 2,
    "num_simulations": 2,
    "num_simulations_initial": 2,
    "num_top_actions": 2,
    "value_support_size": 10,
    "proj_dim": 8,
    "closed_loop_loss_coef": 0.0,
    "reanalyze_batch_size": 0,
    "use_amp": 0,
}


class _DiscreteEnv(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        return np.full(4, float(action) / 3, dtype=np.float32), 1.0, True, False, {}


class _ContinuousEnv(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, shape=(5,), dtype=np.float32)
    action_space = gym.spaces.Box(
        np.array([-2.0, -0.5], dtype=np.float32),
        np.array([2.0, 0.5], dtype=np.float32),
    )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(5, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(5, dtype=np.float32), -float(np.square(action).sum()), True, False, {}


class _ImageEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 255, shape=(3, 32, 32), dtype=np.uint8)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros((3, 32, 32), dtype=np.uint8), {}

    def step(self, action):
        return np.full((3, 32, 32), action, dtype=np.uint8), 0.0, True, False, {}


def _shared_state(algo: NativeResearchImZero) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name in (
        "tokenizer", "action_embed", "transformer", "heads", "projector", "predictor",
        "target_tokenizer", "target_action_embed", "target_transformer", "target_heads",
    ):
        for key, value in getattr(algo, name).state_dict().items():
            state[f"{name}.{key}"] = value.detach().clone()
    state["loss_log_vars"] = algo.loss_log_vars.detach().clone()
    return state


def _assert_state_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> None:
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), key


def _fill_buffer(algo: NativeResearchImZero, length: int = 8) -> None:
    obs = np.zeros(4, dtype=np.float32)
    for step in range(length):
        action_index = step % 3
        action = np.eye(3, dtype=np.float32)[action_index]
        next_obs = np.full(4, (step + 1) / length, dtype=np.float32)
        algo.buffer.add(
            obs, action, float(step % 2), next_obs,
            np.full(3, 1 / 3, dtype=np.float32), step == length - 1,
        )
        obs = next_obs


def test_v7_inherits_research_and_defaults_are_exact_superset() -> None:
    assert issubclass(NativeLatentImZero, NativeResearchImZero)
    assert {key: DEFAULT_HYPERPARAMS[key] for key in RESEARCH_DEFAULTS} == RESEARCH_DEFAULTS
    assert DEFAULT_HYPERPARAMS["embed_dim"] == 128
    assert DEFAULT_HYPERPARAMS["batch_size"] == 64
    assert DEFAULT_HYPERPARAMS["train_steps_per_iter"] == 2
    assert DEFAULT_HYPERPARAMS["num_simulations"] == 32
    assert DEFAULT_HYPERPARAMS["learning_starts"] == 500
    assert DEFAULT_HYPERPARAMS["use_amp"] == 1
    assert DEFAULT_HYPERPARAMS["uncertainty_enabled"] == 1
    assert ALGO_CLASSES["latentimzero"] is NativeLatentImZero
    assert RUNNER_DEFAULTS["latentimzero"] == DEFAULT_HYPERPARAMS


def test_force_mode_has_identical_initial_state_search_and_predict() -> None:
    research = NativeResearchImZero(_DiscreteEnv(), TINY, seed=7, device="cpu")
    latent = NativeLatentImZero(
        _DiscreteEnv(), {**TINY, "force_research_mode": 1, "uncertainty_enabled": 1},
        seed=7, device="cpu",
    )
    _assert_state_equal(_shared_state(research), _shared_state(latent))
    assert latent.uncertainty_runtime_scale == 0.0

    obs = np.zeros((1, 4), dtype=np.float32)
    np.random.seed(31)
    expected = research.search(obs, [None], deterministic=np.array([True]))[0]
    np.random.seed(31)
    actual = latent.search(obs, [None], deterministic=np.array([True]))[0]
    assert actual["env_action"] == expected["env_action"]
    assert np.array_equal(actual["policy_target"], expected["policy_target"])
    assert actual["value_target"] == expected["value_target"]

    np.random.seed(41)
    expected_action, _ = research.predict(obs[0], deterministic=True, episode_start=True)
    np.random.seed(41)
    actual_action, _ = latent.predict(obs[0], deterministic=True, episode_start=True)
    assert actual_action == expected_action


def test_force_mode_keeps_one_core_update_exact_without_probe_step() -> None:
    research = NativeResearchImZero(_DiscreteEnv(), TINY, seed=11, device="cpu")
    latent = NativeLatentImZero(
        _DiscreteEnv(),
        {**TINY, "force_research_mode": 1, "uncertainty_enabled": 1},
        seed=11,
        device="cpu",
    )
    _fill_buffer(research)
    _fill_buffer(latent)
    probe_before = {
        key: value.clone() for key, value in latent.uncertainty_probe.state_dict().items()
    }

    np.random.seed(123)
    rng_state = np.random.get_state()
    research_metrics = research._train_step()
    np.random.set_state(rng_state)
    latent_metrics = latent._train_step()

    _assert_state_equal(_shared_state(research), _shared_state(latent))
    assert research.optimizer.state_dict()["state"].keys() == latent.optimizer.state_dict()["state"].keys()
    assert "uncertainty_aux_loss" not in latent_metrics
    assert latent_metrics == research_metrics
    assert latent.uncertainty_optimizer.state_dict()["state"] == {}
    for key, value in probe_before.items():
        assert torch.equal(value, latent.uncertainty_probe.state_dict()[key])


def test_deleted_v6_modules_are_not_active() -> None:
    algo = NativeLatentImZero(_DiscreteEnv(), TINY, seed=0, device="cpu")
    for name in (
        "world_model", "actor", "critic", "ema_critic", "model_optimizer",
        "actor_optimizer", "critic_optimizer", "observation_target",
    ):
        assert not hasattr(algo, name)
    module_names = {type(module).__name__ for module in algo.modules()} if hasattr(algo, "modules") else set()
    assert "_StochasticWorldModel" not in module_names
    assert not hasattr(latentimzero_module, "_StochasticWorldModel")


def test_auxiliary_probe_gradients_cannot_touch_core() -> None:
    algo = NativeLatentImZero(
        _DiscreteEnv(), {**TINY, "uncertainty_enabled": 1}, seed=0, device="cpu",
    )
    for parameter in algo._params:
        parameter.grad = None
    hidden = torch.randn(2, 5, algo.embed_dim, requires_grad=True)
    batch = {
        "reward": np.ones((2, 2), dtype=np.float32),
        "mask": np.ones((2, 2), dtype=np.float32),
    }
    metrics = algo._after_core_train_step(batch, hidden, torch.zeros(2, dtype=torch.long))
    assert "uncertainty_aux_loss" in metrics
    assert all(parameter.grad is None for parameter in algo._params)
    assert hidden.grad is None
    assert any(parameter.grad is not None for parameter in algo.uncertainty_probe.parameters())


def _set_probe_predictions(algo: NativeLatentImZero, biases: list[float]) -> None:
    with torch.no_grad():
        for head, bias in zip(algo.uncertainty_probe.reward_heads, biases):
            for parameter in head.parameters():
                parameter.zero_()
            head[-1].bias.fill_(bias)


def _set_calibrated_sparse_state(algo: NativeLatentImZero) -> None:
    algo._uncertainty_ema_initialized = True
    algo._uncertainty_probe_loss_ema = 0.0
    algo._uncertainty_disagreement_ema = 1.0
    algo._uncertainty_zero_reward_fraction_ema = 1.0
    algo._num_timesteps = algo.uncertainty_warmup_steps + algo.uncertainty_ramp_steps


def test_warmup_keeps_reward_exact_but_trains_probe() -> None:
    hidden = torch.randn(4, 16)
    reward = torch.randn(4)
    algo = NativeLatentImZero(_DiscreteEnv(), TINY, seed=0, device="cpu")
    assert algo.uncertainty_runtime_scale == 0.0
    assert algo._adjust_search_reward(hidden, reward) is reward
    before = {
        key: value.clone() for key, value in algo.uncertainty_probe.state_dict().items()
    }
    batch = {
        "reward": np.array([[0.0, 1e6], [0.0, 0.0]], dtype=np.float32),
        "mask": np.ones((2, 2), dtype=np.float32),
    }
    metrics = algo._after_core_train_step(
        batch, torch.randn(2, 5, algo.embed_dim), torch.zeros(2, dtype=torch.long),
    )
    assert "uncertainty_aux_loss" in metrics
    assert metrics["uncertainty_schedule_scale"] == 0.0
    assert any(
        not torch.equal(value, algo.uncertainty_probe.state_dict()[key])
        for key, value in before.items()
    )


def test_sparse_uncertainty_bonus_is_positive_and_bounded_after_warmup() -> None:
    algo = NativeLatentImZero(_DiscreteEnv(), TINY, seed=0, device="cpu")
    _set_probe_predictions(algo, [-1.0, 1.0])
    _set_calibrated_sparse_state(algo)
    reward = torch.zeros(4)
    adjusted = algo._adjust_search_reward(torch.zeros(4, algo.embed_dim), reward)
    bonus = adjusted - reward
    assert torch.all(bonus > 0)
    assert float(bonus.max()) <= algo.uncertainty_bonus_coef * algo.uncertainty_scale


def test_identical_probe_predictions_give_zero_bonus() -> None:
    algo = NativeLatentImZero(_DiscreteEnv(), TINY, seed=0, device="cpu")
    _set_probe_predictions(algo, [0.25, 0.25])
    _set_calibrated_sparse_state(algo)
    reward = torch.zeros(3)
    assert torch.equal(
        algo._adjust_search_reward(torch.zeros(3, algo.embed_dim), reward),
        reward,
    )


def test_dense_rewards_suppress_uncertainty_bonus() -> None:
    algo = NativeLatentImZero(_DiscreteEnv(), TINY, seed=0, device="cpu")
    _set_probe_predictions(algo, [-1.0, 1.0])
    _set_calibrated_sparse_state(algo)
    algo._uncertainty_zero_reward_fraction_ema = 0.0
    reward = torch.zeros(3)
    assert algo.uncertainty_sparsity_gate == 0.0
    assert torch.equal(
        algo._adjust_search_reward(torch.zeros(3, algo.embed_dim), reward),
        reward,
    )


def test_v7_checkpoint_roundtrip_and_v6_rejection(tmp_path: Path) -> None:
    algo = NativeLatentImZero(
        _DiscreteEnv(), {**TINY, "uncertainty_enabled": 1}, seed=3, device="cpu",
    )
    algo._uncertainty_ema_initialized = True
    algo._uncertainty_probe_loss_ema = 0.25
    algo._uncertainty_disagreement_ema = 0.75
    algo._uncertainty_zero_reward_fraction_ema = 0.8
    algo._num_timesteps = 12_345
    checkpoint = tmp_path / "v7.pt"
    algo.save(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["checkpoint_version"] == LATENTIMZERO_CHECKPOINT_VERSION
    assert payload["architecture"] == "researchimzero_core+search_uncertainty_probe"

    loaded = NativeLatentImZero.load(checkpoint, _DiscreteEnv(), device="cpu")
    _assert_state_equal(_shared_state(algo), _shared_state(loaded))
    _assert_state_equal(
        {"probe." + key: value for key, value in algo.uncertainty_probe.state_dict().items()},
        {"probe." + key: value for key, value in loaded.uncertainty_probe.state_dict().items()},
    )
    assert loaded.uncertainty_runtime_scale == algo.uncertainty_runtime_scale
    assert loaded._uncertainty_ema_initialized is True
    assert loaded._uncertainty_probe_loss_ema == 0.25
    assert loaded._uncertainty_disagreement_ema == 0.75
    assert loaded._uncertainty_zero_reward_fraction_ema == 0.8

    legacy_v7 = tmp_path / "early-v7.pt"
    for key in (
        "uncertainty_ema_initialized",
        "uncertainty_probe_loss_ema",
        "uncertainty_disagreement_ema",
        "uncertainty_zero_reward_fraction_ema",
        "last_uncertainty_normalized_mean",
        "last_uncertainty_bonus_mean",
        "last_uncertainty_bonus_max",
    ):
        payload.pop(key, None)
    torch.save(payload, legacy_v7)
    early_loaded = NativeLatentImZero.load(legacy_v7, _DiscreteEnv(), device="cpu")
    assert early_loaded._uncertainty_ema_initialized is False
    assert early_loaded.uncertainty_calibration == 0.0

    old = tmp_path / "v6.pt"
    torch.save({"checkpoint_version": 6, "world_model": {}, "actor": {}, "hyperparams": {}}, old)
    with pytest.raises(ValueError, match="checkpoint v6 is incompatible"):
        NativeLatentImZero.load(old, _DiscreteEnv(), device="cpu")


@pytest.mark.parametrize("env", [_DiscreteEnv(), _ContinuousEnv(), _ImageEnv()])
def test_inherited_search_smoke(env: gym.Env) -> None:
    params = {**TINY, "batch_size": 2}
    if isinstance(env.observation_space, gym.spaces.Box) and len(env.observation_space.shape) == 3:
        params["embed_dim"] = 16
    algo = NativeLatentImZero(env, params, seed=0, device="cpu")
    obs, _ = env.reset(seed=0)
    action, _ = algo.predict(obs, deterministic=True, episode_start=True)
    assert env.action_space.contains(action)


def test_probe_requires_at_least_two_members() -> None:
    assert _SearchUncertaintyProbe(8, members=1).members == 2


def test_backend_exposes_research_surface_plus_v7_toggles() -> None:
    entries = {entry["id"]: entry for entry in ALGORITHM_CATALOG}
    research = entries["researchimzero"]
    latent = entries["latentimzero"]
    research_params = {item["key"]: item["default"] for item in research["hyperparams"]}
    latent_params = {item["key"]: item["default"] for item in latent["hyperparams"]}
    assert {key: latent_params[key] for key in research_params} == research_params
    assert latent["name"].startswith("LatentImZero v7")
    assert latent_params["force_research_mode"] == 0
    assert latent_params["uncertainty_enabled"] == 1
