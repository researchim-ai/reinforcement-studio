from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from backend.routes.environments import ALGORITHM_CATALOG
import rl_core.algorithms.native.latentimzero as latentimzero_module
from rl_core.algorithms.base import TrainingCallback
from rl_core.algorithms.native.latentimzero import (
    DEFAULT_HYPERPARAMS,
    LATENTIMZERO_CHECKPOINT_VERSION,
    NativeLatentImZero,
    _SearchUncertaintyProbe,
)
from rl_core.algorithms.native.researchimzero import (
    DEFAULT_HYPERPARAMS as RESEARCH_DEFAULTS,
    NativeResearchImZero,
    _ResearchImZeroBuffer,
    _SearchNode,
    _signed_hyperbolic,
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


class _TwoStepEnv(_DiscreteEnv):
    def reset(self, *, seed=None, options=None):
        self._step = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        self._step += 1
        return np.zeros(4, dtype=np.float32), 1.0, self._step >= 2, False, {}


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


def test_v10_inherits_research_and_only_overrides_latent_features() -> None:
    assert issubclass(NativeLatentImZero, NativeResearchImZero)
    active_overrides = {
        "adaptive_train_steps", "replay_success_fraction", "adaptive_closed_loop",
        "path_consistency_coef", "learning_progress_priority_weight",
    }
    assert {
        key: DEFAULT_HYPERPARAMS[key]
        for key in RESEARCH_DEFAULTS
        if key not in active_overrides
    } == {
        key: value for key, value in RESEARCH_DEFAULTS.items() if key not in active_overrides
    }
    assert RESEARCH_DEFAULTS["adaptive_train_steps"] == 0
    assert RESEARCH_DEFAULTS["replay_success_fraction"] == 0.0
    assert RESEARCH_DEFAULTS["adaptive_closed_loop"] == 0
    assert DEFAULT_HYPERPARAMS["adaptive_train_steps"] == 1
    assert DEFAULT_HYPERPARAMS["replay_success_fraction"] == 0.15
    assert DEFAULT_HYPERPARAMS["adaptive_closed_loop"] == 1
    assert DEFAULT_HYPERPARAMS["embed_dim"] == 128
    assert DEFAULT_HYPERPARAMS["batch_size"] == 64
    assert DEFAULT_HYPERPARAMS["train_steps_per_iter"] == 2
    assert DEFAULT_HYPERPARAMS["num_simulations"] == 32
    assert DEFAULT_HYPERPARAMS["learning_starts"] == 500
    assert DEFAULT_HYPERPARAMS["use_amp"] == 1
    assert DEFAULT_HYPERPARAMS["uncertainty_enabled"] == 1
    assert DEFAULT_HYPERPARAMS["voc_enabled"] == 1
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


def test_v10_full_discrete_shadow_matches_research_max_planner() -> None:
    params = {
        **TINY,
        "num_simulations": 8,
        "num_simulations_initial": 8,
        "num_top_actions": 3,
    }
    research = NativeResearchImZero(_DiscreteEnv(), params, seed=17, device="cpu")
    latent = NativeLatentImZero(
        _DiscreteEnv(),
        {
            **params,
            "voc_label_sample_fraction": 0.0,
            "voc_random_audit_fraction": 0.0,
            "voc_targeted_audit_fraction": 0.0,
        },
        seed=17,
        device="cpu",
    )
    obs = np.zeros((2, 4), dtype=np.float32)
    np.random.seed(91)
    expected = research.search(
        obs, [None, None], deterministic=np.array([True, True]),
    )
    np.random.seed(91)
    actual = latent.search(
        obs, [None, None], deterministic=np.array([True, True]),
    )
    for expected_lane, actual_lane in zip(expected, actual):
        assert actual_lane["env_action"] == expected_lane["env_action"]
        assert np.array_equal(
            actual_lane["policy_target"],
            expected_lane["policy_target"],
        )
        assert actual_lane["value_target"] == expected_lane["value_target"]


def test_v10_1_voc_uses_fixed_canonical_max_despite_model_error() -> None:
    params = {
        **TINY,
        "num_simulations": 32,
        "num_simulations_initial": 8,
        "num_top_actions": 3,
        "voc_budgets": [0, 4, 16, 32],
        "voc_random_audit_fraction": 1.0,
        "voc_targeted_audit_fraction": 0.0,
        "voc_label_sample_fraction": 0.0,
        "voc_uncertainty_filter": 1.0,
    }
    algo = NativeLatentImZero(_DiscreteEnv(), params, seed=0, device="cpu")
    algo._num_timesteps = algo.search_ramp_steps
    progress_for_29 = (29 - 8) / (32 - 8)
    algo._model_error_ema = algo.search_model_error_high - progress_for_29 * (
        algo.search_model_error_high - algo.search_model_error_low
    )
    assert algo._current_num_simulations() == 29
    with torch.no_grad():
        for parameter in algo.uncertainty_probe.parameters():
            parameter.zero_()
    result = algo.search(np.zeros((1, 4), dtype=np.float32), [None])[0]
    assert algo.voc_budgets == [0, 4, 16, 32]
    assert result["requested_budget"] == 32
    assert result["executed_search_expansions"] == 32
    assert {item["edge_id"] for item in algo._voc_replay} == {
        "actor_to_small",
        "small_to_medium",
        "medium_to_max",
    }
    assert {item["edge_id"] for item in algo._voc_audits} == {
        "actor_to_small",
        "small_to_medium",
        "medium_to_max",
    }

    no_voc = NativeLatentImZero(
        _DiscreteEnv(), {**params, "voc_enabled": 0}, seed=0, device="cpu",
    )
    no_voc._num_timesteps = no_voc.search_ramp_steps
    no_voc._model_error_ema = algo._model_error_ema
    assert no_voc._current_num_simulations() == 29
    research = NativeResearchImZero(_DiscreteEnv(), params, seed=0, device="cpu")
    research._num_timesteps = research.search_ramp_steps
    research._model_error_ema = algo._model_error_ema
    assert research._current_num_simulations() == 29


def test_v10_1_custom_ladder_has_stable_generic_stage_ids() -> None:
    algo = NativeLatentImZero(
        _DiscreteEnv(),
        {
            **TINY,
            "num_simulations": 9,
            "voc_budgets": [0, 3, 3, 7, 100],
        },
        seed=0,
        device="cpu",
    )
    assert algo.voc_budgets == [0, 3, 7, 9]
    assert algo.voc_stage_ids == ["stage_0", "stage_1", "stage_2", "stage_3"]
    assert algo.voc_edge_ids == [
        "stage_0_to_stage_1",
        "stage_1_to_stage_2",
        "stage_2_to_stage_3",
    ]


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
    assert latent.voc_optimizer.state_dict()["state"] == {}
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
    metrics = algo._after_core_train_step(
        batch, hidden, torch.zeros(2, dtype=torch.long), torch.zeros(2, 2),
    )
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
        for head in algo.uncertainty_probe.value_heads:
            for parameter in head.parameters():
                parameter.zero_()


def _set_calibrated_return_state(algo: NativeLatentImZero) -> None:
    algo._uncertainty_ema_initialized = True
    algo._uncertainty_probe_loss_ema = 0.0
    algo._uncertainty_disagreement_ema = 1.0
    algo._uncertainty_surprise_ema = 1.0
    algo._uncertainty_zero_reward_fraction_ema = 1.0
    algo._num_timesteps = algo.uncertainty_warmup_steps + algo.uncertainty_ramp_steps


def test_warmup_keeps_reward_exact_but_trains_probe() -> None:
    hidden = torch.randn(4, 16)
    reward = torch.randn(4)
    algo = NativeLatentImZero(_DiscreteEnv(), TINY, seed=0, device="cpu")
    assert algo.uncertainty_runtime_scale == 0.0
    adjusted, next_value = algo._adjust_search_predictions(
        hidden, hidden, reward, torch.zeros_like(reward),
    )
    assert adjusted is reward
    assert torch.equal(next_value, torch.zeros_like(reward))
    before = {
        key: value.clone() for key, value in algo.uncertainty_probe.state_dict().items()
    }
    batch = {
        "reward": np.array([[0.0, 1e6], [0.0, 0.0]], dtype=np.float32),
        "mask": np.ones((2, 2), dtype=np.float32),
    }
    metrics = algo._after_core_train_step(
        batch, torch.randn(2, 5, algo.embed_dim), torch.zeros(2, dtype=torch.long),
        torch.zeros(2, 2),
    )
    assert "uncertainty_aux_loss" in metrics
    assert metrics["uncertainty_schedule_scale"] == 0.0
    assert any(
        not torch.equal(value, algo.uncertainty_probe.state_dict()[key])
        for key, value in before.items()
    )


def test_dense_surprise_bonus_is_positive_and_bounded_after_warmup() -> None:
    algo = NativeLatentImZero(_DiscreteEnv(), TINY, seed=0, device="cpu")
    _set_probe_predictions(algo, [-1.0, 1.0])
    _set_calibrated_return_state(algo)
    reward = torch.zeros(4)
    adjusted, _ = algo._adjust_search_predictions(
        torch.zeros(4, algo.embed_dim), torch.zeros(4, algo.embed_dim),
        reward, torch.zeros_like(reward),
    )
    bonus = adjusted - reward
    assert torch.all(bonus > 0)
    assert float(bonus.max()) <= algo.uncertainty_bonus_coef * algo.uncertainty_scale


def test_identical_probe_predictions_give_zero_bonus() -> None:
    algo = NativeLatentImZero(_DiscreteEnv(), TINY, seed=0, device="cpu")
    _set_probe_predictions(algo, [0.25, 0.25])
    _set_calibrated_return_state(algo)
    reward = torch.zeros(3)
    assert torch.equal(
        algo._adjust_search_predictions(
            torch.zeros(3, algo.embed_dim), torch.zeros(3, algo.embed_dim),
            reward, torch.zeros_like(reward),
        )[0],
        reward,
    )


def test_search_bonus_uses_full_return_value_disagreement() -> None:
    algo = NativeLatentImZero(_DiscreteEnv(), TINY, seed=0, device="cpu")
    _set_probe_predictions(algo, [0.0, 0.0])
    with torch.no_grad():
        for head, bias in zip(algo.uncertainty_probe.value_heads, [-1.0, 1.0]):
            head[-1].bias.fill_(bias)
    _set_calibrated_return_state(algo)
    reward = torch.zeros(3)
    adjusted, _ = algo._adjust_search_predictions(
        torch.zeros(3, algo.embed_dim), torch.zeros(3, algo.embed_dim),
        reward, torch.zeros_like(reward),
    )
    assert torch.all(adjusted > reward)
    assert float((adjusted - reward).max()) <= algo.uncertainty_bonus_coef


def test_dense_reward_diagnostic_does_not_suppress_surprise_bonus() -> None:
    algo = NativeLatentImZero(_DiscreteEnv(), TINY, seed=0, device="cpu")
    _set_probe_predictions(algo, [-1.0, 1.0])
    _set_calibrated_return_state(algo)
    algo._uncertainty_zero_reward_fraction_ema = 0.0
    reward = torch.zeros(3)
    adjusted, _ = algo._adjust_search_predictions(
        torch.zeros(3, algo.embed_dim), torch.zeros(3, algo.embed_dim),
        reward, torch.zeros_like(reward),
    )
    assert torch.all(adjusted > reward)


def test_v11_checkpoint_roundtrip_v10_v9_migration_and_v6_rejection(tmp_path: Path) -> None:
    algo = NativeLatentImZero(
        _DiscreteEnv(), {**TINY, "uncertainty_enabled": 1}, seed=3, device="cpu",
    )
    algo._uncertainty_ema_initialized = True
    algo._uncertainty_probe_loss_ema = 0.25
    algo._uncertainty_disagreement_ema = 0.75
    algo._uncertainty_surprise_ema = 0.5
    algo._uncertainty_zero_reward_fraction_ema = 0.8
    algo._num_timesteps = 12_345
    algo._adaptive_closed_loop_horizon = 2
    algo._adaptive_closed_loop_stable_count = 7
    algo._closed_loop_error_ema[:2] = [0.04, 0.05]
    algo._voc_replay.append(
        {
            "root_hidden": torch.zeros(algo.embed_dim),
            "scalars": torch.zeros(7),
            "label": 1.0,
            "raw_gain": 0.2,
            "valid": True,
            "confidence": 0.9,
            "stage": 0,
            "current_budget": 0,
            "next_budget": 2,
        }
    )
    algo._voc_audits.append(
        {
            "regret": 0.01,
            "divergence": 0.02,
            "probability": 0.8,
            "label": 1.0,
            "confident_mistake": 0.0,
            "stage": 0,
            "current_budget": 0,
            "next_budget": 2,
        }
    )
    algo._voc_shadow_mode = False
    algo._voc_minimum_budget_index = 0
    algo._voc_update_count = 3
    policy = np.full(algo.n_actions, 1.0 / algo.n_actions, dtype=np.float32)
    algo.buffer.add(
        np.zeros(4, dtype=np.float32),
        np.eye(algo.n_actions, dtype=np.float32)[0],
        1.0,
        np.ones(4, dtype=np.float32),
        policy,
        False,
        stage_index=1,
        stage_id="max",
        nominal_stage_budget=2,
        requested_budget=2,
        executed_search_expansions=2,
        common_evaluator_expansions=1,
        total_model_calls=3,
        termination_reason="budget_exhausted",
    )
    checkpoint = tmp_path / "v11.pt"
    algo.save(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["checkpoint_version"] == LATENTIMZERO_CHECKPOINT_VERSION
    assert payload["architecture"] == (
        "researchimzero_core+return_uncertainty_probe+voc_anytime_v10_1"
    )

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
    assert loaded._uncertainty_surprise_ema == 0.5
    assert loaded._uncertainty_zero_reward_fraction_ema == 0.8
    assert loaded._adaptive_closed_loop_horizon == 2
    assert loaded._adaptive_closed_loop_stable_count == 7
    assert loaded._closed_loop_error_ema[:2] == [0.04, 0.05]
    assert len(loaded._voc_replay) == len(loaded._voc_audits) == 1
    assert loaded._voc_shadow_mode is False
    assert loaded._voc_minimum_budget_index == 0
    assert loaded._voc_update_count == 3
    assert loaded.buffer._cur[0]["requested_budget"] == [2]
    assert loaded.buffer._cur[0]["common_evaluator_expansions"] == [1]
    _assert_state_equal(
        {"voc." + key: value for key, value in algo.voc_classifier.state_dict().items()},
        {"voc." + key: value for key, value in loaded.voc_classifier.state_dict().items()},
    )

    incompatible_v10 = tmp_path / "legacy-v10-labels.pt"
    incompatible_payload = dict(payload)
    incompatible_payload["checkpoint_version"] = 10
    incompatible_payload["voc_label_schema_version"] = 2
    torch.save(incompatible_payload, incompatible_v10)
    safely_reset = NativeLatentImZero.load(
        incompatible_v10, _DiscreteEnv(), device="cpu",
    )
    assert safely_reset._voc_shadow_mode is True
    assert safely_reset._voc_minimum_budget_index == len(safely_reset.voc_budgets) - 1
    assert len(safely_reset._voc_replay) == len(safely_reset._voc_audits) == 0
    assert safely_reset.buffer._cur[0]["total_model_calls"] == [3]
    _assert_state_equal(
        {"probe." + key: value for key, value in algo.uncertainty_probe.state_dict().items()},
        {
            "probe." + key: value
            for key, value in safely_reset.uncertainty_probe.state_dict().items()
        },
    )

    legacy_v7 = tmp_path / "legacy-v9.pt"
    payload["checkpoint_version"] = 9
    for key in list(payload):
        if key.startswith("voc_") or key in {
            "last_voc_metrics", "executed_budget_history", "model_expansion_history",
        }:
            payload.pop(key, None)
    for key in (
        "uncertainty_ema_initialized",
        "uncertainty_probe_loss_ema",
        "uncertainty_disagreement_ema",
        "uncertainty_surprise_ema",
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


def _set_voc_stop(algo: NativeLatentImZero) -> None:
    with torch.no_grad():
        for parameter in algo.voc_classifier.parameters():
            parameter.zero_()
        algo.voc_classifier.net[-1].bias.fill_(-20.0)


def test_v10_budget_sanitization_b0_and_policy_mask() -> None:
    algo = NativeLatentImZero(
        _DiscreteEnv(),
        {
            **TINY,
            "voc_budgets": [16, -2, 4, 4, 999],
            "voc_random_audit_fraction": 0.0,
            "voc_targeted_audit_fraction": 0.0,
        },
        seed=0,
        device="cpu",
    )
    assert algo.voc_budgets == [0, 2]
    algo._voc_shadow_mode = False
    algo._voc_minimum_budget_index = 0
    _set_voc_stop(algo)
    calls: list[int] = []
    original = algo._step_imagine

    def counted(caches, action):
        calls.append(len(caches))
        return original(caches, action)

    algo._step_imagine = counted
    result = algo.search(np.zeros((1, 4), dtype=np.float32), [None])[0]
    assert calls == []
    assert result["search_budget"] == result["model_expansions"] == 0
    assert result["stage_index"] == result["predicted_stage_index"] == 0
    assert result["requested_budget"] == result["nominal_stage_budget"] == 0
    assert result["executed_search_expansions"] == 0
    assert result["common_evaluator_expansions"] == result["total_model_calls"] == 0
    assert result["termination_reason"] == "controller_stop"
    assert result["policy_target_valid"] is False
    algo.buffer.add(
        np.zeros(4, dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        1.0,
        np.zeros(4, dtype=np.float32),
        result["policy_target"],
        True,
        policy_target_valid=result["policy_target_valid"],
        search_budget=result["search_budget"],
        model_expansions=result["model_expansions"],
        behavior_policy=result["behavior_policy"],
        stage_index=result["stage_index"],
        stage_id=result["stage_id"],
        nominal_stage_budget=result["nominal_stage_budget"],
        requested_budget=result["requested_budget"],
        predicted_stage_index=result["predicted_stage_index"],
        predicted_stage_id=result["predicted_stage_id"],
        predicted_requested_budget=result["predicted_requested_budget"],
        executed_search_expansions=result["executed_search_expansions"],
        common_evaluator_expansions=result["common_evaluator_expansions"],
        total_model_calls=result["total_model_calls"],
        termination_reason=result["termination_reason"],
    )
    sampled = algo.buffer.sample(1, 1, 1, algo.gamma)
    assert sampled["policy_target_valid"][0, 0] == 0.0
    assert sampled["search_budget"][0, 0] == sampled["model_expansions"][0, 0] == 0
    assert np.array_equal(sampled["behavior_policy"][0, 0], result["behavior_policy"])
    assert sampled["audit_type"][0, 0] == "none"
    assert sampled["stage_index"][0, 0] == 0
    assert sampled["termination_reason"][0, 0] == "controller_stop"


def test_v10_stopped_lanes_leave_batched_expansion_loop() -> None:
    algo = NativeLatentImZero(
        _DiscreteEnv(),
        {
            **TINY,
            "num_simulations": 4,
            "num_simulations_initial": 4,
            "voc_budgets": [0, 2, 4],
            "voc_label_sample_fraction": 1.0,
            "voc_random_audit_fraction": 0.0,
            "voc_targeted_audit_fraction": 0.0,
        },
        seed=0,
        device="cpu",
    )
    algo._voc_shadow_mode = False
    algo._voc_minimum_budget_index = 1
    predictions = iter([
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
    ])
    algo._voc_predict_batch = lambda *_: next(predictions)
    batch_sizes: list[int] = []
    original = algo._step_imagine

    def counted(caches, action):
        batch_sizes.append(len(caches))
        return original(caches, action)

    algo._step_imagine = counted
    results = algo.search(np.zeros((2, 4), dtype=np.float32), [None, None])
    assert batch_sizes[:4] == [2, 2, 1, 1]
    assert batch_sizes[4:] == [results[1]["common_evaluator_expansions"]]
    assert [result["search_budget"] for result in results] == [2, 4]
    assert results[0]["model_expansions"] == 2
    assert results[1]["model_expansions"] == (
        4 + results[1]["common_evaluator_expansions"]
    )
    assert results[0]["termination_reason"] == "safety_floor"
    assert results[1]["termination_reason"] == "budget_exhausted"


def test_v10_shadow_executes_max_and_logs_predicted_budget() -> None:
    algo = NativeLatentImZero(
        _DiscreteEnv(),
        {
            **TINY,
            "num_simulations": 4,
            "num_simulations_initial": 4,
            "voc_budgets": [0, 2, 4],
            "voc_label_sample_fraction": 1.0,
            "voc_random_audit_fraction": 0.0,
            "voc_targeted_audit_fraction": 0.0,
        },
        seed=0,
        device="cpu",
    )
    _set_voc_stop(algo)
    result = algo.search(np.zeros((1, 4), dtype=np.float32), [None])[0]
    assert result["search_budget"] == 4
    assert result["model_expansions"] == 4 + result["common_evaluator_expansions"]
    assert result["common_evaluator_expansions"] > 0
    assert result["predicted_requested_budget"] == 0
    assert result["requested_budget"] == result["executed_search_expansions"] == 4
    assert result["total_model_calls"] == result["model_expansions"]
    assert result["termination_reason"] == "budget_exhausted"
    assert algo._last_voc_search_metrics["shadow_predicted_budget_mean"] == 0.0
    assert algo._last_voc_search_metrics["requested_budget_mean"] == 4.0
    assert algo._last_voc_search_metrics["executed_search_expansions_mean"] == 4.0
    assert algo._last_voc_search_metrics["termination_fraction_budget_exhausted"] == 1.0
    for checkpoint in result["budget_checkpoints"].values():
        assert checkpoint["representative_action"] == int(np.argmax(checkpoint["policy"]))


def test_v10_common_evaluator_avoids_checkpoint_self_score_winner_curse() -> None:
    common_q = np.array([0.9, 0.5, 0.7])
    labels = NativeLatentImZero._common_voc_labels(
        common_q.tolist(), [0, 4, 16], 0.0, 0.0, 1.0,
    )
    assert [item[0] for item in labels] == pytest.approx([-0.4, 0.2])
    assert [item[1] for item in labels] == [0.0, 1.0]


def test_v10_common_ema_evaluator_is_invariant_to_tree_visit_counts() -> None:
    algo = NativeLatentImZero(_DiscreteEnv(), TINY, seed=0, device="cpu")
    root = _SearchNode(1.0)
    root.expand(np.zeros(algo.n_actions), None, 0.0)
    root.visit_count = 1
    actions = [{0: 0, 1: 1, 2: 0}]
    first = algo._common_one_step_evaluator([root], actions, np.array([True]))[0]
    first_labels = algo._common_voc_labels(
        [first["q"][budget] for budget in sorted(actions[0])],
        [0, 1, 2],
        0.01,
        0.0,
        algo.voc_return_scale,
    )
    root.children[0].visit_count = 1000
    root.children[0].value_sum = -10_000.0
    root.children[1].visit_count = 1
    root.children[1].value_sum = 10_000.0
    second = algo._common_one_step_evaluator([root], actions, np.array([True]))[0]
    second_labels = algo._common_voc_labels(
        [second["q"][budget] for budget in sorted(actions[0])],
        [0, 1, 2],
        0.01,
        0.0,
        algo.voc_return_scale,
    )
    assert first["q"] == second["q"]
    assert first_labels == second_labels
    assert first["unique_actions"] == 2


def test_v10_targeted_audits_replenish_random_collisions() -> None:
    algo = NativeLatentImZero(
        _DiscreteEnv(),
        {
            **TINY,
            "voc_random_audit_fraction": 0.5,
            "voc_targeted_audit_fraction": 0.25,
        },
        seed=0,
        device="cpu",
    )
    audit_types = algo._sample_audit_types(
        np.array([0.1, 0.9, 0.8]),
        random_draws=np.array([0.1, 0.9, 0.9]),
        targeted_count=2,
    )
    assert audit_types == ["random", "targeted", "targeted"]


def test_v10_batch1_targeted_audit_has_unconditional_configured_rate() -> None:
    algo = NativeLatentImZero(
        _DiscreteEnv(),
        {
            **TINY,
            "voc_random_audit_fraction": 0.2,
            "voc_targeted_audit_fraction": 0.3,
        },
        seed=0,
        device="cpu",
    )
    audit_types = [
        algo._sample_audit_types(np.array([0.5]))[0]
        for _ in range(20_000)
    ]
    assert audit_types.count("random") / len(audit_types) == pytest.approx(0.2, abs=0.015)
    assert audit_types.count("targeted") / len(audit_types) == pytest.approx(0.3, abs=0.015)


def test_v10_label_sampling_always_includes_audits_and_controls_shadow_cost() -> None:
    algo = NativeLatentImZero(
        _DiscreteEnv(),
        {
            **TINY,
            "voc_label_sample_fraction": 0.25,
        },
        seed=0,
        device="cpu",
    )
    full = np.ones(8, dtype=bool)
    draws = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    quarter = algo._sample_label_lanes(
        full,
        ["none"] * 8,
        draws=draws,
    )
    assert quarter.tolist() == [True, True, False, False, False, False, False, False]
    algo.voc_label_sample_fraction = 1.0
    full_rate = algo._sample_label_lanes(full, ["none"] * 8, draws=draws)
    assert full_rate.sum() == 4 * quarter.sum()

    algo.voc_label_sample_fraction = 0.0
    audits_only = algo._sample_label_lanes(
        full,
        ["none", "random", "none", "targeted", "none", "none", "none", "none"],
        draws=np.zeros(8),
    )
    assert audits_only.tolist() == [False, True, False, True, False, False, False, False]


def test_v10_fraction_zero_evaluates_audits_but_not_plain_shadow() -> None:
    base = {
        **TINY,
        "voc_label_sample_fraction": 0.0,
        "voc_targeted_audit_fraction": 0.0,
    }
    plain = NativeLatentImZero(
        _DiscreteEnv(),
        {**base, "voc_random_audit_fraction": 0.0},
        seed=0,
        device="cpu",
    )
    plain_result = plain.search(np.zeros((1, 4), dtype=np.float32), [None])[0]
    assert plain_result["common_evaluator_expansions"] == 0
    assert plain._last_voc_search_metrics["voc_label_sample_fraction_actual"] == 0.0

    audited = NativeLatentImZero(
        _DiscreteEnv(),
        {**base, "voc_random_audit_fraction": 1.0},
        seed=0,
        device="cpu",
    )
    audited_result = audited.search(np.zeros((1, 4), dtype=np.float32), [None])[0]
    assert audited_result["audit_type"] == "random"
    assert audited_result["common_evaluator_expansions"] > 0
    assert audited._last_voc_search_metrics["voc_label_sample_fraction_actual"] == 1.0


def test_v10_voc_classifier_inference_is_batched_per_stage() -> None:
    algo = NativeLatentImZero(
        _DiscreteEnv(),
        {
            **TINY,
            "num_simulations": 8,
            "num_simulations_initial": 8,
            "voc_budgets": [0, 4, 8],
            "voc_label_sample_fraction": 0.0,
            "voc_random_audit_fraction": 0.0,
            "voc_targeted_audit_fraction": 0.0,
        },
        seed=0,
        device="cpu",
    )
    batch_sizes: list[int] = []
    original = algo._voc_predict_batch

    def counted(hidden, scalars):
        batch_sizes.append(hidden.shape[0])
        return original(hidden, scalars)

    algo._voc_predict_batch = counted
    algo.search(np.zeros((4, 4), dtype=np.float32), [None] * 4)
    assert batch_sizes == [4, 4]


def test_v10_reanalysis_unmasks_fresh_policy_target_active_and_finalized() -> None:
    active = NativeLatentImZero(_DiscreteEnv(), TINY, seed=0, device="cpu")
    active.buffer.add(
        np.zeros(4),
        np.array([1.0, 0.0, 0.0]),
        1.0,
        np.zeros(4),
        np.full(3, 1 / 3),
        False,
        policy_target_valid=False,
    )
    active.buffer.update_reanalyzed_targets(
        np.array([-1]),
        np.array([0]),
        np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
        np.array([0.5], dtype=np.float32),
        np.array([True]),
    )
    assert active.buffer._cur[0]["policy_target_valid"][0] is True

    finalized = NativeLatentImZero(_DiscreteEnv(), TINY, seed=0, device="cpu")
    finalized.buffer.add(
        np.zeros(4),
        np.array([1.0, 0.0, 0.0]),
        1.0,
        np.zeros(4),
        np.full(3, 1 / 3),
        True,
        policy_target_valid=False,
    )
    before = finalized.buffer.sample(1, 1, 1, finalized.gamma)
    assert before["policy_target_valid"][0, 0] == 0.0
    finalized.buffer.update_reanalyzed_targets(
        np.array([0]),
        np.array([0]),
        np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
        np.array([0.5], dtype=np.float32),
        np.array([True]),
    )
    after = finalized.buffer.sample(1, 1, 1, finalized.gamma)
    assert after["policy_target_valid"][0, 0] == 1.0
    assert finalized._train_step()["policy_loss"] > 0.0


def test_v10_eval_and_reanalysis_preserve_collection_diagnostics_and_voc_state() -> None:
    algo = NativeLatentImZero(_DiscreteEnv(), TINY, seed=0, device="cpu")
    _set_calibrated_return_state(algo)
    diagnostics = (0.12, 0.34, 0.56)
    (
        algo._last_uncertainty_normalized_mean,
        algo._last_uncertainty_bonus_mean,
        algo._last_uncertainty_bonus_max,
    ) = diagnostics
    state = (
        len(algo._voc_replay),
        len(algo._voc_audits),
        algo._voc_minimum_budget_index,
        list(algo._voc_safety_streaks),
    )
    obs = np.zeros((1, 4), dtype=np.float32)
    algo.search(obs, [None], deterministic=np.array([True]))
    assert (
        algo._last_uncertainty_normalized_mean,
        algo._last_uncertainty_bonus_mean,
        algo._last_uncertainty_bonus_max,
    ) == diagnostics
    assert (
        len(algo._voc_replay),
        len(algo._voc_audits),
        algo._voc_minimum_budget_index,
        algo._voc_safety_streaks,
    ) == state
    algo.search(obs, [None], deterministic=np.array([False]))
    assert (
        algo._last_uncertainty_normalized_mean,
        algo._last_uncertainty_bonus_mean,
        algo._last_uncertainty_bonus_max,
    ) == diagnostics
    assert (
        len(algo._voc_replay),
        len(algo._voc_audits),
        algo._voc_minimum_budget_index,
        algo._voc_safety_streaks,
    ) == state


def test_v10_continuous_candidate_zero_is_actor_mean() -> None:
    algo = NativeLatentImZero(_ContinuousEnv(), TINY, seed=0, device="cpu")
    mean = np.array([0.25, -0.1], dtype=np.float32)
    candidates, priors = algo._continuous_anytime_candidates(
        mean, np.array([0.2, 0.1], dtype=np.float32),
    )
    assert np.array_equal(candidates[0], mean)
    assert len(candidates) == len(priors) == algo.num_sampled_actions
    result = algo.search(np.zeros((1, 5), dtype=np.float32), [None])[0]
    for checkpoint in result["budget_checkpoints"].values():
        assert np.array_equal(
            checkpoint["representative_action"],
            checkpoint["policy"],
        )


def test_v10_voc_sidecar_cannot_backprop_into_core() -> None:
    algo = NativeLatentImZero(_DiscreteEnv(), TINY, seed=0, device="cpu")
    for parameter in algo._params:
        parameter.grad = None
    hidden = torch.randn(4, algo.embed_dim, requires_grad=True)
    scalars = torch.randn(4, 7)
    loss = algo.voc_classifier(hidden, scalars).sum()
    loss.backward()
    assert hidden.grad is None
    assert all(parameter.grad is None for parameter in algo._params)
    assert any(parameter.grad is not None for parameter in algo.voc_classifier.parameters())


def _safety_algo(**overrides) -> NativeLatentImZero:
    return NativeLatentImZero(
        _DiscreteEnv(),
        {
            **TINY,
            "num_simulations": 8,
            "voc_budgets": [0, 2, 4, 8],
            "voc_min_labels": 1,
            "voc_min_audits": 1,
            "voc_min_stop_audits": 1,
            "voc_min_class_samples": 1,
            "voc_safety_patience": 1,
            "voc_min_stop_precision": 0.5,
            **overrides,
        },
        seed=0,
        device="cpu",
    )


def _add_stage_evidence(
    algo: NativeLatentImZero,
    stage: int,
    *,
    safe: bool,
) -> None:
    current_budget, next_budget = algo.voc_budgets[stage: stage + 2]
    label = 0.0 if safe else 1.0
    probability = 0.1
    algo._voc_replay.append(
        {
            "valid": True,
            "label": label,
            "confidence": 1.0,
            "stage": stage,
            "edge_index": stage,
            "current_budget": current_budget,
            "next_budget": next_budget,
        }
    )
    algo._voc_replay.append(
        {
            "valid": True,
            "label": 1.0 - label,
            "confidence": 1.0,
            "stage": stage,
            "edge_index": stage,
            "current_budget": current_budget,
            "next_budget": next_budget,
        }
    )
    algo._voc_audits.append(
        {
            "stage": stage,
            "edge_index": stage,
            "current_budget": current_budget,
            "next_budget": next_budget,
            "regret": 0.0 if safe else 10.0,
            "divergence": 0.0 if safe else 10.0,
            "probability": probability,
            "label": label,
            "confident_mistake": float(not safe),
        }
    )


def test_v10_stage0_evidence_cannot_unlock_later_stages() -> None:
    algo = _safety_algo()
    _add_stage_evidence(algo, 0, safe=True)
    initial = algo._voc_minimum_budget_index
    algo._update_voc_safety()
    assert algo._voc_minimum_budget_index == initial
    assert algo._voc_safety_streaks[2] == 0


def test_v10_each_stage_unlocks_only_from_its_own_audits() -> None:
    algo = _safety_algo()
    _add_stage_evidence(algo, 2, safe=True)
    algo._update_voc_safety()
    assert algo._voc_minimum_budget_index == 2
    algo._update_voc_safety()
    assert algo._voc_minimum_budget_index == 2
    _add_stage_evidence(algo, 1, safe=True)
    algo._update_voc_safety()
    assert algo._voc_minimum_budget_index == 1
    algo._update_voc_safety()
    assert algo._voc_minimum_budget_index == 1
    _add_stage_evidence(algo, 0, safe=True)
    algo._update_voc_safety()
    assert algo._voc_minimum_budget_index == 0


def test_v10_active_stage_regression_relocks_one_level() -> None:
    algo = _safety_algo()
    for stage in (2, 1):
        _add_stage_evidence(algo, stage, safe=True)
        algo._update_voc_safety()
    assert algo._voc_minimum_budget_index == 1
    _add_stage_evidence(algo, 2, safe=False)
    algo._update_voc_safety()
    assert algo._voc_minimum_budget_index == 3


@pytest.mark.parametrize(("regressed_stage", "expected_floor"), [(1, 2), (2, 3)])
def test_v10_1_relock_covers_the_regressed_edge(
    regressed_stage: int,
    expected_floor: int,
) -> None:
    algo = _safety_algo()
    algo._voc_shadow_mode = False
    algo._voc_minimum_budget_index = 0
    _add_stage_evidence(algo, regressed_stage, safe=False)
    algo._update_voc_safety()
    assert algo._voc_minimum_budget_index == expected_floor


def test_v10_1_relock_chooses_most_conservative_of_multiple_edges() -> None:
    algo = _safety_algo()
    algo._voc_shadow_mode = False
    algo._voc_minimum_budget_index = 0
    _add_stage_evidence(algo, 1, safe=False)
    _add_stage_evidence(algo, 2, safe=False)
    algo._update_voc_safety()
    assert algo._voc_minimum_budget_index == 3


def test_v10_safety_conditions_regret_and_divergence_on_predicted_stops() -> None:
    algo = _safety_algo()
    stage = 2
    _add_stage_evidence(algo, stage, safe=True)
    current_budget, next_budget = algo.voc_budgets[stage: stage + 2]
    algo._voc_replay.append(
        {
            "valid": True,
            "label": 1.0,
            "confidence": 1.0,
            "stage": stage,
            "edge_index": stage,
            "current_budget": current_budget,
            "next_budget": next_budget,
        }
    )
    algo._voc_audits.append(
        {
            "stage": stage,
            "edge_index": stage,
            "current_budget": current_budget,
            "next_budget": next_budget,
            "regret": 10.0,
            "divergence": 10.0,
            "probability": 0.9,
            "label": 1.0,
            "confident_mistake": 0.0,
        }
    )
    summary = algo._voc_stage_summary(stage)
    assert summary["overall_regret"] == summary["overall_divergence"] == 5.0
    assert summary["regret"] == summary["divergence"] == 0.0
    algo._update_voc_safety()
    assert algo._voc_minimum_budget_index == 2

    algo._voc_audits.append(
        {
            "stage": stage,
            "edge_index": stage,
            "current_budget": current_budget,
            "next_budget": next_budget,
            "regret": 10.0,
            "divergence": 0.0,
            "probability": 0.1,
            "label": 0.0,
            "confident_mistake": 0.0,
        }
    )
    assert algo._voc_stage_summary(stage)["stop_precision"] == 1.0
    assert algo._voc_stage_summary(stage)["regret"] == 5.0
    algo._update_voc_safety()
    assert algo._voc_minimum_budget_index == 3


def test_v10_stage_without_predicted_stops_remains_unsafe() -> None:
    algo = _safety_algo()
    stage = 2
    current_budget, next_budget = algo.voc_budgets[stage: stage + 2]
    algo._voc_replay.append(
        {
            "valid": True,
            "label": 1.0,
            "confidence": 1.0,
            "stage": stage,
            "edge_index": stage,
            "current_budget": current_budget,
            "next_budget": next_budget,
        }
    )
    algo._voc_audits.append(
        {
            "stage": stage,
            "edge_index": stage,
            "current_budget": current_budget,
            "next_budget": next_budget,
            "regret": 10.0,
            "divergence": 10.0,
            "probability": 0.9,
            "label": 1.0,
            "confident_mistake": 0.0,
        }
    )
    summary = algo._voc_stage_summary(stage)
    assert summary["stop_precision"] == 0.0
    assert np.isinf(summary["regret"])
    algo._update_voc_safety()
    assert algo._voc_minimum_budget_index == 3


def test_v10_1_actor_tail_gate_is_stricter_than_medium() -> None:
    actor = _safety_algo()
    medium = _safety_algo()
    _add_stage_evidence(actor, 0, safe=True)
    _add_stage_evidence(medium, 2, safe=True)
    actor._voc_audits[-1]["regret"] = 0.03
    medium._voc_audits[-1]["regret"] = 0.03
    assert not actor._voc_stage_is_safe(0, actor._voc_stage_summary(0))
    assert medium._voc_stage_is_safe(2, medium._voc_stage_summary(2))


def test_v10_1_safe_stop_tail_can_unlock_with_mediocre_brier_and_divergence() -> None:
    algo = _safety_algo()
    _add_stage_evidence(algo, 2, safe=True)
    algo._voc_audits[-1]["probability"] = 0.49
    algo._voc_audits[-1]["divergence"] = 0.5
    summary = algo._voc_stage_summary(2)
    assert summary["brier"] > algo.voc_max_brier
    assert summary["divergence"] > algo.voc_max_policy_divergence
    assert algo._voc_stage_is_safe(2, summary)
    algo._update_voc_safety()
    assert algo._voc_minimum_budget_index == 2


def _real_run_like_medium_evidence(
    algo: NativeLatentImZero,
    *,
    catastrophic_count: int,
) -> None:
    stage = 2
    stop_label_count = 6_386
    continue_label_count = 169
    algo._voc_replay.extend(
        {"valid": True, "label": 0.0, "edge_index": stage}
        for _ in range(stop_label_count)
    )
    algo._voc_replay.extend(
        {"valid": True, "label": 1.0, "edge_index": stage}
        for _ in range(continue_label_count)
    )
    catastrophic_regret = 1.218444
    for index in range(334):
        if index < catastrophic_count:
            regret = catastrophic_regret
            label = 1.0
        elif index < catastrophic_count + 3:
            regret = 0.01
            label = 1.0
        else:
            regret = 0.0
            label = 0.0
        algo._voc_audits.append(
            {
                "edge_index": stage,
                "regret": regret,
                "divergence": 0.0,
                "probability": 0.1,
                "label": label,
                "confident_mistake": label,
            }
        )


def test_v10_1_real_run_like_medium_edge_unlocks_on_expected_regret() -> None:
    algo = NativeLatentImZero(
        _DiscreteEnv(),
        {
            **TINY,
            "num_simulations": 8,
            "voc_budgets": [0, 2, 4, 8],
            "voc_safety_patience": 2,
        },
        seed=0,
        device="cpu",
    )
    _real_run_like_medium_evidence(algo, catastrophic_count=5)
    summary = algo._voc_stage_summary(2)
    assert summary["labels"] == 6_555
    assert summary["audits"] == summary["stop_samples"] == 334
    assert summary["label_continue_count"] == 169
    assert summary["label_continue_fraction"] == pytest.approx(169 / 6_555)
    assert summary["stop_precision"] == pytest.approx(326 / 334)
    assert summary["premature_exceedance"] == pytest.approx(5 / 334)
    assert summary["premature_p95"] == 0.0
    assert summary["premature_mean_positive"] == pytest.approx(0.01833, abs=1e-5)
    assert summary["premature_positive_case_mean"] == pytest.approx(
        0.7652775,
        abs=1e-5,
    )
    assert algo._voc_stage_is_safe(2, summary)
    algo._update_voc_safety()
    assert algo._voc_minimum_budget_index == 3
    algo._update_voc_safety()
    assert algo._voc_minimum_budget_index == 2
    # The next stricter edge has no evidence and cannot unlock.
    algo._update_voc_safety()
    assert algo._voc_minimum_budget_index == 2


def test_v10_1_tail_thresholds_still_block_more_catastrophic_stops() -> None:
    algo = NativeLatentImZero(
        _DiscreteEnv(),
        {**TINY, "num_simulations": 8, "voc_budgets": [0, 2, 4, 8]},
        seed=0,
        device="cpu",
    )
    _real_run_like_medium_evidence(algo, catastrophic_count=20)
    summary = algo._voc_stage_summary(2)
    assert summary["premature_exceedance"] > algo.voc_max_premature_exceedance[2]
    assert summary["premature_p95"] > algo.voc_max_p95_premature_regret[2]
    assert not algo._voc_stage_is_safe(2, summary)


def _add_balanced_health_evidence(algo: NativeLatentImZero) -> None:
    for stage in range(len(algo.voc_edge_ids)):
        _add_stage_evidence(algo, stage, safe=True)


def test_v10_1_health_has_no_premature_shadow_desync_failure() -> None:
    algo = _safety_algo(voc_health_check_steps=10, voc_health_desync_window=3)
    _add_balanced_health_evidence(algo)
    algo._num_timesteps = 10
    algo._predicted_budget_history.extend([0, 0, 0])
    algo._requested_budget_history.extend([8, 8, 8])
    algo._permitted_budget_history.extend([8, 8, 8])
    algo._executed_search_history.extend([8, 8, 8])
    metrics = algo._voc_health_metrics()
    assert metrics["voc_health_status"] == 1.0
    assert metrics["voc_health_warming_up"] == 1.0
    assert metrics["voc_fail_fast_recommended"] == 0.0
    assert metrics["voc_health_shadow_prediction_gap"] == 1.0


def test_v10_1_health_detects_missing_edge_and_class_collapse() -> None:
    missing = _safety_algo(voc_health_check_steps=10, voc_health_desync_window=3)
    missing._num_timesteps = 10
    assert missing._voc_health_metrics()["voc_health_status"] == 2.0

    collapsed = _safety_algo(voc_health_check_steps=10, voc_health_desync_window=3)
    collapsed._num_timesteps = 10
    for stage in range(len(collapsed.voc_edge_ids)):
        collapsed._voc_replay.append(
            {"valid": True, "label": 0.0, "edge_index": stage}
        )
        collapsed._voc_audits.append(
            {
                "edge_index": stage,
                "regret": 0.0,
                "divergence": 0.0,
                "probability": 0.1,
                "label": 0.0,
                "confident_mistake": 0.0,
            }
        )
    metrics = collapsed._voc_health_metrics()
    assert metrics["voc_health_status"] == 3.0
    assert metrics["voc_health_activation_failure_class_collapse"] == 1.0


def test_v10_1_health_detects_budget_desync_and_unsafe_collapse() -> None:
    desync = _safety_algo(voc_health_check_steps=10, voc_health_desync_window=3)
    _add_balanced_health_evidence(desync)
    desync._num_timesteps = 10
    desync._predicted_budget_history.extend([0, 0, 0])
    desync._requested_budget_history.extend([8, 8, 8])
    desync._permitted_budget_history.extend([0, 0, 0])
    desync._executed_search_history.extend([8, 8, 8])
    desync._voc_shadow_mode = False
    desync._voc_minimum_budget_index = 0
    metrics = desync._voc_health_metrics()
    assert metrics["voc_health_status"] == 4.0
    assert metrics["voc_fail_fast_recommended"] == 1.0

    unsafe = _safety_algo(voc_health_check_steps=100, voc_health_desync_window=3)
    _add_stage_evidence(unsafe, 0, safe=False)
    unsafe._predicted_budget_history.extend([0, 0, 0])
    unsafe._requested_budget_history.extend([0, 0, 0])
    unsafe._permitted_budget_history.extend([0, 0, 0])
    unsafe._executed_search_history.extend([0, 0, 0])
    metrics = unsafe._voc_health_metrics()
    assert metrics["voc_health_status"] == 5.0
    assert metrics["voc_health_unsafe_compute_collapse"] == 1.0


def test_v10_1_health_accepts_intentional_safety_floor_clamp() -> None:
    algo = _safety_algo(voc_health_check_steps=10, voc_health_desync_window=3)
    _add_balanced_health_evidence(algo)
    algo._num_timesteps = 10
    algo._voc_shadow_mode = False
    algo._voc_minimum_budget_index = 3
    algo._predicted_budget_history.extend([0, 0, 0])
    algo._requested_budget_history.extend([8, 8, 8])
    algo._permitted_budget_history.extend([8, 8, 8])
    algo._executed_search_history.extend([8, 8, 8])
    metrics = algo._voc_health_metrics()
    assert metrics["voc_health_status"] == 0.0
    assert metrics["voc_fail_fast_recommended"] == 0.0


def test_probe_requires_at_least_two_members() -> None:
    assert _SearchUncertaintyProbe(8, members=1).members == 2


def test_backend_exposes_research_surface_plus_v10_toggles() -> None:
    entries = {entry["id"]: entry for entry in ALGORITHM_CATALOG}
    research = entries["researchimzero"]
    latent = entries["latentimzero"]
    research_params = {item["key"]: item["default"] for item in research["hyperparams"]}
    latent_params = {item["key"]: item["default"] for item in latent["hyperparams"]}
    assert {key: latent_params[key] for key in research_params} == research_params
    assert latent["name"].startswith("LatentImZero v10.1")
    assert latent_params["force_research_mode"] == 0
    assert latent_params["uncertainty_enabled"] == 1
    assert latent_params["voc_enabled"] == 1
    assert latent_params["voc_label_sample_fraction"] == 0.25
    assert latent_params["voc_min_class_samples"] == 64


def test_adaptive_replay_pressure_is_high_for_high_error() -> None:
    algo = NativeLatentImZero(
        _DiscreteEnv(), {**TINY, "adaptive_train_steps": 1}, seed=0, device="cpu",
    )
    algo._model_error_ema = algo.search_model_error_high
    assert algo._effective_train_steps_per_iter() == 4
    algo._model_error_ema = algo.search_model_error_low
    assert algo._effective_train_steps_per_iter() == 2
    forced = NativeLatentImZero(
        _DiscreteEnv(), {**TINY, "force_research_mode": 1}, seed=0, device="cpu",
    )
    assert forced._effective_train_steps_per_iter() == forced.train_steps_per_iter == 2


def _make_replay(success_fraction: float) -> _ResearchImZeroBuffer:
    return _ResearchImZeroBuffer(
        20, (1,), 2, 2, context_length=0, recent_fraction=0.0,
        success_fraction=success_fraction, success_top_quantile=0.25,
    )


def _add_return_episode(buffer: _ResearchImZeroBuffer, episode_return: float) -> None:
    buffer.add(
        np.zeros(1), np.array([1.0, 0.0]), episode_return, np.ones(1),
        np.array([0.5, 0.5]), True,
    )


def test_success_replay_fraction_one_samples_top_return_quantile() -> None:
    buffer = _make_replay(1.0)
    for episode_return in (0.0, 1.0, 2.0, 3.0):
        _add_return_episode(buffer, episode_return)
    batch = buffer.sample(32, 1, 1, 0.99)
    assert np.all(batch["reward"][:, 0] == 3.0)
    assert np.isfinite(batch["is_weight"]).all()
    assert float(batch["success_sample_fraction"]) == 1.0
    assert float(batch["success_pool_size"]) == 1.0


def test_success_fraction_zero_preserves_sampling_path() -> None:
    implicit = _make_replay(0.0)
    explicit = _make_replay(0.0)
    for episode_return in (0.0, 1.0, 2.0):
        _add_return_episode(implicit, episode_return)
        _add_return_episode(explicit, episode_return)
    np.random.seed(91)
    expected = implicit.sample(8, 1, 1, 0.99)
    np.random.seed(91)
    actual = explicit.sample(8, 1, 1, 0.99)
    assert np.array_equal(actual["episode_idx"], expected["episode_idx"])
    assert np.array_equal(actual["timestep"], expected["timestep"])
    assert np.array_equal(actual["is_weight"], expected["is_weight"])
    assert float(actual["success_sample_fraction"]) == 0.0


def test_adaptive_closed_loop_upgrades_after_stable_errors_and_caps_at_five() -> None:
    algo = NativeLatentImZero(
        _DiscreteEnv(),
        {
            **TINY,
            "unroll_steps": 5,
            "closed_loop_horizon": 3,
            "adaptive_closed_loop": 1,
            "adaptive_closed_loop_max_horizon": 8,
            "adaptive_closed_loop_min_grad_steps": 1,
            "adaptive_closed_loop_stable_steps": 2,
            "adaptive_closed_loop_ema_decay": 0.0,
        },
        seed=0,
        device="cpu",
    )
    algo._grad_step_count = 1
    errors = [torch.tensor(0.01), torch.tensor(0.02), torch.tensor(0.03)]
    algo._update_closed_loop_error_ema(errors)
    assert algo._effective_closed_loop_horizon() == 3
    algo._update_closed_loop_error_ema(errors)
    assert algo._effective_closed_loop_horizon() == 5


def test_rolling_episode_metrics_flow_on_done_and_non_done_callbacks() -> None:
    algo = NativeResearchImZero(
        _TwoStepEnv(),
        {**TINY, "learning_starts": 100, "episode_reward_rolling_window": 20},
        seed=0,
        device="cpu",
    )
    seen: list[dict[str, float]] = []
    algo.learn(
        3,
        TrainingCallback(
            lambda step, reward=None, length=None, metrics=None: (
                seen.append(dict(metrics or {})) or True
            ),
        ),
    )
    assert seen[0]["episode_reward_rolling_count"] == 0.0
    assert seen[1]["episode_reward_rolling_count"] == 1.0
    assert seen[1]["episode_reward_rolling_mean"] == 2.0
    assert seen[2]["episode_reward_rolling_count"] == 1.0


def test_path_consistency_formula_stops_target_gradients_and_masks_terminal() -> None:
    algo = NativeResearchImZero(
        _DiscreteEnv(), {**TINY, "path_consistency_coef": 0.1}, seed=0, device="cpu",
    )
    bins = 2 * algo.support_size + 1
    current = torch.randn(1, bins, requires_grad=True)
    next_logits = torch.randn(1, bins, requires_grad=True)
    predicted_reward = torch.tensor([0.75], requires_grad=True)
    loss = algo._compute_path_consistency_loss(
        [current, next_logits], [predicted_reward, torch.zeros(1)],
        torch.ones(1, 2), torch.ones(1),
    )
    loss.backward()
    assert current.grad is not None and current.grad.abs().sum() > 0
    assert next_logits.grad is None
    assert predicted_reward.grad is None

    masked = algo._compute_path_consistency_loss(
        [torch.randn(1, bins), torch.randn(1, bins)],
        [torch.tensor([1.0]), torch.tensor([0.0])],
        torch.tensor([[1.0, 0.0]]), torch.ones(1),
    )
    assert masked.item() == 0.0


def test_probe_receives_actual_detached_value_targets(monkeypatch) -> None:
    algo = NativeLatentImZero(
        _DiscreteEnv(), {**TINY, "uncertainty_enabled": 1}, seed=0, device="cpu",
    )
    captured: dict[str, torch.Tensor] = {}
    original = algo.uncertainty_probe.bootstrap_loss

    def wrapped(reward_features, value_features, reward, value, valid):
        captured["value"] = value.detach().clone()
        return original(reward_features, value_features, reward, value, valid)

    monkeypatch.setattr(algo.uncertainty_probe, "bootstrap_loss", wrapped)
    actual_targets = torch.tensor([[2.0, -3.0], [4.0, 5.0]])
    algo._after_core_train_step(
        {"reward": np.zeros((2, 2), dtype=np.float32), "mask": np.ones((2, 2))},
        torch.randn(2, 5, algo.embed_dim),
        torch.zeros(2, dtype=torch.long),
        actual_targets,
    )
    assert torch.allclose(captured["value"], _signed_hyperbolic(actual_targets))


def test_return_calibration_uses_shifted_valid_pairs(monkeypatch) -> None:
    algo = NativeLatentImZero(
        _DiscreteEnv(), {**TINY, "uncertainty_enabled": 1}, seed=0, device="cpu",
    )

    def fake_loss(*args):
        return sum(parameter.sum() * 0.0 for parameter in algo.uncertainty_probe.parameters())

    def fake_reward(features):
        return torch.zeros(*features.shape[:-1], 2)

    def fake_value(features):
        result = torch.zeros(*features.shape[:-1], 2)
        result[:, 0] = 5.0
        return result

    monkeypatch.setattr(algo.uncertainty_probe, "bootstrap_loss", fake_loss)
    monkeypatch.setattr(algo.uncertainty_probe, "reward", fake_reward)
    monkeypatch.setattr(algo.uncertainty_probe, "value", fake_value)
    algo._after_core_train_step(
        {
            "reward": np.zeros((2, 2), dtype=np.float32),
            "mask": np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        },
        torch.randn(2, 5, algo.embed_dim),
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, 2),
    )
    # Only lane 0's k=0 -> k=1 pair is valid; its shifted V_{k+1} is zero.
    assert algo._uncertainty_surprise_ema == 0.0
    assert algo._uncertainty_disagreement_ema == 0.0


def test_return_uncertainty_reduces_sve_mix_but_early_state_keeps_base() -> None:
    algo = NativeLatentImZero(_DiscreteEnv(), TINY, seed=0, device="cpu")
    hidden = torch.zeros(3, algo.embed_dim)
    base = algo.reanalyze_value_mix
    assert algo._effective_reanalyze_value_mix(hidden, hidden, base).item() == base
    _set_probe_predictions(algo, [-1.0, 1.0])
    _set_calibrated_return_state(algo)
    algo._num_timesteps = algo.uncertainty_warmup_steps + 1
    early = algo._effective_reanalyze_value_mix(hidden, hidden, base)
    algo._num_timesteps = (
        algo.uncertainty_warmup_steps + algo.uncertainty_ramp_steps // 2
    )
    middle = algo._effective_reanalyze_value_mix(hidden, hidden, base)
    algo._num_timesteps = algo.uncertainty_warmup_steps + algo.uncertainty_ramp_steps
    effective = algo._effective_reanalyze_value_mix(hidden, hidden, base)
    assert torch.all(early < base)
    assert torch.all(early > middle)
    assert torch.all(middle > effective)
    assert torch.all(effective >= algo.uncertainty_sve_min_mix)
    algo._num_timesteps = algo.uncertainty_anneal_end + 1
    assert algo.uncertainty_runtime_scale == 0.0
    assert algo.uncertainty_reliability_scale > 0.0
    late_effective = algo._effective_reanalyze_value_mix(hidden, hidden, base)
    assert torch.allclose(late_effective, effective)

    zero_scale = NativeLatentImZero(
        _DiscreteEnv(), {**TINY, "uncertainty_scale": 0.0}, seed=0, device="cpu",
    )
    _set_probe_predictions(zero_scale, [-1.0, 1.0])
    _set_calibrated_return_state(zero_scale)
    assert zero_scale.uncertainty_reliability_scale == 0.0
    assert (
        zero_scale._effective_reanalyze_value_mix(hidden, hidden, base).item()
        == base
    )


def test_policy_ambiguity_extra_simulations_are_batched_and_capped() -> None:
    algo = NativeLatentImZero(
        _DiscreteEnv(),
        {
            **TINY, "num_simulations": 8, "num_simulations_initial": 2,
            "uncertainty_extra_simulations_max": 4, "voc_enabled": 0,
        },
        seed=0,
        device="cpu",
    )
    _set_calibrated_return_state(algo)
    uniform_logits = torch.zeros(5, algo.n_actions)
    assert algo._effective_search_simulations(2, uniform_logits, None) == 6
    assert algo._effective_search_simulations(8, uniform_logits, None) == 8


def test_learning_progress_priority_is_bounded_and_zero_weight_is_identity() -> None:
    base = NativeResearchImZero(_DiscreteEnv(), TINY, seed=0, device="cpu")
    _fill_buffer(base)
    current = np.array([0.2])
    unchanged, zero_bonus = base._apply_learning_progress_priority(
        current, np.array([0]), np.array([0]),
    )
    assert unchanged is current
    assert zero_bonus.item() == 0.0
    assert np.isnan(base.buffer.raw_learning_errors_for(np.array([0]), np.array([0]))[0])

    latent = NativeLatentImZero(
        _DiscreteEnv(),
        {
            **TINY, "learning_progress_priority_weight": 0.1,
            "learning_progress_priority_cap": 0.5,
        },
        seed=0,
        device="cpu",
    )
    _fill_buffer(latent)
    latent.buffer.episodes[0].pop("raw_learning_error")
    first, first_bonus = latent._apply_learning_progress_priority(
        current.copy(), np.array([0]), np.array([0]),
    )
    assert first.item() == pytest.approx(0.2)
    assert first_bonus.item() == 0.0

    second, second_bonus = latent._apply_learning_progress_priority(
        np.array([0.1]), np.array([0]), np.array([0]),
    )
    assert second_bonus.item() == pytest.approx(0.01)
    assert second.item() == pytest.approx(0.11)

    third, third_bonus = latent._apply_learning_progress_priority(
        np.array([0.1]), np.array([0]), np.array([0]),
    )
    assert third_bonus.item() == 0.0
    assert third.item() == pytest.approx(0.1)

    active = NativeLatentImZero(
        _DiscreteEnv(),
        {**TINY, "learning_progress_priority_weight": 0.1},
        seed=0,
        device="cpu",
    )
    active.buffer.add(
        np.zeros(4), np.array([1.0, 0.0, 0.0]), 0.0, np.ones(4),
        np.full(3, 1 / 3), False,
    )
    _, active_first_bonus = active._apply_learning_progress_priority(
        np.array([0.5]), np.array([-1]), np.array([0]),
    )
    _, active_second_bonus = active._apply_learning_progress_priority(
        np.array([0.2]), np.array([-1]), np.array([0]),
    )
    assert active_first_bonus.item() == 0.0
    assert active_second_bonus.item() == pytest.approx(0.03)
