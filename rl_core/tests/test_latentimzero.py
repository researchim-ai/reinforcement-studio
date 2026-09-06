from __future__ import annotations

import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from rl_core.algorithms.base import TrainingCallback
from rl_core.algorithms.native.latentimzero import (
    DEFAULT_HYPERPARAMS,
    NativeLatentImZero,
    _LatentSearchNode,
    _backpropagate_variable_discount,
    _balanced_kl,
    _lambda_return,
    _latent_information,
    _st_categorical,
    _stochastic_usage,
    _symexp,
    _symlog,
    _symlog_logits_to_scalar,
    _symlog_scalar_to_two_hot,
)
from rl_core.algorithms.native.researchimzero import _MinMaxStats
from rl_core.algorithms.vec_env import make_env_or_vec


TINY = {
    **DEFAULT_HYPERPARAMS,
    "embed_dim": 16,
    "hidden_dim": 32,
    "num_layers": 1,
    "num_heads": 2,
    "stoch_variables": 4,
    "stoch_classes": 4,
    "value_support_size": 10,
    "batch_size": 2,
    "unroll_steps": 2,
    "td_steps": 2,
    "imagination_horizon": 3,
    "planner_horizon": 2,
    "num_sampled_actions": 3,
    "num_simulations_initial": 2,
    "num_simulations": 4,
    "learning_starts": 2,
    "world_model_learning_starts": 2,
    "policy_learning_starts": 2,
    "min_context_transitions_per_lane": 0,
    "reanalyze_batch_size": 0,
    "fresh_distill_fraction": 0.5,
}


class _OneStepDiscrete(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        return np.full(4, float(action) / 3, dtype=np.float32), 1.0, True, False, {}


class _OneStepContinuous(gym.Env):
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


def test_st_categorical_is_hard_and_has_gradient() -> None:
    logits = torch.randn(3, 4, 5, requires_grad=True)
    sample = _st_categorical(logits, unimix=0.01)
    assert sample.shape == logits.shape
    assert torch.allclose(sample.sum(-1), torch.ones(3, 4))
    assert torch.allclose(sample.detach(), sample.detach().round(), atol=1e-6)
    sample.square().sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_balanced_kl_free_bits_and_stop_gradient_paths() -> None:
    post = torch.randn(2, 3, 4, requires_grad=True)
    prior = torch.randn(2, 3, 4, requires_grad=True)
    loss, raw = _balanced_kl(post, prior, balance=0.8, free_bits=1.0)
    assert loss >= 1.0
    assert torch.isfinite(raw)
    loss.backward()
    assert post.grad is not None and prior.grad is not None


def test_continue_aware_lambda_return_masks_future() -> None:
    reward = torch.tensor([[1.0, 2.0, 100.0]])
    value = torch.zeros_like(reward)
    continuation = torch.tensor([[1.0, 0.0, 1.0]])
    result = _lambda_return(reward, value, torch.zeros(1), continuation, 0.99, 0.95)
    assert torch.allclose(result[:, 1], torch.tensor([2.0]))
    assert result[:, 0].item() < 4.0


def test_symlog_symexp_roundtrip_and_boundaries() -> None:
    values = torch.tensor([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    assert torch.allclose(_symexp(_symlog(values)), values, rtol=1e-5, atol=1e-5)

    support_size = 10
    target = _symlog_scalar_to_two_hot(values, support_size)
    decoded = _symlog_logits_to_scalar(target.log(), support_size)
    assert torch.allclose(decoded, values, rtol=1e-4, atol=1e-4)
    assert torch.allclose(target.sum(-1), torch.ones_like(values))

    outside = torch.tensor([-1e20, 1e20])
    clipped = _symlog_logits_to_scalar(
        _symlog_scalar_to_two_hot(outside, support_size).log(), support_size,
    )
    boundary = _symexp(torch.tensor(float(support_size)))
    assert torch.allclose(clipped, torch.tensor([-boundary, boundary]))


def test_done_is_stored_and_sampled() -> None:
    algo = NativeLatentImZero(_OneStepDiscrete(), TINY, seed=0, device="cpu")
    obs = np.zeros(4, dtype=np.float32)
    algo.buffer.add(obs, np.eye(3, dtype=np.float32)[0], 1.0, obs, np.ones(3) / 3, True)
    assert algo.buffer.episodes[0]["done"].tolist() == [1.0]
    batch = algo.buffer.sample(2, 2, 2, 0.99)
    assert np.all(batch["done"][:, 0] == 1.0)


def test_uncertainty_gates_compute_and_imagination() -> None:
    algo = NativeLatentImZero(_OneStepDiscrete(), TINY, seed=0, device="cpu")
    assert algo._planning_budget(algo.uncertainty_low) == algo.num_simulations
    assert algo._planning_budget(algo.uncertainty_high) == algo.num_simulations_initial
    assert algo._gated_imagination_horizon(algo.uncertainty_low) == algo.horizon
    assert algo._gated_imagination_horizon(algo.uncertainty_high) == 1
    effective = algo._effective_uncertainty(torch.tensor(0.0))
    assert effective >= algo.uncertainty_high
    assert algo._confidence(effective) <= 0.051


def _fill_sequence(algo: NativeLatentImZero, length: int = 8) -> None:
    obs = np.zeros(4, dtype=np.float32)
    for step in range(length):
        action = step % 3
        next_obs = np.full(4, (step + 1) / length, dtype=np.float32)
        algo.buffer.add(
            obs,
            np.eye(3, dtype=np.float32)[action],
            float(step % 2),
            next_obs,
            np.ones(3, dtype=np.float32) / 3,
            step == length - 1,
        )
        obs = next_obs


def test_replay_context_changes_start_and_reaches_training_path(monkeypatch) -> None:
    algo = NativeLatentImZero(_OneStepDiscrete(), TINY, seed=0, device="cpu")
    current = np.zeros((2, 4), dtype=np.float32)
    context_obs = np.zeros((2, 2, 4), dtype=np.float32)
    context_obs[1, :, 0] = 1.0
    context_action = np.zeros((2, 2, 3), dtype=np.float32)
    context_action[0, :, 0] = 1.0
    context_action[1, :, 2] = 1.0
    valid = np.ones((2, 2), dtype=bool)
    states = algo._reconstruct_history(
        context_obs, context_action, valid, current, sample=False,
    )
    features = algo.world_model.features(states)
    assert not torch.allclose(features[0], features[1])
    assert all(state.cache.length == 2 for state in states)  # dynamics transitions only

    _fill_sequence(algo)
    algo.buffer.recent_fraction = 1.0
    algo.buffer.recent_window = 1
    original = algo._reconstruct_history
    seen_context: list[int] = []

    def wrapped(context_obs, context_action, context_valid, current_obs, *, sample):
        seen_context.append(int(context_valid.sum()))
        return original(
            context_obs, context_action, context_valid, current_obs, sample=sample,
        )

    monkeypatch.setattr(algo, "_reconstruct_history", wrapped)
    metrics = algo._train_step()
    assert seen_context and max(seen_context) > 0
    assert np.isfinite(metrics["real_critic_loss"])
    assert np.isfinite(metrics["real_target_mean"])


def test_variable_continue_discount_stops_tree_backup() -> None:
    root = _LatentSearchNode(1.0)
    child = _LatentSearchNode(1.0, root)
    root.children = [child]
    child.reward_value = 2.0
    child.edge_discount = 0.0
    stats = _MinMaxStats(0.01)
    _backpropagate_variable_discount([root, child], 100.0, stats)
    assert child.value() == 100.0
    assert root.value() == 2.0


def test_gumbel_tree_uses_exact_simulations_and_reaches_depth_two() -> None:
    params = {
        **TINY,
        "num_simulations": 8,
        "num_simulations_initial": 8,
        "num_top_actions": 2,
        "planner_horizon": 3,
    }
    algo = NativeLatentImZero(_OneStepDiscrete(), params, seed=0, device="cpu")
    state = algo.world_model.observe(torch.zeros(1, 4), [None], sample=False)[0]
    result = algo._plan_state(state, deterministic=True)
    assert result["simulation_count"] == 8
    assert result["max_depth"] >= 2
    assert np.isclose(result["policy_target"].sum(), 1.0)


def test_grounded_critic_metrics_and_stochastic_diagnostics() -> None:
    algo = NativeLatentImZero(_OneStepDiscrete(), TINY, seed=0, device="cpu")
    _fill_sequence(algo)
    metrics = algo._train_step()
    for key in (
        "real_critic_loss", "imagined_critic_loss", "real_target_mean",
        "model_bias", "posterior_prior_kl", "posterior_entropy",
        "stochastic_usage", "model_error_ema", "planning_budget",
        "imagination_horizon", "observation_prediction_error",
        "observation_target_variance", "observation_predictor_variance",
        "posterior_conditional_entropy", "posterior_marginal_entropy",
        "latent_information_loss", "latent_class_utilization",
        "observation_error_ema", "stochastic_usage_ema",
        "uncertainty_representation", "representation_readiness",
    ):
        assert key in metrics and np.isfinite(metrics[key])
    assert metrics["real_critic_loss"] > 0
    assert 0.0 <= metrics["stochastic_usage"] <= 1.01


def test_terminal_grounded_target_is_real_reward_without_search_value() -> None:
    algo = NativeLatentImZero(_OneStepDiscrete(), TINY, seed=0, device="cpu")
    _fill_sequence(algo)
    algo.buffer.recent_fraction = 0.0
    episode = algo.buffer.episodes[0]
    episode["priority"][:] = algo.buffer.min_priority
    episode["priority"][-1] = 1e12
    algo.buffer._episode_alpha_sum[0] = float(
        np.sum(episode["priority"] ** algo.buffer.priority_alpha),
    )
    batch = algo.buffer.sample(algo.batch_size, algo.unroll_steps, algo.td_steps, algo.gamma)
    targets = algo._grounded_targets(batch)
    assert torch.allclose(targets[:, 0], torch.ones(algo.batch_size))
    assert np.all(batch["search_value"][:, 0] <= algo.buffer._NO_SEARCH_VALUE + 1)


def test_ensemble_bootstrap_masks_preserve_member_diversity(monkeypatch) -> None:
    algo = NativeLatentImZero(_OneStepDiscrete(), TINY, seed=0, device="cpu")
    _fill_sequence(algo)
    reference_prior = algo.world_model.ensemble_prior[0].state_dict()
    reference_reward = algo.world_model.ensemble_reward[0].state_dict()
    for member in algo.world_model.ensemble_prior[1:]:
        member.load_state_dict(reference_prior)
    for member in algo.world_model.ensemble_reward[1:]:
        member.load_state_dict(reference_reward)

    patterns = ([1.0, 0.0], [0.0, 1.0], [1.0, 1.0])
    calls = 0

    def bootstrap_rand(*shape, **kwargs):
        nonlocal calls
        pattern = patterns[calls % len(patterns)]
        calls += 1
        return torch.tensor(pattern, device=kwargs.get("device"))[: shape[0]]

    monkeypatch.setattr(torch, "rand", bootstrap_rand)
    algo._train_step(train_policy=False)
    first = list(algo.world_model.ensemble_prior[0].parameters())
    second = list(algo.world_model.ensemble_prior[1].parameters())
    assert any(not torch.allclose(a, b) for a, b in zip(first, second))


def test_warmup_trains_world_but_never_searches_without_context(monkeypatch) -> None:
    params = {
        **TINY,
        "world_model_learning_starts": 1,
        "policy_learning_starts": 0,
        "min_context_transitions_per_lane": 10,
    }
    algo = NativeLatentImZero(_OneStepDiscrete(), params, seed=0, device="cpu")
    train_policy_flags: list[bool] = []

    def fake_train_step(train_policy=True):
        train_policy_flags.append(bool(train_policy))
        return {"model_loss": 0.0}

    monkeypatch.setattr(algo, "_train_step", fake_train_step)
    monkeypatch.setattr(
        algo, "search",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("search during warm-up")),
    )
    algo.learn(4, TrainingCallback(lambda *args, **kwargs: True))
    assert train_policy_flags and not any(train_policy_flags)


def test_prior_rollout_advances_cache_and_target_is_outside_optimizers() -> None:
    algo = NativeLatentImZero(_OneStepDiscrete(), TINY, seed=0, device="cpu")
    obs = torch.zeros(2, 4)
    posterior = algo.world_model.observe(obs, [None, None], sample=False)
    before = [state.cache.length for state in posterior]
    actions = torch.eye(3)[:2]
    imagined = algo.world_model.prior_step(posterior, actions, sample=False)
    assert [state.cache.length for state in imagined] == [length + 1 for length in before]
    assert torch.isfinite(algo.world_model.features(imagined)).all()

    optimizer_ids = {
        id(parameter)
        for optimizer in (algo.model_optimizer, algo.actor_optimizer, algo.critic_optimizer)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert all(id(parameter) not in optimizer_ids for parameter in algo.ema_critic.parameters())
    assert not list(algo.observation_target.parameters())


def test_causal_prior_is_obs_invariant_and_posterior_corrects_without_advance() -> None:
    algo = NativeLatentImZero(_OneStepDiscrete(), TINY, seed=0, device="cpu")
    model = algo.world_model
    obs_a = torch.zeros(1, 4)
    obs_b = torch.ones(1, 4)
    initial_a = model.observe(obs_a, [None], sample=False)[0]
    initial_b = model.observe(obs_b, [None], sample=False)[0]
    assert initial_a.cache.length == initial_b.cache.length == 0
    assert torch.allclose(initial_a.prior_logits, initial_b.prior_logits)
    assert not torch.allclose(initial_a.post_logits, initial_b.post_logits)

    action = torch.tensor([[0.0, 1.0, 0.0]])
    pending = model.prior_step([initial_a], action, sample=False)[0]
    corrected = model.observe(obs_b, [pending], prev_action=None, sample=False)[0]
    assert pending.prior_pending and not corrected.prior_pending
    assert corrected.cache.length == pending.cache.length == 1
    assert torch.equal(corrected.prior_logits, pending.prior_logits)


def test_uniform_posterior_has_near_zero_stochastic_usage() -> None:
    logits = torch.zeros(8, 4, 16)
    assert _stochastic_usage(logits, unimix=0.01).item() < 1e-6


def test_latent_information_prefers_confident_diverse_classes() -> None:
    uniform = torch.zeros(16, 4, 4)
    single = torch.full((16, 4, 4), -12.0)
    single[..., 0] = 12.0
    diverse = torch.full((16, 4, 4), -12.0)
    for sample in range(16):
        diverse[sample, :, sample % 4] = 12.0
    uniform_loss = _latent_information(uniform, 0.0)[0]
    single_loss = _latent_information(single, 0.0)[0]
    diverse_loss, conditional, marginal, utilization = _latent_information(diverse, 0.0)
    assert abs(float(uniform_loss)) < 1e-6
    assert abs(float(single_loss)) < 1e-5
    assert diverse_loss < uniform_loss and diverse_loss < single_loss
    assert conditional < 1e-4
    assert marginal > 1.3
    assert utilization > 0.99


def test_fixed_observation_target_is_distinct_and_constant(tmp_path: Path) -> None:
    algo = NativeLatentImZero(_OneStepDiscrete(), TINY, seed=7, device="cpu")
    same_seed = NativeLatentImZero(_OneStepDiscrete(), TINY, seed=7, device="cpu")
    assert torch.equal(
        algo.observation_target.projection, same_seed.observation_target.projection,
    )
    observations = torch.stack([torch.zeros(4), torch.ones(4)])
    before = algo.observation_target(observations).clone()
    assert not torch.allclose(before[0], before[1])
    state_before = {
        key: value.clone() for key, value in algo.observation_target.state_dict().items()
    }
    algo._update_ema_critic()
    algo.model_optimizer.zero_grad()
    state = algo.world_model.observe(observations, [None, None], sample=True)
    prediction = F.layer_norm(
        algo.world_model.observation_predictor(algo.world_model.features(state)), (16,),
    )
    F.mse_loss(prediction, before).backward()
    algo.model_optimizer.step()
    assert all(
        torch.equal(value, state_before[key])
        for key, value in algo.observation_target.state_dict().items()
    )
    checkpoint = tmp_path / "fixed-target.pt"
    algo.save(checkpoint)
    loaded = NativeLatentImZero.load(checkpoint, _OneStepDiscrete(), device="cpu")
    assert torch.equal(loaded.observation_target(observations), before)


def test_v2_checkpoint_is_rejected_with_collapse_reason(tmp_path: Path) -> None:
    checkpoint = tmp_path / "v2.pt"
    torch.save({"checkpoint_version": 2}, checkpoint)
    with pytest.raises(ValueError, match="collapsing EMA tokenizer target"):
        NativeLatentImZero.load(checkpoint, _OneStepDiscrete(), device="cpu")


def test_constant_predictor_cannot_fit_varied_fixed_targets() -> None:
    algo = NativeLatentImZero(_OneStepDiscrete(), TINY, seed=0, device="cpu")
    observations = torch.eye(4)
    targets = algo.observation_target(observations)
    constant = targets.mean(0, keepdim=True).expand_as(targets)
    error = F.mse_loss(constant, targets)
    assert targets.var(dim=0, unbiased=False).mean() > 0.05
    assert error > 0.05


def test_batched_reconstruction_matches_lane_reference() -> None:
    algo = NativeLatentImZero(_OneStepDiscrete(), TINY, seed=0, device="cpu")
    rng = np.random.default_rng(4)
    context_obs = rng.normal(size=(3, 3, 4)).astype(np.float32)
    context_action = np.eye(3, dtype=np.float32)[
        rng.integers(0, 3, size=(3, 3))
    ]
    valid = np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool)
    current = rng.normal(size=(3, 4)).astype(np.float32)
    batched = algo._reconstruct_history(
        context_obs, context_action, valid, current, sample=False,
    )
    reference = []
    for lane, count in enumerate(valid.sum(1)):
        state = None
        for t in range(int(count)):
            previous = None if t == 0 else torch.as_tensor(
                context_action[lane, t - 1 : t],
            )
            state = algo.world_model.observe(
                torch.as_tensor(context_obs[lane, t : t + 1]),
                [state],
                previous,
                sample=False,
            )[0]
        previous = None if count == 0 else torch.as_tensor(
            context_action[lane, int(count) - 1 : int(count)],
        )
        reference.append(algo.world_model.observe(
            torch.as_tensor(current[lane : lane + 1]), [state], previous, sample=False,
        )[0])
    for actual, expected in zip(batched, reference):
        assert actual.cache.length == expected.cache.length
        assert torch.allclose(actual.h, expected.h, atol=1e-6)
        assert torch.allclose(actual.z, expected.z, atol=1e-6)
        assert torch.allclose(actual.prior_logits, expected.prior_logits, atol=1e-6)
        assert torch.allclose(actual.post_logits, expected.post_logits, atol=1e-6)


def test_grounded_targets_vectorized_match_reference_and_terminal() -> None:
    algo = NativeLatentImZero(_OneStepDiscrete(), TINY, seed=0, device="cpu")
    _fill_sequence(algo)
    batch = algo.buffer.sample(algo.batch_size, algo.unroll_steps, algo.td_steps, algo.gamma)
    starts = algo._reconstruct_history(
        batch["context_obs"], batch["context_action"], batch["context_valid"],
        batch["obs0"], sample=False,
    )
    calls = []
    handle = algo.ema_critic.register_forward_hook(
        lambda _module, inputs, _output: calls.append(inputs[0].shape[0]),
    )
    actual = algo._grounded_targets(batch, starts)
    handle.remove()
    assert len(calls) <= 1

    states = starts
    landings = []
    actions = torch.as_tensor(batch["bootstrap_action"])
    observations = torch.as_tensor(batch["bootstrap_next_obs"])
    with torch.no_grad():
        for step in range(actions.shape[1]):
            states = algo.world_model.observe(
                observations[:, step],
                algo.world_model.prior_step(states, actions[:, step], sample=False),
                sample=False,
            )
            landings.append(states)
        values = torch.zeros_like(actual)
        for b in range(algo.batch_size):
            for k in range(algo.unroll_steps):
                landing = k + int(batch["td_horizon"][b, k]) - 1
                if 0 <= landing < len(landings):
                    values[b, k] = algo.ema_critic(
                        algo.world_model.features([landings[landing][b]]),
                    )[0]
        expected = (
            torch.as_tensor(batch["td_reward"])
            + torch.as_tensor(batch["td_discount"])
            * torch.as_tensor(batch["td_bootstrap_mask"]) * values
        )
    assert torch.allclose(actual, expected, atol=1e-6)
    terminal = torch.as_tensor(batch["td_bootstrap_mask"]) == 0
    assert torch.equal(actual[terminal], torch.as_tensor(batch["td_reward"])[terminal])


def test_observation_objective_reaches_posterior_and_encoder() -> None:
    algo = NativeLatentImZero(_OneStepDiscrete(), TINY, seed=0, device="cpu")
    model = algo.world_model
    state = model.observe(torch.randn(3, 4), [None] * 3, sample=True)
    prediction = model.observation_predictor(model.features(state))
    prediction = F.layer_norm(prediction, (prediction.shape[-1],))
    with torch.no_grad():
        target = algo.observation_target(torch.randn(3, 4))
    F.mse_loss(prediction, target).backward()
    posterior_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.posterior_head.parameters() if parameter.grad is not None
    )
    encoder_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.tokenizer.parameters() if parameter.grad is not None
    )
    assert posterior_grad > 0 and encoder_grad > 0


def test_actor_imagination_return_has_nonzero_gradient_without_distillation() -> None:
    algo = NativeLatentImZero(_OneStepDiscrete(), TINY, seed=0, device="cpu")
    starts = algo.world_model.observe(torch.randn(4, 4), [None] * 4, sample=True)
    for parameter in algo.world_model.parameters():
        parameter.requires_grad_(False)
    _, reward, _, _, _ = algo._imagine(starts)
    (-reward.mean()).backward()
    gradients = [
        parameter.grad for parameter in algo.actor.parameters() if parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0


def test_normalized_model_error_opens_planner_and_imagination() -> None:
    algo = NativeLatentImZero(_OneStepDiscrete(), TINY, seed=0, device="cpu")
    algo._model_error_ema = 0.02
    algo._posterior_kl_ema = 0.02
    algo._observation_error_ema = 0.01
    algo._stochastic_usage_ema = 0.25
    uncertainty = float(algo._effective_uncertainty(torch.tensor(0.0)))
    assert algo._planning_budget(uncertainty) > algo.num_simulations_initial
    assert algo._gated_imagination_horizon(uncertainty) > 1


def test_collapsed_latent_keeps_planner_and_imagination_closed() -> None:
    algo = NativeLatentImZero(_OneStepDiscrete(), TINY, seed=0, device="cpu")
    algo._model_error_ema = 0.0
    algo._posterior_kl_ema = 0.0
    algo._observation_error_ema = 0.0
    algo._stochastic_usage_ema = 0.024
    uncertainty = float(algo._effective_uncertainty(torch.tensor(0.00044)))
    assert uncertainty >= algo.uncertainty_high
    assert algo._planning_budget(uncertainty) == algo.num_simulations_initial
    assert algo._gated_imagination_horizon(uncertainty) == 1


def test_stochastic_and_continue_ablation_switches_are_exact() -> None:
    params = {**TINY, "use_stochastic_state": 0, "use_continue": 0}
    algo = NativeLatentImZero(_OneStepDiscrete(), params, seed=0, device="cpu")
    states = algo.world_model.observe(torch.zeros(2, 4), [None, None], sample=True)
    assert torch.count_nonzero(torch.stack([state.z for state in states])) == 0
    feat = algo.world_model.features(states)
    assert torch.equal(algo.world_model.continue_prob(feat), torch.ones(2))
    assert algo.kl_coef == 0.0


def test_native_runner_registration() -> None:
    from rl_core.algorithms.native_runner import ALGO_CLASSES, DEFAULT_HYPERPARAMS

    assert ALGO_CLASSES["latentimzero"] is NativeLatentImZero
    assert DEFAULT_HYPERPARAMS["latentimzero"]["imagination_horizon"] == 15


def _smoke(env: gym.Env, steps: int = 5) -> NativeLatentImZero:
    algo = NativeLatentImZero(env, TINY, seed=0, device="cpu")
    seen: list[int] = []
    algo.learn(steps, TrainingCallback(lambda n, *args, **kwargs: seen.append(n) or True))
    assert seen[-1] >= steps
    metrics = algo._last_metrics
    for key in ("model_loss", "actor_loss", "critic_loss", "continue_loss"):
        assert np.isfinite(metrics[key])
    return algo


def test_discrete_single_and_vector_env_learn_predict() -> None:
    single = _smoke(_OneStepDiscrete())
    action, _ = single.predict(np.zeros(4, dtype=np.float32), deterministic=True, episode_start=True)
    assert single._action_space.contains(action)
    single.env.close()

    vector = make_env_or_vec(_OneStepDiscrete, num_envs=2, parallel=False)
    _smoke(vector, steps=4)
    vector.close()


def test_continuous_learn_predict_and_save_load() -> None:
    algo = _smoke(_OneStepContinuous())
    obs = np.zeros(5, dtype=np.float32)
    action, _ = algo.predict(obs, deterministic=True, episode_start=True)
    assert algo._action_space.contains(action)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "latentimzero.pt"
        algo._model_error_ema = 0.123
        algo._posterior_kl_ema = 0.456
        algo._observation_error_ema = 0.234
        algo._stochastic_usage_ema = 0.345
        algo.save(path)
        loaded = NativeLatentImZero.load(path, _OneStepContinuous(), device="cpu")
        loaded_action, _ = loaded.predict(obs, deterministic=True, episode_start=True)
        assert loaded._action_space.contains(loaded_action)
        assert all(not parameter.requires_grad for parameter in loaded.ema_critic.parameters())
        assert loaded._model_error_ema == 0.123
        assert loaded._posterior_kl_ema == 0.456
        assert loaded._observation_error_ema == 0.234
        assert loaded._stochastic_usage_ema == 0.345
    algo.env.close()


def test_image_observation_learn_predict_contract() -> None:
    env = _ImageEnv()
    algo = _smoke(env, steps=3)
    obs, _ = env.reset()
    action, _ = algo.predict(obs, deterministic=True, episode_start=True)
    assert env.action_space.contains(action)
    env.close()
