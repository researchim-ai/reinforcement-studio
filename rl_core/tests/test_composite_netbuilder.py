"""Typed composite_v1 compiler and Zero-family runtime contracts."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from rl_core.algorithms.native.efficientzero import DEFAULT_HYPERPARAMS as EZ_DEFAULTS, NativeEfficientZero
from rl_core.algorithms.native.latentimzero import DEFAULT_HYPERPARAMS as LIZ_DEFAULTS, NativeLatentImZero
from rl_core.algorithms.native.researchimzero import DEFAULT_HYPERPARAMS as RIZ_DEFAULTS, NativeResearchImZero
from rl_core.algorithms.native.unizero import DEFAULT_HYPERPARAMS as UZ_DEFAULTS, NativeUniZero
from rl_core.composite_netbuilder import (
    CompositeEncoder,
    build_composite_encoder,
    default_composite_spec,
    validate_composite_spec,
)
from rl_core.netbuilder import NetworkSpecError


class _TinyImageEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 255, shape=(16, 16, 3), dtype=np.uint8)
    action_space = gym.spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        del action
        return self.observation_space.sample(), 0.0, False, False, {}


def _tiny_spec(family: str) -> dict:
    spec = default_composite_spec(family)
    if family == "efficientzero":
        spec["dimensions"].update({"latent_dim": 16, "hidden_dim": 16, "proj_dim": 8})
        spec["components"]["representation"]["hidden_sizes"] = [16]
        spec["components"]["projector"]["hidden_sizes"] = [8]
        spec["components"]["predictor"]["hidden_sizes"] = [8]
    else:
        spec["dimensions"].update({
            "embed_dim": 16, "num_layers": 1, "num_heads": 2, "ffn_multiplier": 2,
        })
        spec["components"]["tokenizer"]["hidden_sizes"] = [16]
        if family in {"researchimzero", "latentimzero"}:
            spec["dimensions"]["proj_dim"] = 8
            spec["components"]["projector"]["hidden_sizes"] = [8]
            spec["components"]["predictor"]["hidden_sizes"] = [8]
    return spec


@pytest.mark.parametrize("family", ["efficientzero", "unizero", "researchimzero", "latentimzero"])
def test_default_composite_specs_validate(family: str) -> None:
    assert validate_composite_spec(default_composite_spec(family), family)["family"] == family


def test_validation_rejects_cross_family_and_unsafe_shapes() -> None:
    with pytest.raises(NetworkSpecError, match="несовместима"):
        validate_composite_spec(default_composite_spec("unizero"), "researchimzero")

    bad_heads = default_composite_spec("unizero")
    bad_heads["dimensions"].update({"embed_dim": 30, "num_heads": 8})
    with pytest.raises(NetworkSpecError, match="делиться"):
        validate_composite_spec(bad_heads, "unizero")

    bad_bn = default_composite_spec("efficientzero")
    bad_bn["components"]["dynamics"]["batch_norm"] = True
    with pytest.raises(NetworkSpecError, match="batch=1"):
        validate_composite_spec(bad_bn, "efficientzero")


def test_custom_vector_and_image_encoders_produce_flat_features() -> None:
    vector_space = gym.spaces.Box(-1, 1, shape=(6,), dtype=np.float32)
    vector_spec = default_composite_spec("unizero")
    vector_spec["encoder"] = {
        "kind": "vector_mlp",
        "layers": [
            {"type": "linear", "out_features": 12},
            {"type": "activation", "fn": "relu"},
        ],
    }
    vector_encoder = build_composite_encoder(vector_space, validate_composite_spec(vector_spec, "unizero"))
    assert isinstance(vector_encoder, CompositeEncoder)
    assert vector_encoder(torch.zeros(2, 6)).shape == (2, 12)

    image_space = gym.spaces.Box(0, 255, shape=(16, 16, 3), dtype=np.uint8)
    image_spec = default_composite_spec("efficientzero")
    image_spec["encoder"] = {
        "kind": "image_cnn",
        "layers": [
            {"type": "conv2d", "out_channels": 8, "kernel_size": 3, "stride": 2, "padding": 1},
            {"type": "activation", "fn": "relu"},
            {"type": "flatten"},
        ],
    }
    image_encoder = build_composite_encoder(image_space, validate_composite_spec(image_spec, "efficientzero"))
    assert image_encoder(torch.zeros(2, 16, 16, 3)).shape == (2, 8 * 8 * 8)


@pytest.mark.parametrize(
    ("algorithm_cls", "defaults", "family"),
    [
        (NativeEfficientZero, EZ_DEFAULTS, "efficientzero"),
        (NativeUniZero, UZ_DEFAULTS, "unizero"),
        (NativeResearchImZero, RIZ_DEFAULTS, "researchimzero"),
        (NativeLatentImZero, LIZ_DEFAULTS, "latentimzero"),
    ],
)
def test_composite_runtime_save_load_and_optimizer_contract(
    algorithm_cls,
    defaults: dict,
    family: str,
    tmp_path: Path,
) -> None:
    env = gym.make("CartPole-v1")
    spec = _tiny_spec(family)
    hyperparams = {
        **defaults,
        "network_spec": spec,
        "buffer_size": 40,
        "batch_size": 2,
        "num_simulations": 2,
        "num_sampled_actions": 2,
        "num_top_actions": 2,
        "value_support_size": 10,
    }
    algorithm = algorithm_cls(env, hyperparams, seed=0, device="cpu")
    obs = np.stack([env.observation_space.sample(), env.observation_space.sample()]).astype(np.float32)
    results = algorithm.search(obs, *([None, None],) if family != "efficientzero" else ())
    assert len(results) == 2
    assert algorithm.hyperparams["network_spec"]["format"] == "composite_v1"

    optimizers = [
        optimizer for optimizer in (
            getattr(algorithm, "optimizer", None),
            getattr(algorithm, "model_optimizer", None),
            getattr(algorithm, "actor_optimizer", None),
            getattr(algorithm, "critic_optimizer", None),
        )
        if optimizer is not None
    ]
    optimizer_ids = {
        id(parameter)
        for optimizer in optimizers
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    for target_name in ("target_tokenizer", "target_transformer", "target_heads"):
        target_module = getattr(algorithm, target_name, None)
        if target_module is not None:
            assert all(id(parameter) not in optimizer_ids for parameter in target_module.parameters())
    ema_critic = getattr(algorithm, "ema_critic", None)
    if ema_critic is not None:
        assert all(id(parameter) not in optimizer_ids for parameter in ema_critic.parameters())
        assert all(not parameter.requires_grad for parameter in ema_critic.parameters())

    checkpoint = tmp_path / f"{family}.pt"
    algorithm.save(checkpoint)
    loaded = algorithm_cls.load(checkpoint, gym.make("CartPole-v1"), device="cpu")
    assert loaded.hyperparams["network_spec"] == algorithm.hyperparams["network_spec"]
    env.close()


def test_legacy_hyperparams_still_build_without_composite_spec() -> None:
    env = gym.make("CartPole-v1")
    algorithm = NativeEfficientZero(
        env,
        {**EZ_DEFAULTS, "latent_dim": 17, "hidden_dim": 19, "proj_dim": 7},
        seed=0,
        device="cpu",
    )
    assert algorithm.representation.head[-1].out_features == 17
    assert "network_spec" not in algorithm.hyperparams
    env.close()


def test_latentimzero_uses_research_components_and_isolated_probe() -> None:
    env = gym.make("CartPole-v1")
    spec = _tiny_spec("latentimzero")
    widths = {
        "tokenizer": 11,
        "heads": 12,
        "projector": 13,
        "predictor": 14,
    }
    for name, width in widths.items():
        spec["components"][name]["hidden_sizes"] = [width]
    algorithm = NativeLatentImZero(
        env, {**LIZ_DEFAULTS, "network_spec": spec, "value_support_size": 10},
        seed=0, device="cpu",
    )

    modules = {
        "tokenizer": algorithm.tokenizer.head,
        "heads": algorithm.heads.trunk,
        "projector": algorithm.projector.net,
        "predictor": algorithm.predictor.net,
    }
    for name, module in modules.items():
        first_linear = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear))
        assert first_linear.out_features == widths[name]
    env.close()


@pytest.mark.parametrize(
    ("algorithm_cls", "defaults", "family"),
    [
        (NativeEfficientZero, EZ_DEFAULTS, "efficientzero"),
        (NativeUniZero, UZ_DEFAULTS, "unizero"),
        (NativeResearchImZero, RIZ_DEFAULTS, "researchimzero"),
        (NativeLatentImZero, LIZ_DEFAULTS, "latentimzero"),
    ],
)
def test_composite_specs_support_continuous_actions(algorithm_cls, defaults: dict, family: str) -> None:
    env = gym.make("Pendulum-v1")
    algorithm = algorithm_cls(
        env,
        {
            **defaults,
            "network_spec": _tiny_spec(family),
            "buffer_size": 20,
            "batch_size": 2,
            "num_simulations": 2,
            "num_sampled_actions": 2,
            "num_top_actions": 2,
            "value_support_size": 10,
        },
        seed=0,
        device="cpu",
    )
    obs = np.stack([env.observation_space.sample(), env.observation_space.sample()]).astype(np.float32)
    results = algorithm.search(obs, *([None, None],) if family != "efficientzero" else ())
    assert all(env.action_space.contains(np.asarray(result["env_action"], dtype=np.float32)) for result in results)
    env.close()


@pytest.mark.parametrize(
    ("algorithm_cls", "defaults", "family"),
    [
        (NativeEfficientZero, EZ_DEFAULTS, "efficientzero"),
        (NativeUniZero, UZ_DEFAULTS, "unizero"),
        (NativeResearchImZero, RIZ_DEFAULTS, "researchimzero"),
        (NativeLatentImZero, LIZ_DEFAULTS, "latentimzero"),
    ],
)
def test_custom_image_encoder_runs_through_search(algorithm_cls, defaults: dict, family: str) -> None:
    env = _TinyImageEnv()
    spec = _tiny_spec(family)
    spec["encoder"] = {
        "kind": "image_cnn",
        "layers": [
            {"type": "conv2d", "out_channels": 8, "kernel_size": 3, "stride": 2, "padding": 1},
            {"type": "activation", "fn": "relu"},
            {"type": "flatten"},
        ],
    }
    algorithm = algorithm_cls(
        env,
        {
            **defaults,
            "network_spec": spec,
            "buffer_size": 20,
            "batch_size": 2,
            "num_simulations": 2,
            "num_sampled_actions": 2,
            "num_top_actions": 2,
            "value_support_size": 10,
        },
        seed=0,
        device="cpu",
    )
    obs = np.stack([env.observation_space.sample(), env.observation_space.sample()])
    results = algorithm.search(obs, *([None, None],) if family != "efficientzero" else ())
    assert len(results) == 2


@pytest.mark.parametrize(
    ("algorithm_cls", "defaults", "family"),
    [
        (NativeUniZero, UZ_DEFAULTS, "unizero"),
        (NativeResearchImZero, RIZ_DEFAULTS, "researchimzero"),
    ],
)
def test_custom_transformer_incremental_matches_full_forward(algorithm_cls, defaults: dict, family: str) -> None:
    env = gym.make("CartPole-v1")
    spec = _tiny_spec(family)
    spec["dimensions"].update({"ffn_multiplier": 3, "dropout": 0.0})
    algorithm = algorithm_cls(
        env,
        {**defaults, "network_spec": spec, "buffer_size": 20, "batch_size": 2},
        seed=0,
        device="cpu",
    )
    algorithm.transformer.eval()
    tokens = torch.randn(1, 5, 16)
    full = algorithm.transformer(tokens, torch.ones(1, 5, dtype=torch.bool))
    cache = None
    incremental = []
    for position in range(tokens.shape[1]):
        hidden, caches = algorithm.transformer.forward_incremental_batch(
            tokens[:, position],
            torch.tensor([position]),
            [cache],
        )
        incremental.append(hidden)
        cache = caches[0]
    assert torch.allclose(torch.stack(incremental, dim=1), full, atol=1e-5, rtol=1e-5)
    env.close()
