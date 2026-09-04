"""Regression coverage for compound model architecture introspection."""
from __future__ import annotations

import gymnasium as gym
import pytest

from rl_core.algorithms.native.efficientzero import (
    DEFAULT_HYPERPARAMS as EFFICIENTZERO_DEFAULTS,
    NativeEfficientZero,
)
from rl_core.algorithms.native.researchimzero import (
    DEFAULT_HYPERPARAMS as RESEARCHIMZERO_DEFAULTS,
    NativeResearchImZero,
)
from rl_core.algorithms.native.unizero import DEFAULT_HYPERPARAMS as UNIZERO_DEFAULTS, NativeUniZero
from rl_core.inspect import _algorithm_architecture_summary


@pytest.mark.parametrize(
    ("algorithm_cls", "hyperparams", "expected_components"),
    [
        (
            NativeEfficientZero,
            {
                **EFFICIENTZERO_DEFAULTS,
                "latent_dim": 16,
                "hidden_dim": 16,
                "proj_dim": 8,
                "buffer_size": 20,
                "batch_size": 2,
            },
            {"representation", "dynamics", "prediction", "projector", "predictor"},
        ),
        (
            NativeUniZero,
            {
                **UNIZERO_DEFAULTS,
                "embed_dim": 16,
                "num_layers": 1,
                "num_heads": 2,
                "context_length": 2,
                "buffer_size": 20,
                "batch_size": 2,
                "value_support_size": 10,
            },
            {
                "tokenizer",
                "action_embed",
                "transformer",
                "heads",
                "target_tokenizer",
                "target_transformer",
                "target_heads",
            },
        ),
        (
            NativeResearchImZero,
            {
                **RESEARCHIMZERO_DEFAULTS,
                "embed_dim": 16,
                "num_layers": 1,
                "num_heads": 2,
                "context_length": 2,
                "buffer_size": 20,
                "batch_size": 2,
                "value_support_size": 10,
                "proj_dim": 8,
            },
            {
                "tokenizer",
                "action_embed",
                "transformer",
                "heads",
                "target_tokenizer",
                "target_transformer",
                "target_heads",
                "projector",
                "predictor",
                "direct_parameters",
            },
        ),
    ],
)
def test_compound_architecture_reports_every_top_level_module(
    algorithm_cls,
    hyperparams,
    expected_components,
) -> None:
    env = gym.make("CartPole-v1")
    algorithm = algorithm_cls(env, hyperparams, seed=0, device="cpu")
    summary = _algorithm_architecture_summary(algorithm)
    components = {component["name"]: component for component in summary["architecture_components"]}

    assert expected_components <= components.keys()
    assert summary["total_params"] == sum(component["params"] for component in components.values())
    assert summary["trainable_params"] == sum(
        component["trainable_params"] for component in components.values()
    )
    assert all(component["layers"] for component in components.values())
    assert all(
        {"path", "type", "detail", "params", "trainable_params"} <= layer.keys()
        for component in components.values()
        for layer in component["layers"]
    )
    if "target_transformer" in components:
        assert components["target_transformer"]["role"] == "target"
        assert components["target_transformer"]["trainable_params"] == 0

    env.close()
