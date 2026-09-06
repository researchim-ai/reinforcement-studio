"""Typed, JSON-driven component builder for the Zero-family algorithms.

Unlike the legacy Network Builder's free ``trunk -> heads`` graph, these
algorithms have semantic wiring that must not be changed (search protocol,
KV-cache, value-prefix LSTM and auto-sized prediction heads).  ``composite_v1``
therefore exposes only shape-safe slots: the observation encoder, hidden MLP
blocks and global architecture dimensions.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import gymnasium as gym
import torch
import torch.nn as nn

from rl_core.algorithms.native.preprocessing import is_image_space, obs_flat_dim
from rl_core.netbuilder import NetworkSpecError, _build_layer
from rl_core.world_models.nets import ObsEncoder


COMPOSITE_FAMILIES = ("efficientzero", "unizero", "researchimzero", "latentimzero")
ENCODER_KINDS = ("auto", "vector_mlp", "image_cnn", "custom")
ACTIVATIONS = ("relu", "elu", "gelu", "tanh", "leaky_relu")


def _default_encoder() -> dict[str, Any]:
    return {"kind": "auto", "layers": []}


def _default_mlp(hidden_sizes: list[int], activation: str = "elu", *, batch_norm: bool = False) -> dict[str, Any]:
    return {
        "hidden_sizes": hidden_sizes,
        "activation": activation,
        "dropout": 0.0,
        "batch_norm": batch_norm,
    }


def default_composite_spec(family: str) -> dict[str, Any]:
    if family not in COMPOSITE_FAMILIES:
        raise NetworkSpecError(f"Неизвестное composite-семейство: {family}")
    if family == "efficientzero":
        return {
            "format": "composite_v1",
            "family": family,
            "dimensions": {"latent_dim": 64, "hidden_dim": 128, "proj_dim": 64},
            "encoder": _default_encoder(),
            "components": {
                "representation": _default_mlp([128]),
                "dynamics": _default_mlp([]),
                "prediction": _default_mlp([]),
                "projector": _default_mlp([64], batch_norm=True),
                "predictor": _default_mlp([64], batch_norm=True),
            },
        }
    return {
        "format": "composite_v1",
        "family": family,
        "dimensions": {
            "embed_dim": 128,
            "num_layers": 2,
            "num_heads": 8 if family == "unizero" else 4,
            "ffn_multiplier": 4,
            "dropout": 0.1 if family == "unizero" else 0.0,
            "rotary_emb": 1,
            **({"proj_dim": 64} if family in {"researchimzero", "latentimzero"} else {}),
        },
        "encoder": _default_encoder(),
        "components": {
            "tokenizer": _default_mlp([256]),
            "heads": _default_mlp([]),
            **(
                {
                    "projector": _default_mlp([64], batch_norm=True),
                    "predictor": _default_mlp([64], batch_norm=True),
                }
                if family in {"researchimzero", "latentimzero"}
                else {}
            ),
        },
    }


def composite_spec_from_hyperparams(family: str, hyperparams: dict[str, Any]) -> dict[str, Any]:
    """Materialize a default composite spec matching legacy scalar knobs."""
    spec = default_composite_spec(family)
    dimensions = spec["dimensions"]
    if family == "efficientzero":
        for key in ("latent_dim", "hidden_dim", "proj_dim"):
            if key in hyperparams:
                dimensions[key] = hyperparams[key]
        spec["components"]["representation"]["hidden_sizes"] = [int(dimensions["hidden_dim"])]
        for name in ("projector", "predictor"):
            spec["components"][name]["hidden_sizes"] = [int(dimensions["proj_dim"])]
    else:
        dimension_keys = ("embed_dim", "num_layers", "num_heads", "dropout", "rotary_emb", "ffn_multiplier")
        for key in dimension_keys:
            if key in hyperparams:
                dimensions[key] = hyperparams[key]
        spec["components"]["tokenizer"]["hidden_sizes"] = [2 * int(dimensions["embed_dim"])]
        if family in {"researchimzero", "latentimzero"}:
            if "proj_dim" in hyperparams:
                dimensions["proj_dim"] = hyperparams["proj_dim"]
            for name in ("projector", "predictor"):
                spec["components"][name]["hidden_sizes"] = [int(dimensions["proj_dim"])]
    return validate_composite_spec(spec, family)


def is_composite_spec(spec: Any) -> bool:
    return isinstance(spec, dict) and spec.get("format") == "composite_v1"


def _positive_int(value: Any, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise NetworkSpecError(f"{label} должен быть целым числом") from exc
    if result <= 0:
        raise NetworkSpecError(f"{label} должен быть больше нуля")
    return result


def validate_composite_spec(spec: dict[str, Any], expected_family: str | None = None) -> dict[str, Any]:
    if not is_composite_spec(spec):
        raise NetworkSpecError("Ожидался формат composite_v1")
    family = str(spec.get("family", ""))
    if family not in COMPOSITE_FAMILIES:
        raise NetworkSpecError(f"Неизвестное composite-семейство: {family or '—'}")
    if expected_family is not None and family != expected_family:
        raise NetworkSpecError(f"Архитектура {family} несовместима с алгоритмом {expected_family}")

    normalized = deepcopy(spec)
    dimensions = normalized.setdefault("dimensions", {})
    encoder = normalized.setdefault("encoder", _default_encoder())
    components = normalized.setdefault("components", {})

    kind = encoder.get("kind", "auto")
    if kind not in ENCODER_KINDS:
        raise NetworkSpecError(f"Неизвестный тип энкодера: {kind}")
    if not isinstance(encoder.get("layers", []), list):
        raise NetworkSpecError("encoder.layers должен быть списком слоёв")
    if kind != "auto" and not encoder.get("layers"):
        raise NetworkSpecError(f"Для encoder.kind={kind} нужен хотя бы один слой")
    allowed_layers = {"linear", "conv2d", "maxpool2d", "flatten", "activation", "dropout", "batchnorm"}
    for index, layer in enumerate(encoder.get("layers", [])):
        if not isinstance(layer, dict) or layer.get("type") not in allowed_layers:
            raise NetworkSpecError(f"encoder.layers[{index}]: неизвестный тип слоя")
        layer_type = layer["type"]
        if layer_type == "linear":
            _positive_int(layer.get("out_features"), f"encoder.layers[{index}].out_features")
        elif layer_type == "conv2d":
            for key in ("out_channels", "kernel_size", "stride"):
                _positive_int(layer.get(key), f"encoder.layers[{index}].{key}")
            if int(layer.get("padding", 0)) < 0:
                raise NetworkSpecError(f"encoder.layers[{index}].padding не может быть отрицательным")
        elif layer_type == "maxpool2d":
            _positive_int(layer.get("kernel_size"), f"encoder.layers[{index}].kernel_size")
            _positive_int(layer.get("stride", layer.get("kernel_size")), f"encoder.layers[{index}].stride")
        elif layer_type == "activation" and layer.get("fn", "relu") not in {
            "relu", "elu", "gelu", "tanh", "sigmoid", "leaky_relu",
        }:
            raise NetworkSpecError(f"encoder.layers[{index}]: неизвестная активация")
        elif layer_type == "dropout" and not 0.0 <= float(layer.get("p", 0.5)) < 1.0:
            raise NetworkSpecError(f"encoder.layers[{index}].p должен быть в диапазоне [0, 1)")
    if any(layer.get("type") == "batchnorm" for layer in encoder.get("layers", [])):
        raise NetworkSpecError("BatchNorm в observation encoder небезопасен при search batch=1")

    if family == "efficientzero":
        for key in ("latent_dim", "hidden_dim", "proj_dim"):
            dimensions[key] = _positive_int(dimensions.get(key), key)
        required = ("representation", "dynamics", "prediction", "projector", "predictor")
    else:
        for key in ("embed_dim", "num_layers", "num_heads", "ffn_multiplier"):
            dimensions[key] = _positive_int(dimensions.get(key), key)
        dimensions["dropout"] = float(dimensions.get("dropout", 0.0))
        if not 0.0 <= dimensions["dropout"] < 1.0:
            raise NetworkSpecError("dropout должен быть в диапазоне [0, 1)")
        dimensions["rotary_emb"] = int(bool(int(dimensions.get("rotary_emb", 1))))
        if dimensions["embed_dim"] % dimensions["num_heads"] != 0:
            raise NetworkSpecError("embed_dim должен делиться на num_heads")
        if dimensions["rotary_emb"] and (dimensions["embed_dim"] // dimensions["num_heads"]) % 2:
            raise NetworkSpecError("Для RoPE размер одной attention-head должен быть чётным")
        required = ("tokenizer", "heads")
        if family in {"researchimzero", "latentimzero"}:
            dimensions["proj_dim"] = _positive_int(dimensions.get("proj_dim"), "proj_dim")
            required += ("projector", "predictor")

    for name in required:
        cfg = components.setdefault(name, _default_mlp([]))
        hidden_sizes = cfg.setdefault("hidden_sizes", [])
        if not isinstance(hidden_sizes, list):
            raise NetworkSpecError(f"{name}.hidden_sizes должен быть списком")
        cfg["hidden_sizes"] = [_positive_int(size, f"{name}.hidden_sizes") for size in hidden_sizes]
        activation = cfg.setdefault("activation", "elu")
        if activation not in ACTIVATIONS:
            raise NetworkSpecError(f"Неизвестная активация {activation} в компоненте {name}")
        cfg["dropout"] = float(cfg.get("dropout", 0.0))
        if not 0.0 <= cfg["dropout"] < 1.0:
            raise NetworkSpecError(f"{name}.dropout должен быть в диапазоне [0, 1)")
        cfg["batch_norm"] = bool(cfg.get("batch_norm", False))

    for name in ("projector", "predictor"):
        if name in components and not components[name].get("batch_norm", False):
            raise NetworkSpecError(f"{name}: BatchNorm обязателен для защиты SimSiam от collapse")
        if name in components and not components[name].get("hidden_sizes"):
            raise NetworkSpecError(f"{name}: нужен хотя бы один hidden-слой с BatchNorm")
    for name, cfg in components.items():
        if name not in ("projector", "predictor") and cfg.get("batch_norm", False):
            raise NetworkSpecError(
                f"{name}: BatchNorm небезопасен в search/inference при batch=1; доступен только SimSiam-компонентам",
            )
    return normalized


class CompositeEncoder(nn.Module):
    """Custom observation chain with the same ``out_dim`` API as ObsEncoder."""

    def __init__(self, observation_space: gym.Space, encoder_spec: dict[str, Any]) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.is_image = is_image_space(observation_space)
        kind = encoder_spec.get("kind", "auto")
        if kind == "vector_mlp" and self.is_image:
            raise NetworkSpecError("vector_mlp несовместим с изображением; используй custom + Flatten")
        if kind == "image_cnn" and not self.is_image:
            raise NetworkSpecError("image_cnn требует наблюдение-изображение [H,W,C]")
        layers = list(encoder_spec.get("layers", []))
        if not layers:
            raise NetworkSpecError(f"Для encoder.kind={kind} нужен хотя бы один слой")

        if self.is_image:
            h, w, c = observation_space.shape
            shape = [int(c), int(h), int(w)]
        else:
            shape = [obs_flat_dim(observation_space)]
        modules: list[nn.Module] = []
        for layer in layers:
            module, shape = _build_layer(layer, shape)
            modules.append(module)
        if len(shape) != 1:
            raise NetworkSpecError(f"Энкодер должен выдавать вектор, сейчас выход {shape}; добавь Flatten/Linear")
        self.net = nn.Sequential(*modules)
        self.out_dim = int(shape[0])
        self.obs_shape = (
            tuple(int(d) for d in observation_space.shape)
            if self.is_image
            else (obs_flat_dim(observation_space),)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs
        if self.is_image:
            x = x.permute(0, 3, 1, 2).float() / 255.0
        else:
            x = x.float().reshape(x.shape[0], -1)
        return self.net(x)


def build_composite_encoder(observation_space: gym.Space, spec: dict[str, Any]) -> nn.Module:
    encoder_spec = spec.get("encoder", {})
    if encoder_spec.get("kind", "auto") == "auto":
        return ObsEncoder(observation_space)
    return CompositeEncoder(observation_space, encoder_spec)


def _activation(name: str) -> nn.Module:
    return {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
    }[name]()


def build_component_mlp(
    in_dim: int,
    out_dim: int,
    config: dict[str, Any],
    *,
    final_activation: nn.Module | None = None,
) -> nn.Sequential:
    """Build a hidden MLP whose final dimensional contract is auto-sized."""
    modules: list[nn.Module] = []
    previous = in_dim
    hidden_sizes = list(config.get("hidden_sizes", []))
    for width in hidden_sizes:
        width = int(width)
        modules.append(nn.Linear(previous, width))
        if config.get("batch_norm", False):
            modules.append(nn.BatchNorm1d(width))
        modules.append(_activation(str(config.get("activation", "elu"))))
        dropout = float(config.get("dropout", 0.0))
        if dropout:
            modules.append(nn.Dropout(dropout))
        previous = width
    modules.append(nn.Linear(previous, out_dim))
    if final_activation is not None:
        modules.append(final_activation)
    return nn.Sequential(*modules)


def dimensions_from_spec(spec: dict[str, Any] | None, family: str, defaults: dict[str, Any]) -> dict[str, Any]:
    """Validated spec dimensions overlaid on legacy hyperparameter defaults."""
    result = dict(defaults)
    if spec is None:
        return result
    validated = validate_composite_spec(spec, family)
    result.update(validated["dimensions"])
    return result
