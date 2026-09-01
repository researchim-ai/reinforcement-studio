"""Types + defaults + the `(type, config) -> actual nn.Module(s)` factory
for World Model specs — the data-only half (mirrors
`rl_core/netbuilder_store.py`'s relationship to `rl_core/netbuilder.py`:
this module owns *what* a spec means, `rssm.py`/`ensemble.py`/
`vae_mdnrnn.py` own how each type is actually implemented).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import gymnasium as gym
import torch

from rl_core.world_models.ensemble import EnsembleDynamicsModel
from rl_core.world_models.nets import action_dim_of
from rl_core.world_models.rssm import RSSM
from rl_core.world_models.vae_mdnrnn import MDNRNN, VAE

WORLD_MODEL_TYPES = ("rssm", "ensemble", "vae_mdnrnn")

DEFAULT_CONFIG: dict[str, dict[str, Any]] = {
    "rssm": {"deter_dim": 200, "stoch_dim": 30, "hidden_dim": 200},
    "ensemble": {"ensemble_size": 5, "hidden_dim": 200},
    "vae_mdnrnn": {"latent_dim": 32, "hidden_dim": 200, "rnn_hidden_size": 256, "num_mixtures": 5},
}

# Surfaced to the World Model Builder page so it can render a plain
# numeric-field form per type without the frontend needing to hardcode
# min/max/labels for every hyperparam in TypeScript too — same rationale as
# `HyperparamSpec` in `backend/routes/environments.py`'s `ALGORITHM_CATALOG`.
CONFIG_FIELDS: dict[str, list[dict[str, Any]]] = {
    "rssm": [
        {"key": "deter_dim", "label": "Deterministic state size (GRU)", "min": 16, "max": 1024},
        {"key": "stoch_dim", "label": "Stochastic latent size", "min": 4, "max": 256},
        {"key": "hidden_dim", "label": "Hidden size (heads/encoders)", "min": 16, "max": 1024},
    ],
    "ensemble": [
        {"key": "ensemble_size", "label": "Ensemble members", "min": 2, "max": 20},
        {"key": "hidden_dim", "label": "Hidden size per member", "min": 16, "max": 1024},
    ],
    "vae_mdnrnn": [
        {"key": "latent_dim", "label": "VAE latent size (z)", "min": 2, "max": 256},
        {"key": "hidden_dim", "label": "VAE hidden size", "min": 16, "max": 1024},
        {"key": "rnn_hidden_size", "label": "MDN-RNN hidden size", "min": 16, "max": 2048},
        {"key": "num_mixtures", "label": "MDN-RNN mixture components", "min": 1, "max": 20},
    ],
}


def default_config(kind: str) -> dict[str, Any]:
    if kind not in WORLD_MODEL_TYPES:
        raise ValueError(f"Unknown world model type: {kind!r} (expected one of {WORLD_MODEL_TYPES})")
    return dict(DEFAULT_CONFIG[kind])


def merged_config(kind: str, config: dict[str, Any] | None) -> dict[str, Any]:
    return {**default_config(kind), **(config or {})}


VaeMdnRnnModel = dict[str, Union[VAE, MDNRNN]]


def build_world_model(
    kind: str, observation_space: gym.Space, action_space: gym.Space, config: dict[str, Any] | None = None,
) -> Union[RSSM, EnsembleDynamicsModel, VaeMdnRnnModel]:
    """`kind="vae_mdnrnn"` returns a plain `{"vae": VAE, "mdnrnn": MDNRNN}`
    dict rather than a single `nn.Module` — the two halves are trained with
    genuinely different data (single observations vs. `(z, action)`
    sequences) and have no shared parameters at all, unlike the RSSM/
    ensemble, which are each one cohesive module. Callers (`trainer.py`,
    `rl_core/algorithms/native/world_models_ha.py`) already know to expect
    this — see their own docstrings."""
    cfg = merged_config(kind, config)
    if kind == "rssm":
        return RSSM(
            observation_space, action_space,
            deter_dim=int(cfg["deter_dim"]), stoch_dim=int(cfg["stoch_dim"]), hidden_dim=int(cfg["hidden_dim"]),
        )
    if kind == "ensemble":
        hidden = int(cfg["hidden_dim"])
        return EnsembleDynamicsModel(
            observation_space, action_space, ensemble_size=int(cfg["ensemble_size"]), hidden=(hidden, hidden),
        )
    if kind == "vae_mdnrnn":
        latent_dim = int(cfg["latent_dim"])
        vae = VAE(observation_space, latent_dim=latent_dim, hidden_dim=int(cfg["hidden_dim"]))
        mdnrnn = MDNRNN(
            latent_dim, action_dim_of(action_space),
            hidden_size=int(cfg["rnn_hidden_size"]), num_mixtures=int(cfg["num_mixtures"]),
        )
        return {"vae": vae, "mdnrnn": mdnrnn}
    raise ValueError(f"Unknown world model type: {kind!r}")


def save_checkpoint(kind: str, model: Union[RSSM, EnsembleDynamicsModel, VaeMdnRnnModel], path: Path, config: dict[str, Any]) -> None:
    """Every checkpoint carries its own `type`/`config` alongside the raw
    weights, so `load_checkpoint` (and anything reading `<slug>.json` next
    to it) never needs a second source of truth for *which* architecture
    those weights actually belong to."""
    if kind == "vae_mdnrnn":
        payload = {
            "type": kind, "config": config,
            "vae_state_dict": model["vae"].state_dict(),
            "mdnrnn_state_dict": model["mdnrnn"].state_dict(),
        }
    else:
        payload = {"type": kind, "config": config, "state_dict": model.state_dict()}
    torch.save(payload, path)


def load_checkpoint(
    path: Path, observation_space: gym.Space, action_space: gym.Space, device: str = "cpu",
) -> tuple[str, Union[RSSM, EnsembleDynamicsModel, VaeMdnRnnModel], dict[str, Any]]:
    """Rebuilds the exact architecture a checkpoint was trained with (from
    its own saved `type`/`config`) before loading weights into it — callers
    (the four world-model algorithms, warm-starting from a saved
    `world_model_id`) never need to already know which type a given
    checkpoint file is. Returns `(type, model, config)`."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    kind = payload["type"]
    config = payload.get("config", {})
    model = build_world_model(kind, observation_space, action_space, config)
    if kind == "vae_mdnrnn":
        model["vae"].load_state_dict(payload["vae_state_dict"])
        model["mdnrnn"].load_state_dict(payload["mdnrnn_state_dict"])
        model["vae"].to(device)
        model["mdnrnn"].to(device)
    else:
        model.load_state_dict(payload["state_dict"])
        model.to(device)
    return kind, model, config


def resolve_or_build_for_algo(
    world_model_spec: dict[str, Any] | None,
    expected_kind: str,
    observation_space: gym.Space,
    action_space: gym.Space,
    device: str = "cpu",
) -> tuple[Union[RSSM, EnsembleDynamicsModel, VaeMdnRnnModel], dict[str, Any], Union[str, None]]:
    """Turns whatever `rl_core.world_models.store.resolve_world_model_spec()`
    returned (or `None`) into an actual, ready-to-train model — used
    identically by all four world-model algorithms
    (dreamer/mbpo/pets/world_models_ha) at `__init__` time.

    Falls back to a fresh, untrained, default-config model of
    `expected_kind` whenever the run doesn't reference a saved/inline world
    model at all, or references one of a *different* type than this
    algorithm actually needs (rather than erroring — the Designer's own
    dropdown already only offers matching types, but a hand-edited config
    JSON shouldn't crash a whole run over a type mismatch). Returns
    `(model, config, slug)` — `slug`, when not `None`, is where a finished
    run's trained weights should be attached back to (see
    `save_world_model_checkpoint` on each algorithm + `runner_utils.py`)."""
    if world_model_spec and world_model_spec.get("type") == expected_kind:
        config = world_model_spec.get("config") or {}
        checkpoint = world_model_spec.get("checkpoint_path")
        slug = world_model_spec.get("slug")
        if checkpoint:
            _kind, model, config = load_checkpoint(Path(checkpoint), observation_space, action_space, device)
            return model, config, slug
        return build_world_model(expected_kind, observation_space, action_space, config), config, slug
    config = default_config(expected_kind)
    return build_world_model(expected_kind, observation_space, action_space, config), config, None
