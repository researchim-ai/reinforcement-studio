"""Cheap, read-only introspection for the Experiment Designer — computes the
effective observation/action space (env) and the resulting network shape
(input/output dims, layer summary, parameter count) *without* running any
training. Used by `GET /api/environments/inspect` so the UI can show this
live while the user is still picking env/wrappers/algorithm/hyperparams.

Works generically for built-in algorithms and for user-authored plugins
(both Gym-track `CustomAlgorithm`/SB3 subclasses and AlphaZero
`AlphaZeroTrainer` subclasses) — plugin instantiation may fail (e.g. broken
code, mid-edit), in which case `network.error` is set but `environment` is
still returned so the designer stays useful.
"""
from __future__ import annotations

from typing import Any

import torch.nn as nn

from rl_core.algorithms.native.exploration.noisy import NoisyLinear

def _find_torch_module(instance: Any) -> nn.Module | None:
    """Best-effort lookup of "the" network on an arbitrary algorithm
    instance — covers our own native algorithms (`.net`/`.q_net`), SB3
    algorithms (`.policy`), and plugins that follow either convention."""
    candidates = {k: v for k, v in vars(instance).items() if isinstance(v, nn.Module)}
    for name in ("net", "q_net", "policy", "model", "network"):
        if name in candidates:
            return candidates[name]
    return next(iter(candidates.values()), None)


def _count_params(module: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def _describe_layers(module: nn.Module, max_layers: int = 24) -> list[str]:
    """Human-readable, leaf-module-only summary (skips containers like
    `Sequential`) — good enough for a quick "what am I training" glance,
    not meant to replace `print(model)`."""
    out: list[str] = []
    for name, m in module.named_modules():
        if name == "" or list(m.children()):
            continue
        if isinstance(m, NoisyLinear):
            out.append(f"NoisyLinear({m.in_features}\u2192{m.out_features})")
        elif isinstance(m, nn.Linear):
            out.append(f"Linear({m.in_features}\u2192{m.out_features})")
        elif isinstance(m, nn.Conv2d):
            k = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
            s = m.stride[0] if isinstance(m.stride, tuple) else m.stride
            out.append(f"Conv2d({m.in_channels}\u2192{m.out_channels}, k{k}s{s})")
        elif isinstance(m, (nn.LSTM, nn.GRU)):
            kind = "LSTM" if isinstance(m, nn.LSTM) else "GRU"
            out.append(f"{kind}(hidden={m.hidden_size}, layers={m.num_layers})")
        else:
            out.append(m.__class__.__name__)
        if len(out) >= max_layers:
            out.append("\u2026")
            break
    return out


def _network_summary(module: nn.Module, policy: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    total, trainable = _count_params(module)
    return {
        "policy": policy,
        "layers": _describe_layers(module),
        "total_params": total,
        "trainable_params": trainable,
        **(extra or {}),
    }


def _io_shapes(observation_space: Any, action_space: Any) -> dict[str, Any]:
    """Input/output shape the network actually sees, straight from the env
    spaces (independent of the algorithm's internals) — this is the number
    the user asked for: "сколько входов/выходов у сети и у среды"."""
    import gymnasium as gym

    if isinstance(observation_space, gym.spaces.Discrete):
        input_shape = [int(observation_space.n)]  # one-hot encoded
    else:
        input_shape = list(observation_space.shape)
    if isinstance(action_space, gym.spaces.Discrete):
        output_shape = [int(action_space.n)]
    else:
        output_shape = list(action_space.shape)
    return {"input_shape": input_shape, "output_shape": output_shape}


def inspect_gym(env_id: str, wrapper_specs: list[dict], algo_id: str, hyperparams: dict[str, Any]) -> dict[str, Any]:
    import gymnasium as gym

    from rl_core.algorithms.sb3_runner import _make_env, _space_info

    try:
        raw_env = gym.make(env_id)
        raw_observation_space = _space_info(raw_env.observation_space)
        raw_action_space = _space_info(raw_env.action_space)
        raw_env.close()
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not a crash
        return {"environment": None, "network": None, "error": f"Не удалось создать среду: {exc}"}

    try:
        env = _make_env(env_id, wrapper_specs)
    except Exception as exc:  # noqa: BLE001
        return {
            "environment": {
                "raw_observation_space": raw_observation_space,
                "raw_action_space": raw_action_space,
                "observation_space": None,
                "action_space": None,
            },
            "network": None,
            "error": f"Не удалось применить wrapper\u2019ы: {exc}",
        }

    environment = {
        "raw_observation_space": raw_observation_space,
        "raw_action_space": raw_action_space,
        "observation_space": _space_info(env.observation_space),
        "action_space": _space_info(env.action_space),
    }

    network = None
    error = None
    try:
        network = _inspect_gym_network(env, algo_id, hyperparams or {})
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    finally:
        env.close()

    return {"environment": environment, "network": network, "error": error}


def _inspect_gym_network(env: Any, algo_id: str, hyperparams: dict[str, Any]) -> dict[str, Any]:
    from rl_core.algorithms.native.networks import (
        ActorCriticNet,
        DuelingQNetwork,
        GaussianPolicy,
        QNetwork,
        RecurrentActorCriticNet,
        RecurrentDuelingQNetwork,
        RecurrentQNetwork,
        memory_type_from_hyperparams,
        policy_name,
    )

    io = _io_shapes(env.observation_space, env.action_space)

    if isinstance(algo_id, str) and algo_id.startswith("custom:"):
        from stable_baselines3.common.base_class import BaseAlgorithm

        from rl_core.plugins.loader import load_gym_algorithm

        slug = algo_id.split(":", 1)[1]
        cls, hp_specs = load_gym_algorithm(slug)
        defaults = {hp["key"]: hp["default"] for hp in hp_specs}
        hp = {**defaults, **hyperparams}

        if issubclass(cls, BaseAlgorithm):
            from rl_core.algorithms.sb3_runner import _filter_kwargs, _policy_for_env

            policy = _policy_for_env(env)
            model = cls(policy, env, **_filter_kwargs(cls, hp))
            return {**_network_summary(model.policy, policy), **io}

        algo = cls(env, hp, None, "cpu")
        module = _find_torch_module(algo)
        if module is None:
            return {
                "policy": "custom",
                "layers": [],
                "total_params": 0,
                "trainable_params": 0,
                "note": "Плагин не выставил атрибут net/q_net/policy/model \u2014 сеть не проанализирована",
                **io,
            }
        return {**_network_summary(module, "custom"), **io}

    algo_id = (algo_id or "ppo").lower()
    memory_type = memory_type_from_hyperparams(hyperparams) if algo_id in ("dqn", "rainbow_dqn", "ppo", "a2c") else None
    noisy_kwargs = {
        "noisy": int(hyperparams.get("action_exploration", 0) or 0) == 1,
        "noisy_sigma0": float(hyperparams.get("noisy_sigma0", 0.5)),
    }
    memory_kwargs = {
        "hidden_size": int(hyperparams.get("memory_hidden_size", 128)),
        "num_layers": int(hyperparams.get("memory_num_layers", 1)),
    }
    if algo_id == "dqn":
        module = (
            RecurrentQNetwork(env.observation_space, int(env.action_space.n), memory_type, **memory_kwargs, **noisy_kwargs)
            if memory_type
            else QNetwork(env.observation_space, int(env.action_space.n), **noisy_kwargs)
        )
    elif algo_id == "rainbow_dqn":
        module = (
            RecurrentDuelingQNetwork(env.observation_space, int(env.action_space.n), memory_type, **memory_kwargs, **noisy_kwargs)
            if memory_type
            else DuelingQNetwork(env.observation_space, int(env.action_space.n), **noisy_kwargs)
        )
    elif algo_id == "sac":
        import numpy as np

        low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
        module = GaussianPolicy(env.observation_space, low, high)
    else:
        module = (
            RecurrentActorCriticNet(env.observation_space, env.action_space, memory_type, **memory_kwargs)
            if memory_type
            else ActorCriticNet(env.observation_space, env.action_space)
        )
    return {**_network_summary(module, policy_name(env.observation_space)), **io}


def inspect_alphazero(env_id: str, algo_id: str, hyperparams: dict[str, Any]) -> dict[str, Any]:
    from rl_core.games import make_game

    try:
        game = make_game(env_id)
    except Exception as exc:  # noqa: BLE001
        return {"environment": None, "network": None, "error": f"Не удалось создать игру: {exc}"}

    space = {"type": "Box", "shape": [3, game.rows, game.cols]}
    action_space = {"type": "Discrete", "shape": [], "n": game.action_size}
    environment = {
        "raw_observation_space": space,
        "raw_action_space": action_space,
        "observation_space": space,
        "action_space": action_space,
    }

    network = None
    error = None
    try:
        network = _inspect_alphazero_network(game.__class__, algo_id, hyperparams or {})
    except Exception as exc:  # noqa: BLE001
        error = str(exc)

    return {"environment": environment, "network": network, "error": error}


def _inspect_alphazero_network(game_cls: type, algo_id: str, hyperparams: dict[str, Any]) -> dict[str, Any]:
    if isinstance(algo_id, str) and algo_id.startswith("custom:"):
        from rl_core.plugins.loader import load_alphazero_trainer

        slug = algo_id.split(":", 1)[1]
        cls = load_alphazero_trainer(slug)
        label = "AlphaZeroNet (custom)"
    else:
        from rl_core.alphazero.base import BuiltinAlphaZeroTrainer

        cls = BuiltinAlphaZeroTrainer
        label = "AlphaZeroNet"

    trainer = cls(game_cls, hyperparams, "cpu")
    info = dict(trainer.network_info)
    io = {
        "input_shape": [info.get("input_planes", 3), info.get("rows"), info.get("cols")],
        "output_shape": [info.get("action_size")],
    }
    return _network_summary(trainer.net, label, extra={**info, **io})
