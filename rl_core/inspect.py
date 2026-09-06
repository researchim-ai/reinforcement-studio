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


def _layer_detail(module: nn.Module) -> str:
    """Compact dimensions/config for one leaf module."""
    if isinstance(module, NoisyLinear):
        return f"{module.in_features}\u2192{module.out_features}"
    if isinstance(module, nn.Linear):
        return f"{module.in_features}\u2192{module.out_features}"
    if isinstance(module, nn.Conv2d):
        kernel = "\u00d7".join(str(v) for v in module.kernel_size)
        stride = "\u00d7".join(str(v) for v in module.stride)
        return f"{module.in_channels}\u2192{module.out_channels}, k={kernel}, s={stride}"
    if isinstance(module, nn.Embedding):
        return f"{module.num_embeddings}\u00d7{module.embedding_dim}"
    if isinstance(module, nn.LayerNorm):
        return f"shape={tuple(module.normalized_shape)}"
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        return f"features={module.num_features}"
    if isinstance(module, (nn.LSTM, nn.GRU)):
        return f"in={module.input_size}, hidden={module.hidden_size}, layers={module.num_layers}"
    if isinstance(module, nn.Dropout):
        return f"p={module.p:g}"
    if isinstance(module, nn.Flatten):
        return f"dim {module.start_dim}\u2026{module.end_dim}"
    return ""


def _architecture_components(instance: Any) -> tuple[list[dict[str, Any]], int, int]:
    """Describe every top-level torch component an algorithm owns.

    Model-based algorithms are systems of several modules, so selecting
    only `.net` (or the first module) is materially wrong for monitoring.
    Components retain construction order and expose every leaf module,
    including its qualified path, dimensions and own parameter count.
    """
    components: list[dict[str, Any]] = []
    seen_modules: set[int] = set()
    seen_parameters: set[int] = set()
    total_params = 0
    trainable_params = 0
    optimizer_parameter_ids = {
        id(parameter)
        for value in vars(instance).values()
        if hasattr(value, "param_groups")
        for group in value.param_groups
        for parameter in group.get("params", [])
    }

    def is_trainable(parameter: nn.Parameter) -> bool:
        return parameter.requires_grad and (
            not optimizer_parameter_ids or id(parameter) in optimizer_parameter_ids
        )

    for component_name, component in vars(instance).items():
        if not isinstance(component, nn.Module) or id(component) in seen_modules:
            continue
        seen_modules.add(id(component))
        component_total = sum(parameter.numel() for parameter in component.parameters())
        component_trainable = sum(
            parameter.numel() for parameter in component.parameters() if is_trainable(parameter)
        )
        layers: list[dict[str, Any]] = []
        for path, layer in component.named_modules():
            if path == "" or list(layer.children()):
                continue
            own_parameters = list(layer.parameters(recurse=False))
            layers.append({
                "path": path,
                "type": layer.__class__.__name__,
                "detail": _layer_detail(layer),
                "params": sum(parameter.numel() for parameter in own_parameters),
                "trainable_params": sum(
                    parameter.numel() for parameter in own_parameters if is_trainable(parameter)
                ),
            })
        components.append({
            "name": component_name,
            "type": component.__class__.__name__,
            "params": component_total,
            "trainable_params": component_trainable,
            "role": "target" if component_name.startswith("target_") else "online",
            "layers": layers,
        })
        for parameter in component.parameters():
            if id(parameter) in seen_parameters:
                continue
            seen_parameters.add(id(parameter))
            total_params += parameter.numel()
            if is_trainable(parameter):
                trainable_params += parameter.numel()

    # A few algorithms own trainable scalar/vector parameters directly
    # rather than through a module (e.g. adaptive loss log-variances).
    direct_parameters = [
        (name, value) for name, value in vars(instance).items()
        if isinstance(value, nn.Parameter) and id(value) not in seen_parameters
    ]
    if direct_parameters:
        direct_total = sum(parameter.numel() for _, parameter in direct_parameters)
        direct_trainable = sum(
            parameter.numel() for _, parameter in direct_parameters if is_trainable(parameter)
        )
        components.append({
            "name": "direct_parameters",
            "type": "Parameters",
            "params": direct_total,
            "trainable_params": direct_trainable,
            "role": "online",
            "layers": [
                {
                    "path": name,
                    "type": "Parameter",
                    "detail": f"shape={tuple(parameter.shape)}",
                    "params": parameter.numel(),
                    "trainable_params": parameter.numel() if is_trainable(parameter) else 0,
                }
                for name, parameter in direct_parameters
            ],
        })
        total_params += direct_total
        trainable_params += direct_trainable

    return components, total_params, trainable_params


def _algorithm_architecture_summary(instance: Any) -> dict[str, Any]:
    components, total, trainable = _architecture_components(instance)
    return {
        "architecture_components": components,
        "total_params": total,
        "trainable_params": trainable,
    }


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
    from rl_core.algorithms.vec_env import action_space as single_action_space, obs_space
    from rl_core.envs.factory import is_shared_world_env_id, make_inspect_env

    try:
        if is_shared_world_env_id(env_id):
            raw_env = make_inspect_env(env_id, wrapper_specs)
            raw_observation_space = _space_info(obs_space(raw_env))
            raw_action_space = _space_info(single_action_space(raw_env))
            raw_env.close()
        else:
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
        "observation_space": _space_info(obs_space(env)),
        "action_space": _space_info(single_action_space(env)),
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
    from rl_core.algorithms.vec_env import action_space as vec_action_space, obs_space
    from rl_core.algorithms.native.networks import (
        ActorCriticNet,
        DeterministicPolicy,
        DuelingQNetwork,
        ESPolicyNet,
        GaussianPolicy,
        QNetwork,
        QuantileDuelingQNetwork,
        RecurrentActorCriticNet,
        RecurrentDuelingQNetwork,
        RecurrentQNetwork,
        memory_type_from_hyperparams,
        policy_name,
    )

    obs_sp = obs_space(env)
    act_sp = vec_action_space(env)
    io = _io_shapes(obs_sp, act_sp)

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
        return {
            **_network_summary(module, "custom"),
            **_algorithm_architecture_summary(algo),
            **io,
        }

    algo_id = (algo_id or "ppo").lower()
    if algo_id in ("efficientzero", "unizero", "researchimzero", "latentimzero"):
        if algo_id == "efficientzero":
            from rl_core.algorithms.native.efficientzero import DEFAULT_HYPERPARAMS, NativeEfficientZero

            algorithm_cls = NativeEfficientZero
            label = "EfficientZero world model"
        elif algo_id == "unizero":
            from rl_core.algorithms.native.unizero import DEFAULT_HYPERPARAMS, NativeUniZero

            algorithm_cls = NativeUniZero
            label = "UniZero Transformer world model"
        elif algo_id == "researchimzero":
            from rl_core.algorithms.native.researchimzero import DEFAULT_HYPERPARAMS, NativeResearchImZero

            algorithm_cls = NativeResearchImZero
            label = "ResearchImZero Transformer world model"
        else:
            from rl_core.algorithms.native.latentimzero import DEFAULT_HYPERPARAMS, NativeLatentImZero

            algorithm_cls = NativeLatentImZero
            label = "LatentImZero v7 Research core + uncertainty sidecar"
        algorithm = algorithm_cls(env, {**DEFAULT_HYPERPARAMS, **hyperparams}, 0, "cpu")
        module = _find_torch_module(algorithm)
        return {
            **(_network_summary(module, label) if module is not None else {"policy": label, "layers": []}),
            **_algorithm_architecture_summary(algorithm),
            **io,
        }

    # A hand-designed architecture (Network Builder page, or the Designer's
    # quick layer editor) — see `rl_core/algorithms/native/{ppo,a2c,dqn,
    # rainbow_dqn}.py`, which all check for this exact key and build the
    # matching `Spec*Net` instead of their fixed-architecture default. Takes
    # priority over memory/NoisyNet/distributional, exactly like those
    # runtime algorithms do, so the preview never disagrees with the actual
    # run.
    network_spec = hyperparams.get("network_spec")
    memory_type = memory_type_from_hyperparams(hyperparams) if algo_id in ("dqn", "rainbow_dqn", "ppo", "a2c") else None
    noisy_kwargs = {
        "noisy": int(hyperparams.get("action_exploration", 0) or 0) == 1,
        "noisy_sigma0": float(hyperparams.get("noisy_sigma0", 0.5)),
    }
    memory_kwargs = {
        "hidden_size": int(hyperparams.get("memory_hidden_size", 128)),
        "num_layers": int(hyperparams.get("memory_num_layers", 1)),
    }
    if network_spec and algo_id in ("dqn", "rainbow_dqn", "ppo", "a2c"):
        from rl_core.netbuilder import SpecActorCriticNet, SpecDuelingQNetwork, SpecQNetwork

        if algo_id == "dqn":
            module = SpecQNetwork(obs_sp, int(act_sp.n), network_spec)
        elif algo_id == "rainbow_dqn":
            module = SpecDuelingQNetwork(obs_sp, int(act_sp.n), network_spec)
        else:
            module = SpecActorCriticNet(obs_sp, act_sp, network_spec)
    elif algo_id == "dqn":
        module = (
            RecurrentQNetwork(obs_sp, int(act_sp.n), memory_type, **memory_kwargs, **noisy_kwargs)
            if memory_type
            else QNetwork(obs_sp, int(act_sp.n), **noisy_kwargs)
        )
    elif algo_id == "rainbow_dqn":
        distributional = bool(int(hyperparams.get("distributional", 0) or 0)) and not memory_type
        if memory_type:
            module = RecurrentDuelingQNetwork(obs_sp, int(act_sp.n), memory_type, **memory_kwargs, **noisy_kwargs)
        elif distributional:
            module = QuantileDuelingQNetwork(
                obs_sp, int(act_sp.n),
                num_quantiles=max(2, int(hyperparams.get("num_quantiles", 51))), **noisy_kwargs,
            )
        else:
            module = DuelingQNetwork(obs_sp, int(act_sp.n), **noisy_kwargs)
    elif algo_id in ("sac", "ddpg", "td3"):
        import numpy as np

        low = np.asarray(act_sp.low, dtype=np.float32).reshape(-1)
        high = np.asarray(act_sp.high, dtype=np.float32).reshape(-1)
        module = GaussianPolicy(obs_sp, low, high) if algo_id == "sac" else DeterministicPolicy(obs_sp, low, high)
    elif algo_id == "es":
        module = ESPolicyNet(obs_sp, act_sp)
    else:
        module = (
            RecurrentActorCriticNet(obs_sp, act_sp, memory_type, **memory_kwargs)
            if memory_type
            else ActorCriticNet(obs_sp, act_sp)
        )
    policy = "custom-net" if (network_spec and algo_id in ("dqn", "rainbow_dqn", "ppo", "a2c")) else policy_name(obs_sp)
    return {**_network_summary(module, policy), **io}


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
