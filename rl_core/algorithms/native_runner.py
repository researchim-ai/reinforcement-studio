"""Entrypoint for the built-in Gym-track algorithms — `ppo`/`dqn`/`a2c` — run
with our own from-scratch PyTorch implementations (`rl_core/algorithms/native/`)
instead of Stable-Baselines3. SB3 remains an optional dependency, used only
when a custom algorithm plugin explicitly subclasses its `BaseAlgorithm`
(see `rl_core/algorithms/custom_runner.py`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from rl_core.algorithms.native.a2c import DEFAULT_HYPERPARAMS as A2C_DEFAULTS
from rl_core.algorithms.native.a2c import NativeA2C
from rl_core.algorithms.native.dqn import DEFAULT_HYPERPARAMS as DQN_DEFAULTS
from rl_core.algorithms.native.dqn import NativeDQN
from rl_core.algorithms.native.networks import policy_name
from rl_core.algorithms.native.ppo import DEFAULT_HYPERPARAMS as PPO_DEFAULTS
from rl_core.algorithms.native.ppo import NativePPO
from rl_core.algorithms.runner_utils import run_custom_algorithm

ALGO_CLASSES = {"ppo": NativePPO, "dqn": NativeDQN, "a2c": NativeA2C}
DEFAULT_HYPERPARAMS = {"ppo": PPO_DEFAULTS, "dqn": DQN_DEFAULTS, "a2c": A2C_DEFAULTS}


def run(config: dict[str, Any], run_dir: Path) -> None:
    algo_cfg = config.get("algorithm", {})
    algo_id = algo_cfg.get("id", "ppo").lower()
    if algo_id not in ALGO_CLASSES:
        raise ValueError(f"Unknown algorithm: {algo_id}")
    hyperparams = {**DEFAULT_HYPERPARAMS.get(algo_id, {}), **(algo_cfg.get("hyperparams") or {})}
    run_custom_algorithm(ALGO_CLASSES[algo_id], hyperparams, config, run_dir, algo_label=algo_id, policy_label=policy_name)
