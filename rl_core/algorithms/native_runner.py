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
from rl_core.algorithms.native.ddpg import DEFAULT_HYPERPARAMS as DDPG_DEFAULTS
from rl_core.algorithms.native.ddpg import NativeDDPG
from rl_core.algorithms.native.dqn import DEFAULT_HYPERPARAMS as DQN_DEFAULTS
from rl_core.algorithms.native.dqn import NativeDQN
from rl_core.algorithms.native.es import DEFAULT_HYPERPARAMS as ES_DEFAULTS
from rl_core.algorithms.native.es import NativeES
from rl_core.algorithms.native.marl_ppo import DEFAULT_HYPERPARAMS as IPPO_DEFAULTS
from rl_core.algorithms.native.marl_ppo import MultiAgentPPO
from rl_core.algorithms.native.networks import policy_name
from rl_core.algorithms.native.ppo import DEFAULT_HYPERPARAMS as PPO_DEFAULTS
from rl_core.algorithms.native.ppo import NativePPO
from rl_core.algorithms.native.rainbow_dqn import DEFAULT_HYPERPARAMS as RAINBOW_DQN_DEFAULTS
from rl_core.algorithms.native.rainbow_dqn import NativeRainbowDQN
from rl_core.algorithms.native.sac import DEFAULT_HYPERPARAMS as SAC_DEFAULTS
from rl_core.algorithms.native.sac import NativeSAC
from rl_core.algorithms.native.td3 import DEFAULT_HYPERPARAMS as TD3_DEFAULTS
from rl_core.algorithms.native.td3 import NativeTD3
from rl_core.algorithms.runner_utils import run_custom_algorithm

ALGO_CLASSES = {
    "ppo": NativePPO,
    "dqn": NativeDQN,
    "a2c": NativeA2C,
    "rainbow_dqn": NativeRainbowDQN,
    "sac": NativeSAC,
    "ddpg": NativeDDPG,
    "td3": NativeTD3,
    "es": NativeES,
    "ippo": MultiAgentPPO,
}
DEFAULT_HYPERPARAMS = {
    "ppo": PPO_DEFAULTS,
    "dqn": DQN_DEFAULTS,
    "a2c": A2C_DEFAULTS,
    "rainbow_dqn": RAINBOW_DQN_DEFAULTS,
    "sac": SAC_DEFAULTS,
    "ddpg": DDPG_DEFAULTS,
    "td3": TD3_DEFAULTS,
    "es": ES_DEFAULTS,
    "ippo": IPPO_DEFAULTS,
}


def run(config: dict[str, Any], run_dir: Path) -> None:
    algo_cfg = config.get("algorithm", {})
    algo_id = algo_cfg.get("id", "ppo").lower()
    if algo_id not in ALGO_CLASSES:
        raise ValueError(f"Unknown algorithm: {algo_id}")
    hyperparams = {**DEFAULT_HYPERPARAMS.get(algo_id, {}), **(algo_cfg.get("hyperparams") or {})}
    run_custom_algorithm(ALGO_CLASSES[algo_id], hyperparams, config, run_dir, algo_label=algo_id, policy_label=policy_name)
