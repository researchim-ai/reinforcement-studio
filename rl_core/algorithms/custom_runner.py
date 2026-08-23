"""Entrypoint for custom Gym-track algorithm plugins (`algorithm.id ==
"custom:<slug>"`).

Two supported plugin shapes (see rl_core/algorithms/base.py and the loader):
- A subclass of `stable_baselines3.common.base_class.BaseAlgorithm` -> reuses
  the existing SB3 runner + MetricsCallback pipeline unchanged (covers "I
  just want to tweak PPO's network/train() step").
- A subclass of `CustomAlgorithm` -> a fully from-scratch loop driven by
  `TrainingCallback` (same harness as our own native PPO/DQN/A2C, see
  `rl_core/algorithms/native_runner.py`), no SB3 dependency at all.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3.common.base_class import BaseAlgorithm

from rl_core.algorithms.base import CustomAlgorithm
from rl_core.algorithms.runner_utils import run_custom_algorithm
from rl_core.algorithms.sb3_runner import _run_sb3
from rl_core.plugins.loader import load_gym_algorithm


def run(config: dict[str, Any], run_dir: Path, slug: str) -> None:
    algo_cfg = config.get("algorithm", {})
    cls, hyperparam_specs = load_gym_algorithm(slug)
    defaults = {hp["key"]: hp["default"] for hp in hyperparam_specs}
    hyperparams = {**defaults, **(algo_cfg.get("hyperparams") or {})}

    if issubclass(cls, BaseAlgorithm):
        _run_sb3(cls, f"custom:{slug}", hyperparams, config, run_dir)
        return

    assert issubclass(cls, CustomAlgorithm)
    run_custom_algorithm(cls, hyperparams, config, run_dir, algo_label=f"custom:{slug}", policy_label="custom")
