"""Built-in AlphaZero training entrypoint: self-play -> train -> arena-gate.

See `rl_core/alphazero/base.py` for the reusable trainer contract (also used
by custom AlphaZero plugins) and `rl_core/alphazero/loop.py` for the shared
iterate/checkpoint/metrics-snapshot driver — this module just wires the
built-in trainer into that driver.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from rl_core.alphazero.base import BuiltinAlphaZeroTrainer, DEFAULT_HYPERPARAMS
from rl_core.alphazero.loop import run_training_loop
from rl_core.device import resolve_device
from rl_core.games import make_game
from rl_core.netbuilder_store import resolve_network_spec

__all__ = ["DEFAULT_HYPERPARAMS", "run"]


def run(config: dict[str, Any], run_dir: Path) -> None:
    env_cfg = config.get("environment", {})
    algo_cfg = config.get("algorithm", {})
    training_cfg = config.get("training", {})

    game_id = env_cfg["id"]
    hyperparams = dict(algo_cfg.get("hyperparams") or {})
    network_spec = resolve_network_spec(config)
    if network_spec:
        hyperparams["network_spec"] = network_spec
    seed = training_cfg.get("seed")
    device = resolve_device(training_cfg)

    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

    game_cls = make_game(game_id).__class__
    trainer = BuiltinAlphaZeroTrainer(game_cls, hyperparams, device)
    run_training_loop(trainer, config, run_dir, checkpoint_meta={"hyperparams": trainer.hp})
