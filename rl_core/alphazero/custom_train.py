"""Entrypoint for custom AlphaZero trainer plugins (`algorithm.id ==
"custom:<slug>"`). Loads the user's `AlphaZeroTrainer` subclass and drives
it through the same shared loop as the built-in trainer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from rl_core.alphazero.loop import run_training_loop
from rl_core.device import resolve_device
from rl_core.games import make_game
from rl_core.netbuilder_store import resolve_network_spec, write_network_snapshot
from rl_core.plugins.loader import load_alphazero_trainer


def run(config: dict[str, Any], run_dir: Path, slug: str) -> None:
    env_cfg = config.get("environment", {})
    algo_cfg = config.get("algorithm", {})
    training_cfg = config.get("training", {})

    game_id = env_cfg["id"]
    hyperparams = dict(algo_cfg.get("hyperparams") or {})
    network_spec = resolve_network_spec(config)
    write_network_snapshot(run_dir, config, network_spec)
    if network_spec:
        hyperparams["network_spec"] = network_spec
    seed = training_cfg.get("seed")
    device = resolve_device(training_cfg)

    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

    game_cls = make_game(game_id).__class__
    trainer_cls = load_alphazero_trainer(slug)
    trainer = trainer_cls(game_cls, hyperparams, device)
    run_training_loop(
        trainer, config, run_dir,
        checkpoint_meta={"custom_trainer_slug": slug, "hyperparams": trainer.hp},
    )
