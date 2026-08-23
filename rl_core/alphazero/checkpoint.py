"""Save/load AlphaZeroNet checkpoints with enough metadata to reconstruct
the architecture (needed since the FastAPI process that serves the Arena
page is not the same process that trained the model)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from rl_core.alphazero.network import AlphaZeroNet


def save_checkpoint(net: AlphaZeroNet, path: Path, meta: dict[str, Any]) -> None:
    torch.save(
        {
            "state_dict": net.state_dict(),
            "rows": net.rows,
            "cols": net.cols,
            "action_size": net.action_size,
            "meta": meta,
        },
        path,
    )


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[Any, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    meta = payload.get("meta", {})
    custom_slug = meta.get("custom_trainer_slug")
    if custom_slug:
        # Trained by a custom AlphaZero plugin, which may use a custom
        # network architecture — reconstruct it via the same trainer class
        # (its build_network()) rather than assuming the built-in AlphaZeroNet.
        from rl_core.games import make_game
        from rl_core.plugins.loader import load_alphazero_trainer

        trainer_cls = load_alphazero_trainer(custom_slug)
        game_cls = make_game(meta["game_id"]).__class__
        trainer = trainer_cls(game_cls, meta.get("hyperparams", {}), device)
        net = trainer.net
    else:
        net = AlphaZeroNet(payload["rows"], payload["cols"], payload["action_size"])
    net.load_state_dict(payload["state_dict"])
    net.to(device)
    net.eval()
    return net, meta
