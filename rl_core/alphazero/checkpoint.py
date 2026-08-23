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


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[AlphaZeroNet, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    net = AlphaZeroNet(payload["rows"], payload["cols"], payload["action_size"])
    net.load_state_dict(payload["state_dict"])
    net.to(device)
    net.eval()
    return net, payload.get("meta", {})
