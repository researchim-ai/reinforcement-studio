"""Single place that decides CPU vs GPU for a run, shared by every runner
(native Gym algorithms, the SB3 path, and both the built-in and custom
AlphaZero trainers) so "Использовать GPU" in the Designer means the same
thing everywhere instead of only being wired up for one of the tracks."""
from __future__ import annotations

from typing import Any


def resolve_device(training_cfg: dict[str, Any]) -> str:
    """`training.use_gpu` is opt-in (default off) — most of the built-in
    envs/boards here are tiny enough that a GPU's kernel-launch overhead can
    make it *slower* than CPU, so we don't want to grab a GPU just because
    one happens to be present. When the user does opt in, fall back to CPU
    silently (rather than erroring) if no CUDA device is actually available
    — e.g. the installed torch build is CPU-only, which is the default for
    this app's own bundled Python env (see rl_core/requirements.txt)."""
    if not training_cfg.get("use_gpu", False):
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
