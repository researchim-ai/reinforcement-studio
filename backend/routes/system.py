"""System endpoints: health, hardware info."""
from __future__ import annotations

import os
import platform
import subprocess

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/info")
async def info():
    torch_cuda = False
    try:
        import torch

        torch_cuda = torch.cuda.is_available()
    except Exception:
        pass

    gpus: list[dict] = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    gpus.append({
                        "id": int(parts[0]),
                        "name": parts[1],
                        "memory_used_mb": float(parts[2]),
                        "memory_total_mb": float(parts[3]),
                        "utilization": int(parts[4]) if parts[4].isdigit() else None,
                    })
    except Exception:
        pass

    return {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_cuda_available": torch_cuda,
        "gpus": gpus,
    }
