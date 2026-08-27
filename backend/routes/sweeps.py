"""Hyperparameter sweeps: launch a grid of hyperparameter values × seeds as
one batch, queued to run one at a time — see `backend/sweep_manager.py`."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend import sweep_manager
from backend.routes.training import _run_summary

router = APIRouter()


class StartSweepRequest(BaseModel):
    kind: str = "gym"
    name: str | None = None
    environment: dict[str, Any]
    algorithm: dict[str, Any]
    training: dict[str, Any] = {}
    # e.g. {"learning_rate": [0.0003, 0.001, 0.003], "gamma": [0.95, 0.99]}
    grid: dict[str, list[float]] = {}
    # Repeats every grid combination once per seed — plain multi-seed
    # (no grid at all, just several seeds of the same config) is just the
    # special case of an empty `grid`.
    seeds: list[int] = []


@router.post("/start")
async def start_sweep(req: StartSweepRequest):
    base = {
        "kind": req.kind,
        "environment": req.environment,
        "algorithm": req.algorithm,
        "training": req.training,
    }
    try:
        result = sweep_manager.create_sweep(base, req.grid, req.seeds, req.name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось создать sweep: {exc}") from exc
    return result


@router.get("")
async def list_sweeps():
    return {"sweeps": sweep_manager.list_sweeps()}


@router.get("/{sweep_id}")
async def get_sweep(sweep_id: str):
    run_ids = sweep_manager.sweep_run_ids(sweep_id)
    if not run_ids:
        raise HTTPException(status_code=404, detail="Sweep not found")
    runs = [s for rid in run_ids if (s := _run_summary(rid)) is not None]
    return {"sweep_id": sweep_id, "runs": runs}


@router.delete("/{sweep_id}")
async def remove_sweep(sweep_id: str):
    deleted = sweep_manager.delete_sweep(sweep_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Sweep not found")
    return {"success": True, "deleted_runs": deleted}
