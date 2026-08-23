"""Start/stop/inspect training runs (Gymnasium algorithms or AlphaZero)."""
from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend import process_manager
from rl_core.paths import RUNS_DIR, run_dir

router = APIRouter()


class StartRunRequest(BaseModel):
    kind: str  # "gym" | "alphazero"
    environment: dict[str, Any]
    algorithm: dict[str, Any]
    training: dict[str, Any] = {}
    name: str | None = None


def _read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _run_summary(run_id: str) -> dict[str, Any] | None:
    rdir = RUNS_DIR / run_id
    config = _read_json(rdir / "config.json")
    if config is None:
        return None
    metrics = _read_json(rdir / "metrics.json", {}) or {}
    running = process_manager.is_running(run_id)
    status = metrics.get("status", "running" if running else "unknown")
    if not running and status == "running":
        status = "interrupted"
    has_model = (rdir / "model.zip").exists() or (rdir / "model.pt").exists()
    return {
        "run_id": run_id,
        "name": config.get("name") or run_id,
        "kind": config.get("kind", "gym"),
        "environment_id": config.get("environment", {}).get("id"),
        "algorithm_id": config.get("algorithm", {}).get("id"),
        "status": status,
        "running": running,
        "metrics": metrics,
        "has_model": has_model,
        "created_at": config.get("created_at"),
    }


@router.post("/start")
async def start_run(req: StartRunRequest):
    run_id = f"{req.environment.get('id', 'run').lower().replace(' ', '-')}-{uuid.uuid4().hex[:8]}"
    rdir = run_dir(run_id)

    config = {
        "run_id": run_id,
        "kind": req.kind,
        "name": req.name,
        "environment": req.environment,
        "algorithm": req.algorithm,
        "training": req.training,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    config_path = rdir / "config.json"
    config_path.write_text(json.dumps(config, indent=2))

    try:
        process_manager.start_run(run_id, config_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start run: {exc}") from exc

    return {"run_id": run_id}


@router.post("/stop/{run_id}")
async def stop_run(run_id: str):
    if not (RUNS_DIR / run_id).exists():
        raise HTTPException(status_code=404, detail="Run not found")
    process_manager.stop_run(run_id)
    return {"success": True}


@router.get("/runs")
async def list_runs():
    if not RUNS_DIR.exists():
        return {"runs": []}
    runs = []
    for d in sorted(RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        summary = _run_summary(d.name)
        if summary:
            runs.append(summary)
    return {"runs": runs}


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    summary = _run_summary(run_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return summary


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str):
    if process_manager.is_running(run_id):
        raise HTTPException(status_code=400, detail="Stop the run before deleting it")
    rdir = RUNS_DIR / run_id
    if not rdir.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    shutil.rmtree(rdir)
    return {"success": True}


@router.get("/runs/{run_id}/logs")
async def run_logs(run_id: str, lines: int = 200):
    rdir = RUNS_DIR / run_id
    if not rdir.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    def tail(path, n):
        if not path.exists():
            return ""
        content = path.read_text(errors="replace").splitlines()
        return "\n".join(content[-n:])

    return {
        "run_id": run_id,
        "stdout": tail(rdir / "stdout.log", lines),
        "stderr": tail(rdir / "stderr.log", lines),
        "error": tail(rdir / "error.log", lines),
    }


@router.get("/runs/{run_id}/games")
async def list_self_play_games(run_id: str):
    games_dir = RUNS_DIR / run_id / "games"
    if not games_dir.exists():
        return {"iterations": []}
    files = sorted(games_dir.glob("iteration_*.json"))
    return {"iterations": [f.stem.replace("iteration_", "") for f in files]}


@router.get("/runs/{run_id}/games/{iteration}")
async def get_self_play_games(run_id: str, iteration: str):
    path = RUNS_DIR / run_id / "games" / f"iteration_{iteration}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Iteration not found")
    return {"iteration": iteration, "games": _read_json(path, [])}
