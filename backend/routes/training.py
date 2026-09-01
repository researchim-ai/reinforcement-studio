"""Start/stop/inspect training runs (Gymnasium algorithms or AlphaZero)."""
from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend import process_manager
from rl_core.metrics_history import read_history
from rl_core.netbuilder_store import read_network_snapshot
from rl_core.paths import RUNS_DIR, run_dir
from rl_core.world_models.store import read_world_model_snapshot

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
    # A sweep member (see `backend/sweep_manager.py`) sits with a
    # `queued.flag` and no `metrics.json` at all until the scheduler starts
    # it — surfaced as its own status rather than the "unknown" a run with
    # no metrics yet would otherwise get, so the Sweeps page can show an
    # accurate progress count.
    queued = (rdir / "queued.flag").exists()
    if queued and not running and not metrics:
        status = "queued"
    else:
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
        "sweep": config.get("sweep"),
        "created_at": config.get("created_at"),
        # As seen by *this* (backend) process — correct for native/dev, but
        # a container-internal path when the backend runs inside Docker.
        # The Electron Training Monitor resolves+opens the real host path
        # itself instead (see `runs:hostPath`/`runs:openFolder` in
        # electron/main.ts); this is only a plain-text fallback for the
        # browser/web build, which has no way to open a native folder at
        # all regardless of which path it's shown.
        "run_dir": str(rdir.resolve()),
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


@router.get("/runs/{run_id}/metrics_history")
async def get_metrics_history(run_id: str):
    """Full recorded time series for this run (see `rl_core.metrics_history`)
    — unlike `metrics.json` (only ever the *latest* snapshot), this lets the
    Training Monitor rebuild the whole reward/loss curve after navigating
    away, reopening the app, or opening a run that already finished, instead
    of only ever showing whatever accumulated in memory during this one
    live websocket connection."""
    rdir = RUNS_DIR / run_id
    if not rdir.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    return {"history": read_history(rdir)}


@router.get("/runs/{run_id}/config")
async def get_run_config(run_id: str):
    """The full `config.json` this run was started with (environment +
    wrappers + algorithm hyperparams/network_spec + training settings) —
    used by Evaluation mode and by the Designer's "Дообучить" (resume)
    flow to preselect/lock the exact setup a source run trained with."""
    rdir = RUNS_DIR / run_id
    config = _read_json(rdir / "config.json")
    if config is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return config


@router.get("/runs/{run_id}/preview.gif")
async def run_preview_gif(run_id: str):
    """Serves the run's latest live-preview episode (`persist_episode_gif`
    in metrics_callback.py) straight from disk. Exists so the Training
    Monitor can display it via a plain `<img src>` instead of relying on
    `metrics.json` re-embedding the (up to a few MB) base64 blob on every
    periodic write — see `episode_gif_base64`'s throttled inclusion in
    `runner_utils.py`/`metrics_callback.py`. Always re-fetched (no-cache):
    this exact path keeps being overwritten with a newer episode as
    training progresses, unlike the environment gallery's preview GIFs."""
    path = RUNS_DIR / run_id / "episode_preview.gif"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="No preview recorded yet")
    return FileResponse(path, media_type="image/gif", headers={"Cache-Control": "no-cache"})


@router.get("/runs/{run_id}/network")
async def get_run_network(run_id: str):
    """The architecture this run actually trained with (`network.json`,
    written once at run start — see `write_network_snapshot`), so the
    Training Monitor can offer "save this architecture" without having to
    re-derive it from `config.json` (which only ever has a slug, or an
    inline spec never persisted anywhere else). `spec: null` either means
    the algorithm's own default net was used, or (for runs that predate
    this file) it's simply unknown — the UI treats both the same way:
    nothing to save."""
    rdir = RUNS_DIR / run_id
    if not rdir.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    snapshot = read_network_snapshot(rdir)
    return snapshot or {"family": None, "spec": None, "source": "unknown"}


@router.get("/runs/{run_id}/world_model")
async def get_run_world_model(run_id: str):
    """Counterpart to `get_run_network` for World Model runs (standalone
    `kind: "world_model"` runs, or any of the four world-model algorithms
    trained with a `world_model_id`) — the exact spec (`world_model.json`,
    see `rl_core.world_models.store.write_world_model_snapshot`) a run
    actually used, so the Training Monitor can offer "save/attach this
    world model" without re-deriving it from `config.json`."""
    rdir = RUNS_DIR / run_id
    if not rdir.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    snapshot = read_world_model_snapshot(rdir)
    return snapshot or {"type": None, "config": None, "world_model_id": None}


@router.get("/runs/{run_id}/latent_space.png")
async def run_latent_space_png(run_id: str):
    """Latest latent-space scatter (`rl_core.world_models.viz.render_latent_scatter`)
    for a World Model run using a recurrent-latent type (RSSM/VAE+MDN-RNN;
    the ensemble type has no single latent vector to plot at all) — written
    periodically alongside `episode_preview.gif` by both
    `rl_core/world_models/trainer.py` (standalone runs) and `dreamer.py`/
    `world_models_ha.py` if they ever grow their own preview generation.
    Always re-fetched (no-cache), same convention as `/preview.gif`."""
    path = RUNS_DIR / run_id / "latent_space.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="No latent space preview recorded yet")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-cache"})


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
