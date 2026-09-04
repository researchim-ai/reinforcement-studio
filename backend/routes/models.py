"""Model Zoo: browse trained checkpoints (raw runs + promoted/named saves)."""
from __future__ import annotations

import asyncio
import json
import shutil
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from rl_core.algorithms.evaluate import EvaluationError, evaluate_gym_run
from rl_core.paths import CHECKPOINTS_DIR, RUNS_DIR

router = APIRouter()


class PromoteRequest(BaseModel):
    name: str


class EvaluateRequest(BaseModel):
    source: str  # "run" | "checkpoint"
    id: str
    episodes: int = 10
    record_gif: bool = True
    seed: int | None = None


def _read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


@router.get("/list")
async def list_models():
    models: list[dict[str, Any]] = []

    if RUNS_DIR.exists():
        for d in RUNS_DIR.iterdir():
            if not d.is_dir():
                continue
            config = _read_json(d / "config.json")
            if not config:
                continue
            model_file = d / "model.zip" if (d / "model.zip").exists() else d / "model.pt"
            if not model_file.exists():
                continue
            metrics = _read_json(d / "metrics.json", {}) or {}
            models.append({
                "source": "run",
                "id": d.name,
                "label": config.get("name") or d.name,
                "kind": config.get("kind", "gym"),
                "environment_id": config.get("environment", {}).get("id"),
                "algorithm_id": config.get("algorithm", {}).get("id"),
                "status": metrics.get("status"),
                "size_bytes": model_file.stat().st_size,
                "modified_at": datetime.fromtimestamp(model_file.stat().st_mtime, tz=timezone.utc).isoformat(),
                "metrics": metrics,
            })

    if CHECKPOINTS_DIR.exists():
        for meta_path in CHECKPOINTS_DIR.glob("*.meta.json"):
            meta = _read_json(meta_path, {}) or {}
            model_file = meta_path.with_suffix("").with_suffix(".zip" if meta.get("kind") == "gym" else ".pt")
            if not model_file.exists():
                continue
            models.append({
                "source": "checkpoint",
                "id": meta_path.stem.replace(".meta", ""),
                "label": meta.get("name", meta_path.stem),
                "kind": meta.get("kind", "gym"),
                "environment_id": meta.get("environment_id"),
                "algorithm_id": meta.get("algorithm_id"),
                "status": "saved",
                "size_bytes": model_file.stat().st_size,
                "modified_at": datetime.fromtimestamp(model_file.stat().st_mtime, tz=timezone.utc).isoformat(),
                "metrics": meta.get("metrics", {}),
            })

    models.sort(key=lambda m: m["modified_at"], reverse=True)
    return {"models": models}


@router.post("/promote/{run_id}")
async def promote_run(run_id: str, req: PromoteRequest):
    run_path = RUNS_DIR / run_id
    config = _read_json(run_path / "config.json")
    if not config:
        raise HTTPException(status_code=404, detail="Run not found")

    kind = config.get("kind", "gym")
    src = run_path / ("model.zip" if kind == "gym" else "model.pt")
    if not src.exists():
        raise HTTPException(status_code=400, detail="Run has no saved model yet")

    safe_name = "".join(c for c in req.name if c.isalnum() or c in "-_") or run_id
    dst = CHECKPOINTS_DIR / f"{safe_name}{src.suffix}"
    shutil.copy2(src, dst)

    # Carries the run's actual network architecture along with the promoted
    # weights (see `write_network_snapshot` in `rl_core/netbuilder_store.py`)
    # so it stays reusable even after the source run itself is deleted.
    network_snapshot_src = run_path / "network.json"
    if network_snapshot_src.exists():
        shutil.copy2(network_snapshot_src, CHECKPOINTS_DIR / f"{safe_name}.network.json")

    # Also carries the run's full config (environment + wrappers +
    # hyperparams) so a promoted checkpoint remains usable for Evaluation
    # mode and "Дообучить" (resume/fine-tune) even after the source run is
    # deleted — both need the exact wrappers/hyperparams the model was
    # trained with, not just `environment_id`/`algorithm_id`.
    (CHECKPOINTS_DIR / f"{safe_name}.config.json").write_text(json.dumps(config, indent=2))

    metrics = _read_json(run_path / "metrics.json", {}) or {}
    meta = {
        "name": req.name,
        "kind": kind,
        "environment_id": config.get("environment", {}).get("id"),
        "algorithm_id": config.get("algorithm", {}).get("id"),
        "source_run_id": run_id,
        "metrics": metrics,
    }
    (CHECKPOINTS_DIR / f"{safe_name}.meta.json").write_text(json.dumps(meta, indent=2))
    return {"success": True, "id": safe_name}


@router.get("/checkpoints/{name}/network")
async def get_checkpoint_network(name: str):
    """Counterpart to `GET /training/runs/{run_id}/network` for promoted
    checkpoints — reads the `network.json` copied alongside the weights at
    promote time (see `promote_run` above); `spec: null` for anything
    promoted before this existed, or that simply used the algorithm's
    default architecture."""
    path = CHECKPOINTS_DIR / f"{name}.network.json"
    snapshot = _read_json(path)
    config = _read_json(CHECKPOINTS_DIR / f"{name}.config.json")
    if config and (snapshot is None or not snapshot.get("spec")):
        try:
            from rl_core.composite_netbuilder import COMPOSITE_FAMILIES, composite_spec_from_hyperparams

            algorithm = config.get("algorithm", {})
            family = (snapshot or {}).get("family") or algorithm.get("id")
            if family in COMPOSITE_FAMILIES:
                snapshot = {
                    **(snapshot or {}),
                    "family": family,
                    "format": "composite_v1",
                    "spec": composite_spec_from_hyperparams(family, algorithm.get("hyperparams", {})),
                    "source": (snapshot or {}).get("source", "default"),
                    "algorithm_id": family,
                    "environment_id": config.get("environment", {}).get("id"),
                }
                path.write_text(json.dumps(snapshot, indent=2))
        except Exception:
            pass
    return snapshot or {"family": None, "spec": None, "source": "unknown"}


@router.post("/evaluate")
async def evaluate_model(req: EvaluateRequest):
    """Rolls out N deterministic episodes of an already-trained model with
    no further learning — separate from the periodic in-training preview
    GIF, and the only place the app reports actual episode reward
    statistics for a finished model. Runs in a worker thread (`torch`/gym
    rollouts are all blocking, synchronous CPU/GIL-bound work) so the event
    loop stays responsive; usually finishes in well under a minute for the
    default episode count on anything but the heaviest pixel-input envs."""
    if req.source == "run":
        run_path = RUNS_DIR / req.id
        config = _read_json(run_path / "config.json")
        if config is None:
            raise HTTPException(status_code=404, detail="Run not found")
        model_path = run_path / "model.zip"
    elif req.source == "checkpoint":
        config = _read_json(CHECKPOINTS_DIR / f"{req.id}.config.json")
        if config is None:
            raise HTTPException(
                status_code=400,
                detail="У этого чекпоинта нет сохранённого конфига среды (сохранён до появления оценки)",
            )
        model_path = CHECKPOINTS_DIR / f"{req.id}.zip"
    else:
        raise HTTPException(status_code=400, detail=f"Unknown source: {req.source}")

    if config.get("kind", "gym") != "gym":
        raise HTTPException(status_code=400, detail="Оценка пока поддерживается только для Gym-алгоритмов")
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Файл модели не найден")

    try:
        return await asyncio.to_thread(
            evaluate_gym_run, config, model_path, req.episodes, req.record_gif, req.seed,
        )
    except EvaluationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/checkpoints/{name}/config")
async def get_checkpoint_config(name: str):
    """The full `config.json` this checkpoint was trained with (see
    `promote_run` above) — used by Evaluation mode (needs the exact env +
    wrappers) and by the Designer's "Дообучить" (resume) flow (needs
    hyperparams/network_spec to preselect/lock, since the architecture
    can't change once weights are loaded into it). `None` for anything
    promoted before this existed."""
    path = CHECKPOINTS_DIR / f"{name}.config.json"
    config = _read_json(path)
    if config is None:
        raise HTTPException(status_code=404, detail="У этого чекпоинта нет сохранённого конфига (промоутнут до появления этой функции)")
    return config


@router.delete("/checkpoints/{name}")
async def delete_checkpoint(name: str):
    deleted = False
    for suffix in (".zip", ".pt", ".meta.json", ".network.json", ".config.json"):
        p = CHECKPOINTS_DIR / f"{name}{suffix}"
        if p.exists():
            p.unlink()
            deleted = True
    if not deleted:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    return {"success": True}
