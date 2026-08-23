"""Model Zoo: browse trained checkpoints (raw runs + promoted/named saves)."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from rl_core.paths import CHECKPOINTS_DIR, RUNS_DIR

router = APIRouter()


class PromoteRequest(BaseModel):
    name: str


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


@router.delete("/checkpoints/{name}")
async def delete_checkpoint(name: str):
    deleted = False
    for suffix in (".zip", ".pt", ".meta.json"):
        p = CHECKPOINTS_DIR / f"{name}{suffix}"
        if p.exists():
            p.unlink()
            deleted = True
    if not deleted:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    return {"success": True}
