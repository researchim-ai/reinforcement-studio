"""CRUD + checkpoint attach for user-created World Models — the World
Model Builder page (`/world-models`). Mirrors `backend/routes/networks.py`'s
relationship to `rl_core/netbuilder_store.py` almost exactly, with one
addition networks don't need: a World Model spec can carry *trained
weights* too (`attach_checkpoint`, this file's `POST /{slug}/attach`), since
unlike a hand-designed network architecture, a World Model is genuinely
trained as its own artifact (standalone `kind: "world_model"` runs, see
`rl_core/world_models/trainer.py`) independent of any one algorithm run."""
from __future__ import annotations

import re
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from rl_core.paths import RUNS_DIR
from rl_core.world_models import store
from rl_core.world_models.spec import CONFIG_FIELDS, DEFAULT_CONFIG, WORLD_MODEL_TYPES

router = APIRouter()

WorldModelType = Literal["rssm", "ensemble", "vae_mdnrnn"]
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class WorldModelDoc(BaseModel):
    name: str
    description: str = ""
    type: WorldModelType = "rssm"
    config: dict[str, Any] = {}
    environment_id: str | None = None


class AttachRequest(BaseModel):
    run_id: str


def _check_slug(slug: str) -> None:
    if not _SLUG_RE.match(slug):
        raise HTTPException(
            status_code=400,
            detail="Идентификатор world model должен содержать только a-z, 0-9, '-' или '_' и начинаться с буквы/цифры",
        )


@router.get("/types")
async def list_types():
    """Static metadata the World Model Builder's "new model" form needs to
    render a per-type config form without hardcoding any of it in
    TypeScript — same rationale as `/networks/families`."""
    return {
        "types": [
            {"id": t, "default_config": DEFAULT_CONFIG[t], "fields": CONFIG_FIELDS[t]}
            for t in WORLD_MODEL_TYPES
        ],
    }


@router.get("")
async def list_world_models():
    return {"world_models": store.list_meta()}


@router.get("/{slug}")
async def get_world_model(slug: str):
    try:
        return {"slug": slug, **store.load(slug)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/{slug}")
async def save_world_model(slug: str, body: WorldModelDoc):
    _check_slug(slug)
    store.save(slug, body.model_dump())
    return {"success": True}


@router.delete("/{slug}")
async def delete_world_model(slug: str):
    store.delete(slug)
    return {"success": True}


@router.post("/{slug}/attach")
async def attach_checkpoint(slug: str, req: AttachRequest):
    """Promotes a finished run's trained weights into this spec — either a
    standalone `kind: "world_model"` run's `model.pt` (see `trainer.py`),
    or a `dreamer`/`mbpo`/`pets`/`world_models_ha` run's own
    `world_model.pt` (written by `runner_utils.py` only when that run's
    `world_model_id` already pointed at this exact slug — otherwise it
    won't exist, since there was never anywhere to attach it *to*)."""
    store.load(slug)  # 404s below if the slug itself doesn't exist
    run_path = RUNS_DIR / req.run_id
    candidates = [run_path / "world_model.pt", run_path / "model.pt"]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        raise HTTPException(status_code=400, detail="У этого запуска нет сохранённых весов world model")
    store.attach_checkpoint(slug, src, source_run_id=req.run_id)
    return {"success": True}
