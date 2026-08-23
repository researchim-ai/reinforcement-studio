"""CRUD + validation for user-authored plugins (custom Gym/AlphaZero
algorithms and reward functions edited in-app via the Plugins page)."""
from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from rl_core.paths import CUSTOM_ALPHAZERO_ALGOS_DIR, CUSTOM_GYM_ALGOS_DIR, CUSTOM_REWARDS_DIR
from rl_core.plugins import loader
from rl_core.plugins.templates import TEMPLATES

router = APIRouter()

PluginKind = Literal["gym-algorithms", "alphazero-algorithms", "reward-functions"]

_KIND_CONFIG = {
    "gym-algorithms": {
        "dir": CUSTOM_GYM_ALGOS_DIR,
        "meta": loader.gym_algorithm_meta,
        "list_slugs": loader.list_gym_algorithm_slugs,
        "validate_at": loader.validate_gym_algorithm_at,
    },
    "alphazero-algorithms": {
        "dir": CUSTOM_ALPHAZERO_ALGOS_DIR,
        "meta": loader.alphazero_trainer_meta,
        "list_slugs": loader.list_alphazero_trainer_slugs,
        "validate_at": loader.validate_alphazero_trainer_at,
    },
    "reward-functions": {
        "dir": CUSTOM_REWARDS_DIR,
        "meta": loader.reward_fn_meta,
        "list_slugs": loader.list_reward_fn_slugs,
        "validate_at": loader.validate_reward_fn_at,
    },
}

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class PluginCode(BaseModel):
    code: str


def _config(kind: str) -> dict:
    cfg = _KIND_CONFIG.get(kind)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"Unknown plugin kind: {kind}")
    return cfg


def _check_slug(slug: str) -> None:
    if not _SLUG_RE.match(slug):
        raise HTTPException(
            status_code=400,
            detail="Идентификатор скрипта должен содержать только a-z, 0-9, '-' или '_' и начинаться с буквы/цифры",
        )


def _path_for(kind: str, slug: str) -> Path:
    return _config(kind)["dir"] / f"{slug}.py"


@router.get("/templates")
async def list_templates():
    return {"templates": TEMPLATES}


@router.get("/{kind}")
async def list_scripts(kind: PluginKind):
    cfg = _config(kind)
    scripts = []
    for slug in cfg["list_slugs"]():
        try:
            scripts.append(cfg["meta"](slug))
        except Exception as exc:  # noqa: BLE001 - a broken file shouldn't hide the rest of the list
            scripts.append({"id": slug, "slug": slug, "name": slug, "description": "", "hyperparams": [],
                             "is_custom": True, "broken": True, "error": str(exc)})
    return {"scripts": scripts}


@router.get("/{kind}/{slug}")
async def get_script(kind: PluginKind, slug: str):
    path = _path_for(kind, slug)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Script not found")
    return {"slug": slug, "code": path.read_text()}


@router.put("/{kind}/{slug}")
async def save_script(kind: PluginKind, slug: str, body: PluginCode):
    _check_slug(slug)
    path = _path_for(kind, slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.code)
    return {"success": True}


@router.delete("/{kind}/{slug}")
async def delete_script(kind: PluginKind, slug: str):
    path = _path_for(kind, slug)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Script not found")
    path.unlink()
    return {"success": True}


@router.post("/{kind}/{slug}/validate")
async def validate_script(kind: PluginKind, slug: str, body: PluginCode):
    cfg = _config(kind)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / f"{slug}.py"
        tmp_path.write_text(body.code)
        result = cfg["validate_at"](tmp_path)
    return result
