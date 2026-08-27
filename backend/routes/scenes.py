"""CRUD for user-designed RL scenes (Scene Builder page — `/scene-builder`)."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from rl_core import scene_store

router = APIRouter()


class SceneDoc(BaseModel):
    name: str
    description: str = ""
    world: dict[str, Any] = Field(default_factory=dict)
    objects: list[dict[str, Any]] = Field(default_factory=list)
    items: list[dict[str, Any]] = Field(default_factory=list)
    agents: list[dict[str, Any]] = Field(default_factory=list)
    episode: dict[str, Any] = Field(default_factory=dict)


@router.get("")
async def list_scenes():
    return {"scenes": scene_store.list_meta()}


@router.get("/default")
async def default_scene():
    return scene_store.default_spec()


@router.get("/{slug}")
async def get_scene(slug: str):
    try:
        return {"slug": slug, **scene_store.load(slug)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/{slug}")
async def save_scene(slug: str, body: SceneDoc):
    try:
        scene_store.validate_slug(slug)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    scene_store.save(slug, body.model_dump())
    return {"success": True, "id": scene_store.scene_env_id(slug)}


@router.delete("/{slug}")
async def delete_scene(slug: str):
    scene_store.delete(slug)
    return {"success": True}
