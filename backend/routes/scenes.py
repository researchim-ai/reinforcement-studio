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
    # MARL reward rules (predator/prey tagging, cooperative team-shared
    # reward) — see `_tag_rule`/`_team_shared_reward` in
    # rl_core/envs/scene_env.py. Empty for every non-MARL (1-team) scene.
    rules: dict[str, Any] = Field(default_factory=dict)


@router.get("")
async def list_scenes():
    return {"scenes": scene_store.list_meta()}


_TEMPLATES = {
    "default": scene_store.default_spec,
    "predator_prey": scene_store.default_predator_prey_spec,
    "team_battle": scene_store.default_team_battle_spec,
    "pack_hunt": scene_store.default_pack_hunt_spec,
    "team_battle_large": scene_store.default_team_battle_large_spec,
}


@router.get("/default")
async def default_scene(template: str = "default"):
    factory = _TEMPLATES.get(template)
    if factory is None:
        raise HTTPException(status_code=400, detail=f"Неизвестный шаблон сцены: {template}")
    return factory()


@router.get("/templates")
async def list_templates():
    return {
        "templates": [
            {"id": "default", "name": "Пустая сцена", "description": "Одна команда, коллекционирование монет"},
            {"id": "predator_prey", "name": "Хищник и жертва", "description": "2 команды, 2 роли — предатор гоняется за жертвами (MARL, ippo/qmix)"},
            {"id": "team_battle", "name": "Команда на команду", "description": "2 команды (2×2) соревнуются за монеты, общая награда внутри команды (MARL, ippo/qmix)"},
            {"id": "pack_hunt", "name": "Стая против жертв", "description": "4 хищника кооперативно охотятся на 2 быстрых жертв, общая награда стаи (MARL, ippo/qmix)"},
            {"id": "team_battle_large", "name": "Командная битва 3×3", "description": "Увеличенная арена, 3×3 агента, общая награда внутри команды (MARL, ippo/qmix)"},
        ],
    }


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
