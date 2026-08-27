"""CRUD storage for user-designed RL scenes (JSON specs from the Scene
Builder page — `/scene-builder`). Unlike code plugins these are pure data;
the gym-compatible `SceneMultiAgentEnv` is built at train time by
`rl_core/envs/scene_env.py`."""
from __future__ import annotations

import json
import re
from typing import Any

from rl_core.paths import CUSTOM_SCENES_DIR

_SCENE_PREFIX = "scene:"
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

_DISCRETE_MOVEMENT = {"discrete4", "discrete8"}
_CONTINUOUS_MOVEMENT = {"continuous"}


def is_scene_env_id(env_id: str) -> bool:
    return env_id.startswith(_SCENE_PREFIX)


def scene_slug_from_env_id(env_id: str) -> str:
    if not is_scene_env_id(env_id):
        raise ValueError(f"Not a scene env id: {env_id}")
    return env_id[len(_SCENE_PREFIX):]


def scene_env_id(slug: str) -> str:
    return f"{_SCENE_PREFIX}{slug}"


def _path_for(slug: str):
    return CUSTOM_SCENES_DIR / f"{slug}.json"


def list_slugs() -> list[str]:
    if not CUSTOM_SCENES_DIR.exists():
        return []
    return sorted(p.stem for p in CUSTOM_SCENES_DIR.glob("*.json"))


def load(slug: str) -> dict[str, Any]:
    path = _path_for(slug)
    if not path.exists():
        raise FileNotFoundError(f"Сцена «{slug}» не найдена")
    return json.loads(path.read_text())


def save(slug: str, doc: dict[str, Any]) -> None:
    _path_for(slug).write_text(json.dumps(doc, indent=2, ensure_ascii=False))


def delete(slug: str) -> None:
    path = _path_for(slug)
    if path.exists():
        path.unlink()


def agent_count(spec: dict[str, Any]) -> int:
    return sum(max(1, int(g.get("count", 1))) for g in spec.get("agents") or [])


def action_kind(spec: dict[str, Any]) -> str:
    groups = spec.get("agents") or []
    if not groups:
        return "discrete"
    movement = (groups[0].get("movement") or {}).get("type", "discrete4")
    if movement in _CONTINUOUS_MOVEMENT:
        return "continuous"
    return "discrete"


def meta(slug: str) -> dict[str, Any]:
    doc = load(slug)
    return {
        "id": scene_env_id(slug),
        "slug": slug,
        "name": doc.get("name") or slug,
        "description": doc.get("description", ""),
        "agent_count": agent_count(doc),
        "action_kind": action_kind(doc),
    }


def list_meta() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for slug in list_slugs():
        try:
            out.append(meta(slug))
        except Exception as exc:  # noqa: BLE001
            out.append({
                "id": scene_env_id(slug),
                "slug": slug,
                "name": slug,
                "description": "",
                "agent_count": 1,
                "action_kind": "discrete",
                "broken": True,
                "error": str(exc),
            })
    return out


def default_spec(name: str = "Новая сцена") -> dict[str, Any]:
    return {
        "name": name,
        "description": "",
        "world": {"width": 20.0, "depth": 20.0, "wall_height": 2.0},
        "objects": [
            {
                "id": "wall_n", "type": "wall", "position": [0.0, 1.0, -10.0], "size": [20.0, 2.0, 1.0],
                "shape": "box", "material": {"pattern": "brick", "color": "#9ca3af", "color2": "#4b5563"},
            },
            {
                "id": "wall_s", "type": "wall", "position": [0.0, 1.0, 10.0], "size": [20.0, 2.0, 1.0],
                "shape": "box", "material": {"pattern": "brick", "color": "#9ca3af", "color2": "#4b5563"},
            },
            {
                "id": "wall_w", "type": "wall", "position": [-10.0, 1.0, 0.0], "size": [1.0, 2.0, 20.0],
                "shape": "box", "material": {"pattern": "brick", "color": "#9ca3af", "color2": "#4b5563"},
            },
            {
                "id": "wall_e", "type": "wall", "position": [10.0, 1.0, 0.0], "size": [1.0, 2.0, 20.0],
                "shape": "box", "material": {"pattern": "brick", "color": "#9ca3af", "color2": "#4b5563"},
            },
            {
                "id": "pillar", "type": "prop", "position": [6.0, 1.0, -4.0], "size": [1.0, 2.0, 1.0],
                "shape": "cylinder", "material": {"pattern": "stripes", "color": "#c084fc", "color2": "#4c1d95"},
            },
        ],
        "items": [
            {
                "id": "coin1", "type": "reward", "position": [4.0, 0.0, 4.0], "radius": 0.5,
                "reward": 1.0, "respawn": True, "cooldown_steps": 40, "shape": "crystal",
                "material": {"pattern": "dots", "color": "#22c55e", "color2": "#14532d", "emissive": True},
            },
            {
                "id": "coin2", "type": "reward", "position": [-4.0, 0.0, -3.0], "radius": 0.5,
                "reward": 1.0, "respawn": True, "cooldown_steps": 40, "shape": "pyramid",
                "material": {"pattern": "checker", "color": "#eab308", "color2": "#713f12", "emissive": True},
            },
            {
                "id": "hazard1", "type": "hazard", "position": [0.0, 0.0, 6.0], "radius": 0.6,
                "reward": -1.0, "terminate": True, "shape": "sphere",
                "material": {"pattern": "noise", "color": "#ef4444", "color2": "#7f1d1d", "emissive": True},
            },
        ],
        "agents": [
            {
                "id": "team_a",
                "count": 2,
                "team": "a",
                "spawn": {"center": [0.0, 0.0, 0.0], "radius": 4.0},
                "body_radius": 0.4,
                "movement": {"type": "discrete4", "speed": 0.5},
                "sensors": {"type": "nearest_k", "k": 4, "range": 10.0},
                "shape": "capsule",
                "material": {"pattern": "solid", "color": "#3b82f6"},
            },
        ],
        "episode": {"max_steps": 500},
    }


def validate_slug(slug: str) -> None:
    if not _SLUG_RE.match(slug):
        raise ValueError(
            "Идентификатор сцены должен содержать только a-z, 0-9, '-' или '_' "
            "и начинаться с буквы/цифры",
        )
