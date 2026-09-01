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


def team_count(spec: dict[str, Any]) -> int:
    """Number of *distinct* `team` labels across agent groups — a scene
    with `team_count() >= 2` is what makes it a genuine MARL environment
    (independent, potentially adversarial rewards per team) rather than
    just N parameter-sharing copies of one policy. `team` defaults to
    `"default"` when unset, so every pre-existing single-group scene
    (before this field existed) still reports `team_count() == 1` and
    keeps training with the plain single-policy algorithms exactly as
    before — `ippo` (see `rl_core/algorithms/native/marl_ppo.py`) only
    becomes selectable once a scene actually defines 2+ teams."""
    return len({str(g.get("team") or "default") for g in spec.get("agents") or []}) or 1


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
        "team_count": team_count(doc),
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
                "team_count": 1,
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


def default_predator_prey_spec(name: str = "Хищник и жертва") -> dict[str, Any]:
    """2 teams, 2 roles — the classic predator-prey MARL benchmark: a
    lone (fast) predator chases 3 (slower) prey, prey survive by grabbing
    coins while avoiding the predator, predator scores by tagging prey
    (see `rules.tag` in `rl_core/envs/scene_env.py`). Trains with `ippo`
    (Multi-Agent PPO, `rl_core/algorithms/native/marl_ppo.py`) — each team
    gets its *own* policy, so the predator's and prey's rewards never mix
    into one training signal the way single-policy parameter sharing
    would."""
    return {
        "name": name,
        "description": "Хищник (команда predator) ловит жертв (команда prey); жертвы собирают монеты и убегают.",
        "world": {"width": 24.0, "depth": 24.0, "wall_height": 2.0},
        "objects": [
            {
                "id": "wall_n", "type": "wall", "position": [0.0, 1.0, -12.0], "size": [24.0, 2.0, 1.0],
                "shape": "box", "material": {"pattern": "brick", "color": "#9ca3af", "color2": "#4b5563"},
            },
            {
                "id": "wall_s", "type": "wall", "position": [0.0, 1.0, 12.0], "size": [24.0, 2.0, 1.0],
                "shape": "box", "material": {"pattern": "brick", "color": "#9ca3af", "color2": "#4b5563"},
            },
            {
                "id": "wall_w", "type": "wall", "position": [-12.0, 1.0, 0.0], "size": [1.0, 2.0, 24.0],
                "shape": "box", "material": {"pattern": "brick", "color": "#9ca3af", "color2": "#4b5563"},
            },
            {
                "id": "wall_e", "type": "wall", "position": [12.0, 1.0, 0.0], "size": [1.0, 2.0, 24.0],
                "shape": "box", "material": {"pattern": "brick", "color": "#9ca3af", "color2": "#4b5563"},
            },
        ],
        "items": [
            {
                "id": f"coin{i}", "type": "reward",
                "position": [float(x), 0.0, float(z)], "radius": 0.5,
                "reward": 1.0, "respawn": True, "cooldown_steps": 30, "shape": "crystal",
                "restrict_team": "prey",
                "material": {"pattern": "dots", "color": "#22c55e", "color2": "#14532d", "emissive": True},
            }
            for i, (x, z) in enumerate([(6, 6), (-6, 6), (6, -6), (-6, -6), (0, 0)])
        ],
        "agents": [
            {
                "id": "predator", "count": 1, "team": "predator", "role": "predator",
                "spawn": {"center": [0.0, 0.0, 0.0], "radius": 2.0}, "body_radius": 0.5,
                "movement": {"type": "discrete8", "speed": 0.65}, "sensors": {"type": "nearest_k", "k": 6, "range": 14.0},
                "shape": "cone", "material": {"pattern": "solid", "color": "#ef4444"},
            },
            {
                "id": "prey", "count": 3, "team": "prey", "role": "prey",
                "spawn": {"center": [0.0, 0.0, 0.0], "radius": 10.0}, "body_radius": 0.35,
                "movement": {"type": "discrete8", "speed": 0.65}, "sensors": {"type": "nearest_k", "k": 6, "range": 14.0},
                "shape": "capsule", "material": {"pattern": "solid", "color": "#3b82f6"},
            },
        ],
        "rules": {
            "tag": {
                "enabled": True, "predator_role": "predator", "prey_role": "prey",
                "predator_reward": 2.0, "prey_reward": -2.0,
                "prey_terminates": False, "prey_respawns": True,
            },
            "team_shared_reward": False,
        },
        "episode": {"max_steps": 400},
    }


def default_team_battle_spec(name: str = "Команда на команду") -> dict[str, Any]:
    """2 same-role teams (red vs. blue) racing for the same open coins —
    whichever agent physically reaches one first collects it (no
    `restrict_team` lock, unlike predator-prey's per-role item rule), so
    this is a symmetric competitive resource-race rather than
    predator-prey's asymmetric chase. `team_shared_reward` is on: a coin
    any red agent collects counts toward *every* red agent's reward
    (cooperative within the team, competitive across teams) — the
    standard CTDE-friendly setup `ippo` is meant for."""
    return {
        "name": name,
        "description": "Команда «красные» против команды «синие» — обе гонятся за одними монетами, награда общая внутри команды.",
        "world": {"width": 24.0, "depth": 24.0, "wall_height": 2.0},
        "objects": [
            {
                "id": "wall_n", "type": "wall", "position": [0.0, 1.0, -12.0], "size": [24.0, 2.0, 1.0],
                "shape": "box", "material": {"pattern": "brick", "color": "#9ca3af", "color2": "#4b5563"},
            },
            {
                "id": "wall_s", "type": "wall", "position": [0.0, 1.0, 12.0], "size": [24.0, 2.0, 1.0],
                "shape": "box", "material": {"pattern": "brick", "color": "#9ca3af", "color2": "#4b5563"},
            },
            {
                "id": "wall_w", "type": "wall", "position": [-12.0, 1.0, 0.0], "size": [1.0, 2.0, 24.0],
                "shape": "box", "material": {"pattern": "brick", "color": "#9ca3af", "color2": "#4b5563"},
            },
            {
                "id": "wall_e", "type": "wall", "position": [12.0, 1.0, 0.0], "size": [1.0, 2.0, 24.0],
                "shape": "box", "material": {"pattern": "brick", "color": "#9ca3af", "color2": "#4b5563"},
            },
        ],
        "items": [
            {
                "id": f"coin{i}", "type": "reward",
                "position": [float(x), 0.0, float(z)], "radius": 0.5,
                "reward": 1.0, "respawn": True, "cooldown_steps": 20, "shape": "crystal",
                "material": {"pattern": "dots", "color": "#eab308", "color2": "#713f12", "emissive": True},
            }
            for i, (x, z) in enumerate([(0, 0), (5, 0), (-5, 0), (0, 5), (0, -5), (5, 5), (-5, -5)])
        ],
        "agents": [
            {
                "id": "red", "count": 2, "team": "red", "role": "agent",
                "spawn": {"center": [-8.0, 0.0, 0.0], "radius": 3.0}, "body_radius": 0.4,
                "movement": {"type": "discrete8", "speed": 0.55}, "sensors": {"type": "nearest_k", "k": 6, "range": 14.0},
                "shape": "capsule", "material": {"pattern": "solid", "color": "#ef4444"},
            },
            {
                "id": "blue", "count": 2, "team": "blue", "role": "agent",
                "spawn": {"center": [8.0, 0.0, 0.0], "radius": 3.0}, "body_radius": 0.4,
                "movement": {"type": "discrete8", "speed": 0.55}, "sensors": {"type": "nearest_k", "k": 6, "range": 14.0},
                "shape": "capsule", "material": {"pattern": "solid", "color": "#3b82f6"},
            },
        ],
        "rules": {"team_shared_reward": True},
        "episode": {"max_steps": 400},
    }


def default_pack_hunt_spec(name: str = "Стая против жертв") -> dict[str, Any]:
    """2 teams, reversed ratio from `default_predator_prey_spec` (4
    cooperative hunters vs. 2 evasive runners, not 1-vs-3) — a genuinely
    cooperative-pursuit MARL benchmark (in the spirit of the classic
    "Pursuit"/"Predator-Prey" multi-agent particle environments): no
    single hunter can reliably corner a runner alone, so the hunters'
    `team_shared_reward` (every tag anyone on the team lands counts for
    the whole pack) is what actually makes coordination pay off — a good
    fit for `qmix` (rl_core/algorithms/native/qmix.py), whose mixing
    network is only interesting once a team has several agents to
    combine. Larger arena + more obstacles than `default_predator_prey_spec`
    so a lone hunter chasing head-on is rarely the fastest way to a tag."""
    return {
        "name": name,
        "description": "4 хищника (команда hunters, общая награда) кооперативно окружают 2 быстрых жертв (команда runners).",
        "world": {"width": 30.0, "depth": 30.0, "wall_height": 2.0},
        "objects": [
            {
                "id": "wall_n", "type": "wall", "position": [0.0, 1.0, -15.0], "size": [30.0, 2.0, 1.0],
                "shape": "box", "material": {"pattern": "brick", "color": "#9ca3af", "color2": "#4b5563"},
            },
            {
                "id": "wall_s", "type": "wall", "position": [0.0, 1.0, 15.0], "size": [30.0, 2.0, 1.0],
                "shape": "box", "material": {"pattern": "brick", "color": "#9ca3af", "color2": "#4b5563"},
            },
            {
                "id": "wall_w", "type": "wall", "position": [-15.0, 1.0, 0.0], "size": [1.0, 2.0, 30.0],
                "shape": "box", "material": {"pattern": "brick", "color": "#9ca3af", "color2": "#4b5563"},
            },
            {
                "id": "wall_e", "type": "wall", "position": [15.0, 1.0, 0.0], "size": [1.0, 2.0, 30.0],
                "shape": "box", "material": {"pattern": "brick", "color": "#9ca3af", "color2": "#4b5563"},
            },
            {
                "id": "pillar_a", "type": "prop", "position": [7.0, 1.0, 5.0], "size": [1.2, 2.0, 1.2],
                "shape": "cylinder", "material": {"pattern": "stripes", "color": "#c084fc", "color2": "#4c1d95"},
            },
            {
                "id": "pillar_b", "type": "prop", "position": [-7.0, 1.0, -5.0], "size": [1.2, 2.0, 1.2],
                "shape": "cylinder", "material": {"pattern": "stripes", "color": "#c084fc", "color2": "#4c1d95"},
            },
        ],
        "items": [
            {
                "id": f"coin{i}", "type": "reward",
                "position": [float(x), 0.0, float(z)], "radius": 0.5,
                "reward": 1.0, "respawn": True, "cooldown_steps": 30, "shape": "crystal",
                "restrict_team": "runners",
                "material": {"pattern": "dots", "color": "#22c55e", "color2": "#14532d", "emissive": True},
            }
            for i, (x, z) in enumerate([(8, 8), (-8, 8), (8, -8), (-8, -8), (0, 10), (0, -10)])
        ],
        "agents": [
            {
                "id": "hunters", "count": 4, "team": "hunters", "role": "predator",
                "spawn": {"center": [0.0, 0.0, 0.0], "radius": 3.0}, "body_radius": 0.45,
                "movement": {"type": "discrete8", "speed": 0.6}, "sensors": {"type": "nearest_k", "k": 6, "range": 16.0},
                "shape": "cone", "material": {"pattern": "solid", "color": "#ef4444"},
            },
            {
                "id": "runners", "count": 2, "team": "runners", "role": "prey",
                "spawn": {"center": [0.0, 0.0, 0.0], "radius": 13.0}, "body_radius": 0.35,
                "movement": {"type": "discrete8", "speed": 0.75}, "sensors": {"type": "nearest_k", "k": 6, "range": 16.0},
                "shape": "capsule", "material": {"pattern": "solid", "color": "#3b82f6"},
            },
        ],
        "rules": {
            "tag": {
                "enabled": True, "predator_role": "predator", "prey_role": "prey",
                "predator_reward": 2.0, "prey_reward": -2.0,
                "prey_terminates": False, "prey_respawns": True,
            },
            "team_shared_reward": True,
        },
        "episode": {"max_steps": 500},
    }


def default_team_battle_large_spec(name: str = "Командная битва 3×3") -> dict[str, Any]:
    """Scaled-up `default_team_battle_spec`: 3 agents per side instead of
    2, on a bigger arena with more coins and a couple of central obstacles
    to fight over — the extra teammate per side is exactly what makes
    `qmix`'s mixing network combine something non-trivial (2 Q-values is
    the minimum useful case, 3 gives it noticeably more to work with)
    versus `default_team_battle_spec`'s minimal 2-vs-2."""
    return {
        "name": name,
        "description": "3×3 — команда «красные» против команды «синие» на большой арене, общая награда внутри команды.",
        "world": {"width": 30.0, "depth": 30.0, "wall_height": 2.0},
        "objects": [
            {
                "id": "wall_n", "type": "wall", "position": [0.0, 1.0, -15.0], "size": [30.0, 2.0, 1.0],
                "shape": "box", "material": {"pattern": "brick", "color": "#9ca3af", "color2": "#4b5563"},
            },
            {
                "id": "wall_s", "type": "wall", "position": [0.0, 1.0, 15.0], "size": [30.0, 2.0, 1.0],
                "shape": "box", "material": {"pattern": "brick", "color": "#9ca3af", "color2": "#4b5563"},
            },
            {
                "id": "wall_w", "type": "wall", "position": [-15.0, 1.0, 0.0], "size": [1.0, 2.0, 30.0],
                "shape": "box", "material": {"pattern": "brick", "color": "#9ca3af", "color2": "#4b5563"},
            },
            {
                "id": "wall_e", "type": "wall", "position": [15.0, 1.0, 0.0], "size": [1.0, 2.0, 30.0],
                "shape": "box", "material": {"pattern": "brick", "color": "#9ca3af", "color2": "#4b5563"},
            },
            {
                "id": "pillar_mid", "type": "prop", "position": [0.0, 1.0, 0.0], "size": [1.5, 2.0, 1.5],
                "shape": "cylinder", "material": {"pattern": "stripes", "color": "#c084fc", "color2": "#4c1d95"},
            },
        ],
        "items": [
            {
                "id": f"coin{i}", "type": "reward",
                "position": [float(x), 0.0, float(z)], "radius": 0.5,
                "reward": 1.0, "respawn": True, "cooldown_steps": 20, "shape": "crystal",
                "material": {"pattern": "dots", "color": "#eab308", "color2": "#713f12", "emissive": True},
            }
            for i, (x, z) in enumerate([
                (0, 0), (6, 0), (-6, 0), (0, 6), (0, -6), (6, 6), (-6, -6), (6, -6), (-6, 6), (0, 11), (0, -11),
            ])
        ],
        "agents": [
            {
                "id": "red", "count": 3, "team": "red", "role": "agent",
                "spawn": {"center": [-10.0, 0.0, 0.0], "radius": 3.5}, "body_radius": 0.4,
                "movement": {"type": "discrete8", "speed": 0.55}, "sensors": {"type": "nearest_k", "k": 6, "range": 16.0},
                "shape": "capsule", "material": {"pattern": "solid", "color": "#ef4444"},
            },
            {
                "id": "blue", "count": 3, "team": "blue", "role": "agent",
                "spawn": {"center": [10.0, 0.0, 0.0], "radius": 3.5}, "body_radius": 0.4,
                "movement": {"type": "discrete8", "speed": 0.55}, "sensors": {"type": "nearest_k", "k": 6, "range": 16.0},
                "shape": "capsule", "material": {"pattern": "solid", "color": "#3b82f6"},
            },
        ],
        "rules": {"team_shared_reward": True},
        "episode": {"max_steps": 500},
    }


def validate_slug(slug: str) -> None:
    if not _SLUG_RE.match(slug):
        raise ValueError(
            "Идентификатор сцены должен содержать только a-z, 0-9, '-' или '_' "
            "и начинаться с буквы/цифры",
        )
