"""CRUD + live shape-preview for user-designed network architectures — the
visual Network Architecture Builder page (`/network-builder`). Unlike
`backend/routes/plugins.py` these are pure JSON specs (no Python execution),
compiled into a real `torch.nn.Module` only via `rl_core/netbuilder.py`,
either here (read-only, for the preview) or at actual train time."""
from __future__ import annotations

import re
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from rl_core import netbuilder_store as store
from rl_core.composite_netbuilder import COMPOSITE_FAMILIES, validate_composite_spec
from rl_core.netbuilder import FAMILY_HEADS, preview_network

router = APIRouter()

NetworkFamily = Literal[
    "actor_critic", "q_network", "dueling_q", "alphazero",
    "efficientzero", "unizero", "researchimzero", "latentimzero",
]
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class NetworkDoc(BaseModel):
    name: str
    description: str = ""
    family: NetworkFamily = "actor_critic"
    spec: dict[str, Any]


class PreviewRequest(BaseModel):
    family: NetworkFamily
    spec: dict[str, Any]
    environment_id: str | None = None
    wrappers: list[dict[str, Any]] = []
    game_id: str | None = None
    hyperparams: dict[str, Any] = {}


def _check_slug(slug: str) -> None:
    if not _SLUG_RE.match(slug):
        raise HTTPException(
            status_code=400,
            detail="Идентификатор архитектуры должен содержать только a-z, 0-9, '-' или '_' и начинаться с буквы/цифры",
        )


@router.get("/families")
async def list_families():
    flat = [{"id": name, "required_heads": list(heads), "format": "trunk_heads_v1"} for name, heads in FAMILY_HEADS.items()]
    composite = [{"id": name, "required_heads": [], "format": "composite_v1"} for name in COMPOSITE_FAMILIES]
    return {"families": flat + composite}


@router.get("")
async def list_networks():
    return {"networks": store.list_meta()}


@router.get("/{slug}")
async def get_network(slug: str):
    try:
        return {"slug": slug, **store.load(slug)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/{slug}")
async def save_network(slug: str, body: NetworkDoc):
    _check_slug(slug)
    doc = body.model_dump()
    try:
        if body.family in COMPOSITE_FAMILIES:
            doc["spec"] = validate_composite_spec(body.spec, body.family)
        elif body.spec.get("format") == "composite_v1":
            raise HTTPException(status_code=400, detail="Composite-spec несовместим с flat network family")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - validation error is user input
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    store.save(slug, doc)
    return {"success": True}


@router.delete("/{slug}")
async def delete_network(slug: str):
    store.delete(slug)
    return {"success": True}


def _shape_for_gym(environment_id: str, wrappers: list[dict[str, Any]], family: NetworkFamily) -> tuple[list[int], dict[str, int]]:
    import gymnasium as gym

    from rl_core.algorithms.native.preprocessing import is_image_space, obs_flat_dim
    from rl_core.algorithms.sb3_runner import _make_env

    env = _make_env(environment_id, wrappers)
    try:
        obs_space, action_space = env.observation_space, env.action_space
        input_shape = list(obs_space.shape) if is_image_space(obs_space) else [obs_flat_dim(obs_space)]
        discrete = isinstance(action_space, gym.spaces.Discrete)
        if family == "q_network":
            if not discrete:
                raise HTTPException(status_code=400, detail="Q-Network поддерживает только дискретные действия")
            return input_shape, {"q": int(action_space.n)}
        if family == "dueling_q":
            if not discrete:
                raise HTTPException(status_code=400, detail="Dueling Q-Network поддерживает только дискретные действия")
            return input_shape, {"advantage": int(action_space.n), "value": 1}
        action_dim = int(action_space.n) if discrete else int(__import__("numpy").prod(action_space.shape))
        return input_shape, {"action": action_dim, "value": 1}
    finally:
        env.close()


def _shape_for_alphazero(game_id: str) -> tuple[list[int], dict[str, int]]:
    from rl_core.games import make_game

    game = make_game(game_id)
    return [3, game.rows, game.cols], {"policy": int(game.action_size), "value": 1}


@router.post("/preview")
async def preview(req: PreviewRequest):
    """Compiles the spec against a *real* environment/game's observation and
    action space (same code the actual run would use), so the shapes and
    parameter count the Builder shows are exactly what you'd get — errors
    (missing Flatten, oversized Conv2d kernel, ...) come back as a plain
    message rather than a traceback."""
    try:
        if req.family in COMPOSITE_FAMILIES:
            if not req.environment_id:
                raise HTTPException(status_code=400, detail="environment_id обязателен для composite-сети")
            spec = validate_composite_spec(req.spec, req.family)
            from rl_core.inspect import inspect_gym

            inspected = inspect_gym(
                req.environment_id,
                req.wrappers,
                req.family,
                {**req.hyperparams, "network_spec": spec},
            )
            network = inspected.get("network")
            if network is None:
                return {
                    "ok": False,
                    "error": inspected.get("error") or "Не удалось собрать composite-архитектуру",
                    "components": [],
                    "total_params": None,
                    "trainable_params": None,
                }
            return {
                "ok": True,
                "error": None,
                "input_shape": network.get("input_shape"),
                "output_shape": network.get("output_shape"),
                "components": network.get("architecture_components", []),
                "total_params": network.get("total_params"),
                "trainable_params": network.get("trainable_params"),
                "fixed_outputs": _composite_fixed_outputs(req.environment_id, req.wrappers, req.family, req.hyperparams),
            }
        if req.family == "alphazero":
            if not req.game_id:
                raise HTTPException(status_code=400, detail="game_id обязателен для family=alphazero")
            input_shape, head_out_features = _shape_for_alphazero(req.game_id)
        else:
            if not req.environment_id:
                raise HTTPException(status_code=400, detail="environment_id обязателен для этого типа сети")
            input_shape, head_out_features = _shape_for_gym(req.environment_id, req.wrappers, req.family)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not a crash
        return {
            "ok": False,
            "error": f"Не удалось создать среду: {exc}",
            "trunk": [],
            "trunk_error": None,
            "trunk_error_index": None,
            "heads": {},
            "total_params": None,
        }

    return preview_network(req.spec, input_shape, req.family, head_out_features=head_out_features)


def _composite_fixed_outputs(
    environment_id: str, wrappers: list[dict[str, Any]], family: str, hyperparams: dict[str, Any],
) -> dict[str, list[int | str]]:
    import gymnasium as gym
    import numpy as np

    from rl_core.algorithms.sb3_runner import _make_env

    env = _make_env(environment_id, wrappers)
    try:
        action_space = env.action_space
        action_dim = int(action_space.n) if isinstance(action_space, gym.spaces.Discrete) else int(np.prod(action_space.shape))
        default_support = 50 if family == "unizero" else 300
        support = max(1, int(hyperparams.get("value_support_size", default_support)))
        outputs = {"policy": [action_dim], "value": [2 * support + 1]}
        if family == "efficientzero":
            outputs.update({"value_prefix": [2 * support + 1], "next_latent": ["latent_dim"]})
        elif family == "latentimzero":
            outputs.update({
                "reward": [2 * support + 1],
                "next_token": ["embed_dim"],
                "uncertainty_rewards": ["uncertainty_members"],
            })
        else:
            outputs.update({"reward": [2 * support + 1], "next_token": ["embed_dim"]})
        return outputs
    finally:
        env.close()
