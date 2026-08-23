"""Interactive AlphaZero Arena: play a board game against a trained agent."""
from __future__ import annotations

import json
import random
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from rl_core.alphazero.arena import suggest_move
from rl_core.alphazero.checkpoint import load_checkpoint
from rl_core.games import GAME_REGISTRY, make_game
from rl_core.paths import CHECKPOINTS_DIR, RUNS_DIR

router = APIRouter()

_SESSIONS: dict[str, dict[str, Any]] = {}
_NETWORK_CACHE: dict[str, Any] = {}


def _read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _load_network(source: str, opponent_id: str):
    if opponent_id == "random":
        return None
    cache_key = f"{source}:{opponent_id}"
    if cache_key in _NETWORK_CACHE:
        return _NETWORK_CACHE[cache_key]

    if source == "run":
        path = RUNS_DIR / opponent_id / "model.pt"
    else:
        path = CHECKPOINTS_DIR / f"{opponent_id}.pt"

    if not path.exists():
        raise HTTPException(status_code=404, detail="Model checkpoint not found")

    net, _meta = load_checkpoint(path)
    _NETWORK_CACHE[cache_key] = net
    return net


@router.get("/opponents")
async def list_opponents(game_id: str):
    if game_id not in GAME_REGISTRY:
        raise HTTPException(status_code=400, detail="Unknown game")

    opponents = [{"id": "random", "source": "builtin", "label": "Случайный соперник"}]

    if RUNS_DIR.exists():
        for d in RUNS_DIR.iterdir():
            config = _read_json(d / "config.json")
            if not config or config.get("kind") != "alphazero":
                continue
            if config.get("environment", {}).get("id") != game_id:
                continue
            if not (d / "model.pt").exists():
                continue
            opponents.append({"id": d.name, "source": "run", "label": config.get("name") or d.name})

    if CHECKPOINTS_DIR.exists():
        for meta_path in CHECKPOINTS_DIR.glob("*.meta.json"):
            meta = _read_json(meta_path, {}) or {}
            if meta.get("kind") != "alphazero" or meta.get("environment_id") != game_id:
                continue
            name = meta_path.stem.replace(".meta", "")
            if (CHECKPOINTS_DIR / f"{name}.pt").exists():
                opponents.append({"id": name, "source": "checkpoint", "label": meta.get("name", name)})

    return {"opponents": opponents}


class NewSessionRequest(BaseModel):
    game_id: str
    opponent_id: str = "random"
    opponent_source: str = "builtin"
    human_first: bool = True
    num_simulations: int = 80


def _serialize_state(session: dict[str, Any], ai_info: dict | None = None) -> dict[str, Any]:
    game = session["game"]
    return {
        "session_id": session["id"],
        "board": game.board_list(),
        "rows": game.rows,
        "cols": game.cols,
        "current_player": int(game.current_player),
        "human_player": session["human_player"],
        "done": game.done,
        "winner": int(game.winner) if game.winner is not None else None,
        "legal_actions": game.legal_actions(),
        "ai_info": ai_info,
    }


def _ai_move(session: dict[str, Any]) -> dict[str, Any] | None:
    game = session["game"]
    net = session["net"]
    if game.done:
        return None
    if net is None:
        legal = game.legal_actions()
        action = random.choice(legal)
        game.step(action)
        return {"action": action, "visit_probs": {}}

    result = suggest_move(game, net, num_simulations=session["num_simulations"])
    if result["action"] is not None:
        game.step(result["action"])
    return result


@router.post("/session")
async def new_session(req: NewSessionRequest):
    game = make_game(req.game_id)
    game.reset()
    net = _load_network(req.opponent_source, req.opponent_id)

    session_id = uuid.uuid4().hex[:12]
    session = {
        "id": session_id,
        "game": game,
        "net": net,
        "human_player": 1 if req.human_first else -1,
        "num_simulations": req.num_simulations,
    }
    _SESSIONS[session_id] = session

    ai_info = None
    if not req.human_first:
        ai_info = _ai_move(session)

    return _serialize_state(session, ai_info)


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    session = _SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return _serialize_state(session)


class MoveRequest(BaseModel):
    action: int


@router.post("/session/{session_id}/move")
async def make_move(session_id: str, req: MoveRequest):
    session = _SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    game = session["game"]

    if game.done:
        raise HTTPException(status_code=400, detail="Game already finished")
    if game.current_player != session["human_player"]:
        raise HTTPException(status_code=400, detail="Not your turn")
    if req.action not in game.legal_actions():
        raise HTTPException(status_code=400, detail="Illegal move")

    game.step(req.action)
    ai_info = _ai_move(session)
    return _serialize_state(session, ai_info)


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    _SESSIONS.pop(session_id, None)
    return {"success": True}


@router.get("/games")
async def list_games():
    return {"games": [{"id": g_id, "name": cls().name, "rows": cls.rows, "cols": cls.cols} for g_id, cls in GAME_REGISTRY.items()]}
