"""Environment/wrapper/algorithm catalogs consumed by the Experiment Designer."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from rl_core.envs.previews import preview_media_type, resolve_preview_file
from rl_core.envs.registry import list_environments
from rl_core.envs.wrappers import WRAPPER_CATALOG

router = APIRouter()

ALGORITHM_CATALOG = [
    {
        "id": "dqn",
        "name": "DQN",
        "kind": "gym",
        "description": "Deep Q-Network — value-based, для дискретных действий.",
        "hyperparams": [
            {"key": "learning_rate", "label": "Learning rate", "type": "float", "default": 1e-3, "min": 1e-6, "max": 1e-1},
            {"key": "buffer_size", "label": "Replay buffer size", "type": "int", "default": 50_000, "min": 1_000, "max": 1_000_000},
            {"key": "batch_size", "label": "Batch size", "type": "int", "default": 64, "min": 8, "max": 512},
            {"key": "gamma", "label": "Discount (gamma)", "type": "float", "default": 0.99, "min": 0.5, "max": 0.999},
            {"key": "exploration_fraction", "label": "Exploration fraction", "type": "float", "default": 0.2, "min": 0.0, "max": 1.0},
        ],
    },
    {
        "id": "ppo",
        "name": "PPO",
        "kind": "gym",
        "description": "Proximal Policy Optimization — стабильный policy-gradient метод, discrete и continuous.",
        "hyperparams": [
            {"key": "learning_rate", "label": "Learning rate", "type": "float", "default": 3e-4, "min": 1e-6, "max": 1e-1},
            {"key": "n_steps", "label": "Steps per update", "type": "int", "default": 2048, "min": 32, "max": 8192},
            {"key": "batch_size", "label": "Batch size", "type": "int", "default": 64, "min": 8, "max": 1024},
            {"key": "gamma", "label": "Discount (gamma)", "type": "float", "default": 0.99, "min": 0.5, "max": 0.999},
            {"key": "ent_coef", "label": "Entropy coefficient", "type": "float", "default": 0.0, "min": 0.0, "max": 0.1},
        ],
    },
    {
        "id": "a2c",
        "name": "A2C",
        "kind": "gym",
        "description": "Advantage Actor-Critic — простой и быстрый on-policy метод.",
        "hyperparams": [
            {"key": "learning_rate", "label": "Learning rate", "type": "float", "default": 7e-4, "min": 1e-6, "max": 1e-1},
            {"key": "n_steps", "label": "Steps per update", "type": "int", "default": 5, "min": 1, "max": 256},
            {"key": "gamma", "label": "Discount (gamma)", "type": "float", "default": 0.99, "min": 0.5, "max": 0.999},
            {"key": "ent_coef", "label": "Entropy coefficient", "type": "float", "default": 0.01, "min": 0.0, "max": 0.1},
        ],
    },
    {
        "id": "alphazero",
        "name": "AlphaZero",
        "kind": "alphazero",
        "description": "Self-play + MCTS + dual-head сеть — для настольных игр.",
        "hyperparams": [
            {"key": "num_simulations", "label": "MCTS simulations/move", "type": "int", "default": 25, "min": 4, "max": 400},
            {"key": "games_per_iteration", "label": "Self-play games/iteration", "type": "int", "default": 20, "min": 2, "max": 200},
            {"key": "epochs", "label": "Training epochs/iteration", "type": "int", "default": 4, "min": 1, "max": 20},
            {"key": "batch_size", "label": "Batch size", "type": "int", "default": 64, "min": 8, "max": 512},
            {"key": "learning_rate", "label": "Learning rate", "type": "float", "default": 1e-3, "min": 1e-6, "max": 1e-1},
            {"key": "eval_games", "label": "Arena games/iteration", "type": "int", "default": 10, "min": 2, "max": 100},
            {"key": "win_rate_threshold", "label": "Accept win-rate", "type": "float", "default": 0.55, "min": 0.5, "max": 0.9},
            {"key": "channels", "label": "Conv channels", "type": "int", "default": 48, "min": 8, "max": 256},
            {"key": "num_blocks", "label": "Residual blocks", "type": "int", "default": 3, "min": 1, "max": 10},
        ],
    },
]


@router.get("/list")
async def list_envs():
    return {"environments": list_environments()}


@router.get("/preview")
async def env_preview(id: str = Query(..., min_length=1), thumb: bool = False):
    path = resolve_preview_file(id, thumb=thumb)
    if path is None:
        raise HTTPException(status_code=404, detail=f"No preview for {id}")
    return FileResponse(
        path,
        media_type=preview_media_type(path),
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/wrappers")
async def list_wrappers():
    return {"wrappers": WRAPPER_CATALOG}


@router.get("/algorithms")
async def list_algorithms():
    return {"algorithms": ALGORITHM_CATALOG}
