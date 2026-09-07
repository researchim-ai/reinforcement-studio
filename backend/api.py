"""FastAPI backend — thin API layer between Electron/React and rl_core."""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import asyncio

from backend import __version__, sweep_manager
from backend.routes import alphazero, environments, models, networks, plugins, scenes, sweeps, system, training, world_models
from backend.ws import router as ws_router

app = FastAPI(title="Reinforcement Studio API", version=__version__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router, prefix="/api/system", tags=["system"])
app.include_router(environments.router, prefix="/api/environments", tags=["environments"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(alphazero.router, prefix="/api/alphazero", tags=["alphazero"])
app.include_router(plugins.router, prefix="/api/plugins", tags=["plugins"])
app.include_router(networks.router, prefix="/api/networks", tags=["networks"])
app.include_router(world_models.router, prefix="/api/world-models", tags=["world-models"])
app.include_router(scenes.router, prefix="/api/scenes", tags=["scenes"])
app.include_router(sweeps.router, prefix="/api/sweeps", tags=["sweeps"])
app.include_router(ws_router)


@app.on_event("startup")
async def _start_sweep_scheduler() -> None:
    # Picks up the next queued run of any sweep (see
    # `backend/sweep_manager.py`) whenever the previous member of that same
    # sweep finishes — runs for the lifetime of the app, restarted for free
    # on every backend restart since sweep state lives entirely in
    # `RUNS_DIR`, not in memory.
    asyncio.create_task(sweep_manager.scheduler_loop())
