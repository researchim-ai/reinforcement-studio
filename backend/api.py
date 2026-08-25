"""FastAPI backend — thin API layer between Electron/React and rl_core."""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.routes import alphazero, environments, models, networks, plugins, system, training
from backend.ws import router as ws_router

app = FastAPI(title="Reinforcement Studio API", version="0.1.0")

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
app.include_router(ws_router)
