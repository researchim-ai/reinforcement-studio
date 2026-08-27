"""WebSocket endpoint for real-time training metrics streaming."""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from rl_core.metrics_history import json_safe
from rl_core.paths import RUNS_DIR

router = APIRouter()


@router.websocket("/ws/metrics/{run_id}")
async def metrics_ws(websocket: WebSocket, run_id: str):
    """Streams metrics.json from a run dir every time it changes (both the
    Gymnasium/SB3 callback and the AlphaZero trainer write the same shape:
    one JSON object per step/iteration snapshot)."""
    await websocket.accept()

    metrics_path = RUNS_DIR / run_id / "metrics.json"
    last_mtime = 0.0
    last_step = -1

    try:
        while True:
            if metrics_path.exists():
                mtime = metrics_path.stat().st_mtime
                if mtime > last_mtime:
                    last_mtime = mtime
                    try:
                        data = json.loads(metrics_path.read_text())
                        step = data.get("step", 0)
                        if step != last_step or data.get("status") != "running":
                            last_step = step
                            # `metrics.json` may have been written before this
                            # sanitization existed (or by an older run) — scrub
                            # NaN/Infinity here too, not just on write, so a
                            # single bad value can never make `JSON.parse`
                            # silently drop the whole message client-side and
                            # freeze every chart for the rest of the run.
                            await websocket.send_json(json_safe(data))
                    except (json.JSONDecodeError, IOError):
                        pass
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    except Exception:
        await websocket.close()
