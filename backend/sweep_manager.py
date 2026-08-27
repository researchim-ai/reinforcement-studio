"""Hyperparameter sweeps: run the same experiment across a grid of
hyperparameter values × multiple seeds, one at a time (so a sweep never
fights itself for CPU/GPU), without a heavyweight job queue — just plain
run directories the rest of the app already understands, plus one small
per-run `queued.flag` marker and a background scheduler tick that starts the
next queued run once the previous one (from the *same* sweep) finishes.

Every sweep member is a completely normal run (same `config.json`, same
`rl_core.runner` subprocess, shows up in Training Monitor/Model Zoo like
any other) — the only difference is `config["sweep"]` and, until it's
picked up, a `queued.flag` file sitting next to `config.json` instead of the
run being started immediately. This means sweeps survive a backend restart
for free: the scheduler just re-derives everything by scanning `RUNS_DIR`.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend import process_manager
from rl_core.paths import RUNS_DIR, run_dir

logger = logging.getLogger(__name__)

_TICK_SECONDS = 3.0


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _combo_label(combo: dict[str, float]) -> str:
    if not combo:
        return "baseline"
    return ", ".join(f"{k}={v}" for k, v in combo.items())


def create_sweep(
    base: dict[str, Any], grid: dict[str, list[float]], seeds: list[int], name: str | None,
) -> dict[str, Any]:
    """Writes one queued run per (grid combination × seed) and returns
    the new sweep's id + every run_id it created, in order."""
    sweep_id = uuid.uuid4().hex[:10]
    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys if grid[k]]
    keys = [k for k in keys if grid[k]]
    combos: list[dict[str, float]] = (
        [dict(zip(keys, values)) for values in itertools.product(*value_lists)] if keys else [{}]
    )
    seeds = seeds or [int(base.get("training", {}).get("seed", 42))]

    env_id = base.get("environment", {}).get("id", "run")
    base_algo = base.get("algorithm", {})
    base_training = base.get("training", {})
    total = len(combos) * len(seeds)
    run_ids: list[str] = []
    created_at = datetime.now(timezone.utc).isoformat()

    index = 0
    for combo in combos:
        for seed in seeds:
            run_id = f"{env_id.lower().replace(' ', '-')}-sweep-{sweep_id}-{index}"
            rdir = run_dir(run_id)
            config = {
                "run_id": run_id,
                "kind": base.get("kind", "gym"),
                "name": f"{name or 'Sweep'} #{index + 1}/{total} ({_combo_label(combo)}, seed={seed})",
                "environment": base.get("environment", {}),
                "algorithm": {
                    **base_algo,
                    "hyperparams": {**(base_algo.get("hyperparams") or {}), **combo},
                },
                "training": {**base_training, "seed": seed},
                "created_at": created_at,
                "sweep": {
                    "sweep_id": sweep_id,
                    "name": name,
                    "index": index,
                    "total": total,
                    "params": combo,
                    "seed": seed,
                },
            }
            (rdir / "config.json").write_text(json.dumps(config, indent=2))
            (rdir / "queued.flag").write_text("1")
            run_ids.append(run_id)
            index += 1

    return {"sweep_id": sweep_id, "run_ids": run_ids, "total": total}


def list_sweeps() -> list[dict[str, Any]]:
    """One entry per distinct `sweep_id` found among all run configs,
    newest first — cheap enough to just scan on every request (same
    pattern `GET /training/runs` already uses)."""
    sweeps: dict[str, dict[str, Any]] = {}
    if not RUNS_DIR.exists():
        return []
    for d in RUNS_DIR.iterdir():
        if not d.is_dir():
            continue
        config = _read_json(d / "config.json")
        if not config:
            continue
        sweep = config.get("sweep")
        if not sweep:
            continue
        sid = sweep["sweep_id"]
        entry = sweeps.setdefault(sid, {
            "sweep_id": sid,
            "name": sweep.get("name"),
            "total": sweep.get("total", 0),
            "run_ids": [],
            "created_at": config.get("created_at"),
            "environment_id": config.get("environment", {}).get("id"),
            "algorithm_id": config.get("algorithm", {}).get("id"),
        })
        entry["run_ids"].append(d.name)
    return sorted(sweeps.values(), key=lambda s: s.get("created_at") or "", reverse=True)


def sweep_run_ids(sweep_id: str) -> list[str]:
    if not RUNS_DIR.exists():
        return []
    out: list[tuple[int, str]] = []
    for d in RUNS_DIR.iterdir():
        if not d.is_dir():
            continue
        config = _read_json(d / "config.json")
        sweep = (config or {}).get("sweep")
        if sweep and sweep.get("sweep_id") == sweep_id:
            out.append((sweep.get("index", 0), d.name))
    out.sort(key=lambda x: x[0])
    return [run_id for _, run_id in out]


def _tick() -> None:
    if not RUNS_DIR.exists():
        return
    queued_by_sweep: dict[str, list[tuple[int, str, Path]]] = {}
    active_sweeps: set[str] = set()

    for d in RUNS_DIR.iterdir():
        if not d.is_dir():
            continue
        config_path = d / "config.json"
        config = _read_json(config_path)
        if not config:
            continue
        sweep = config.get("sweep")
        if not sweep:
            continue
        sid = sweep["sweep_id"]
        if process_manager.is_running(d.name):
            active_sweeps.add(sid)
            continue
        if (d / "queued.flag").exists():
            queued_by_sweep.setdefault(sid, []).append((sweep.get("index", 0), d.name, config_path))

    for sid, items in queued_by_sweep.items():
        if sid in active_sweeps:
            continue
        items.sort(key=lambda x: x[0])
        _, run_id, config_path = items[0]
        try:
            (RUNS_DIR / run_id / "queued.flag").unlink(missing_ok=True)
            process_manager.start_run(run_id, config_path)
        except Exception:
            logger.exception("Failed to start queued sweep run %s", run_id)


async def scheduler_loop() -> None:
    """Started once from the FastAPI app's startup event — never returns;
    cancelled automatically when the app shuts down."""
    while True:
        try:
            _tick()
        except Exception:
            logger.exception("Sweep scheduler tick failed")
        await asyncio.sleep(_TICK_SECONDS)


def delete_sweep(sweep_id: str) -> int:
    """Stops (if running) and deletes every run belonging to this sweep.
    Returns how many run directories were removed."""
    import shutil

    count = 0
    for run_id in sweep_run_ids(sweep_id):
        rdir = RUNS_DIR / run_id
        if process_manager.is_running(run_id):
            process_manager.stop_run(run_id)
        if rdir.exists():
            shutil.rmtree(rdir, ignore_errors=True)
            count += 1
    return count
