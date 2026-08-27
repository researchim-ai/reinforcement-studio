"""Shared helpers for persisting a run's full metrics time series (not just
the latest snapshot) and for keeping any JSON payload free of NaN/Infinity.

Two problems this fixes:

1. `metrics.json` is a single file that every snapshot *overwrites* — it's
   great for "what's the current state of this run" (used for the run list
   and as a fallback before the websocket connects), but it means the whole
   reward/loss curve only ever lived in the frontend's in-memory `history`
   array, accumulated from live websocket messages. Navigate away and come
   back, reopen the app, or just look at a run that already finished by the
   time you check on it, and there's nothing left to rebuild the chart from
   — it looked like "the reward chart stopped working" when actually the
   data was simply never kept anywhere. `append_history`/`read_history` add
   a small companion `metrics_history.jsonl` next to `metrics.json` so the
   full curve survives all of that.

2. `json.dumps` happily writes literal `NaN`/`Infinity` (Python's `json`
   module allows it by default), but that's not valid JSON per spec —
   JavaScript's `JSON.parse` throws on it, and the frontend's websocket
   client wraps `JSON.parse` in a bare try/catch that silently drops the
   *entire* message on any parse error. A single NaN anywhere in a snapshot
   (e.g. a running-stats computation that briefly divides by ~0 early in
   training) doesn't just blank one field — it makes every future websocket
   message for that run vanish silently, freezing every chart. `json_safe`
   scrubs those before anything gets written or sent.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

_HISTORY_FILENAME = "metrics_history.jsonl"
# Once the history file passes this size, it's thinned back down by half
# (dropping every other entry) — bounds both disk usage and the size of the
# one-shot REST response for very long runs, at the cost of halving temporal
# resolution for old parts of the curve (never the most recent data).
_MAX_HISTORY_BYTES = 2_000_000
# Heavy/transient fields that belong in the "latest" metrics.json (for the
# live preview) but would massively bloat the history file for no benefit —
# nobody needs to see the base64 GIF/board from 50 snapshots ago.
_HISTORY_EXCLUDE_KEYS = {"episode_gif_base64", "board_history"}


def json_safe(value: Any) -> Any:
    """Recursively replaces non-finite floats (NaN/+-Infinity) with `None`
    — see module docstring for why this matters."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def _read_last_line(path: Path) -> str | None:
    """Cheaply grabs the last non-empty line without reading the whole file
    — history files can grow to a couple MB over a long run, and this runs
    on every single snapshot write. Falls back to a full read if the tail
    chunk turns out not to contain a full line (line longer than the tail
    window, e.g. an unusually large `hyperparams` blob)."""
    tail_bytes = 8192
    with path.open("rb") as f:
        size = f.seek(0, 2)
        f.seek(max(0, size - tail_bytes))
        chunk = f.read().decode("utf-8", errors="ignore")
    lines = [line for line in chunk.splitlines() if line.strip()]
    if not lines:
        return None
    # If the tail window was cut off mid-line and there's nothing before it
    # to fall back on, we can't tell — but a single line filling 8KB+ is
    # pathological enough to just accept the (rare) risk of a duplicate.
    return lines[-1]


def append_history(run_dir: Path, snapshot: dict[str, Any]) -> None:
    """Appends a slim, JSON-safe copy of `snapshot` to this run's history
    log — or, if the *previous* entry has the exact same `step` (every
    caller writes one snapshot mid-training via its periodic timer, then
    one more right after `learn()`/the training loop returns to capture the
    final status; those two are frequently the same step), overwrites it in
    place instead of appending a duplicate. Charts key their x-axis off
    `step`, so two consecutive rows with an identical step value are at best
    a wasted point and at worst make a category-axis line double back on
    itself. Best-effort throughout: a failure here (disk full, permissions,
    ...) must never take down training itself."""
    try:
        path = run_dir / _HISTORY_FILENAME
        slim = {k: v for k, v in snapshot.items() if k not in _HISTORY_EXCLUDE_KEYS}
        line = json.dumps(json_safe(slim))

        last_line = _read_last_line(path) if path.exists() else None
        replace_last = False
        if last_line is not None:
            try:
                if json.loads(last_line).get("step") == slim.get("step"):
                    replace_last = True
            except json.JSONDecodeError:
                pass

        if replace_last:
            lines = path.read_text().splitlines()
            lines[-1] = line
            path.write_text("\n".join(lines) + "\n")
        else:
            with path.open("a") as f:
                f.write(line + "\n")
        _maybe_thin(path)
    except OSError:
        pass


def _maybe_thin(path: Path) -> None:
    try:
        if path.stat().st_size <= _MAX_HISTORY_BYTES:
            return
        lines = path.read_text().splitlines()
        if len(lines) <= 200:
            return
        thinned = lines[::2]
        if thinned[-1] != lines[-1]:
            thinned.append(lines[-1])
        path.write_text("\n".join(thinned) + "\n")
    except OSError:
        pass


def read_history(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / _HISTORY_FILENAME
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        text = path.read_text()
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
