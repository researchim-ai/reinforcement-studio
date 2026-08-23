"""Tracks the subprocess for each training run so /training/stop can act
immediately instead of only relying on the cooperative stop.flag file."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from rl_core.paths import PROJECT_ROOT, run_dir

_PROCESSES: dict[str, subprocess.Popen] = {}


def start_run(run_id: str, config_path: Path) -> subprocess.Popen:
    rdir = run_dir(run_id)
    stdout_log = open(rdir / "stdout.log", "a")
    stderr_log = open(rdir / "stderr.log", "a")
    proc = subprocess.Popen(
        [sys.executable, "-m", "rl_core.runner", str(config_path), str(rdir)],
        cwd=str(PROJECT_ROOT),
        stdout=stdout_log,
        stderr=stderr_log,
    )
    _PROCESSES[run_id] = proc
    return proc


def is_running(run_id: str) -> bool:
    proc = _PROCESSES.get(run_id)
    if proc is None:
        return False
    return proc.poll() is None


def stop_run(run_id: str, timeout: float = 5.0) -> bool:
    rdir = run_dir(run_id)
    (rdir / "stop.flag").write_text("1")

    proc = _PROCESSES.get(run_id)
    if proc is None:
        return True

    deadline = time.time() + timeout
    while proc.poll() is None and time.time() < deadline:
        time.sleep(0.2)

    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()

    return True


def active_run_ids() -> list[str]:
    return [run_id for run_id, proc in _PROCESSES.items() if proc.poll() is None]
