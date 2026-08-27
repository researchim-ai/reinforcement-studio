"""Resolves `training.resume_from` (see `ExperimentConfig` on the frontend)
into the actual checkpoint file to load — shared by `runner_utils.py` (native
PPO/DQN/A2C/... and CustomAlgorithm plugins) and `sb3_runner.py` (custom
plugins that subclass an SB3 `BaseAlgorithm`).

Deliberately *not* a full "continue this exact run" feature: the resumed run
gets its own fresh `run_id`/`total_timesteps`/step counter (so its own
reward/loss chart always starts at step 0) rather than trying to splice two
runs' histories together — see `training.resume_from` in `config.json` for
how the UI instead shows "continued from X" and lets you overlay both curves
with the Compare Runs view if you want to see the whole picture. What *is*
carried over is the actual weights (+ optimizer state, where `save()`
includes it) via `AlgoClass.load(...)`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rl_core.paths import CHECKPOINTS_DIR, RUNS_DIR


class ResumeError(RuntimeError):
    """Raised for anything wrong with a `resume_from` request — caught by
    the runner's normal failure handling (error.log + status=failed) same as
    any other startup error, so the reason always reaches the UI."""


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def resolve_resume_source(resume_cfg: dict[str, Any], algo_id: str) -> Path:
    """Returns the checkpoint file to load for this resumed run, after
    checking the source actually used the *same* algorithm — loading PPO
    weights into a DQN's Q-network (or vice versa) would fail with a
    confusing low-level `state_dict` key-mismatch error instead of this
    clear one."""
    source = resume_cfg.get("source")
    source_id = resume_cfg.get("id")
    if not source or not source_id:
        raise ResumeError("resume_from требует поля source и id")

    if source == "run":
        rdir = RUNS_DIR / str(source_id)
        src_config = _read_json(rdir / "config.json")
        if src_config is None:
            raise ResumeError(f"Исходный запуск для дообучения не найден: {source_id}")
        src_algo_id = src_config.get("algorithm", {}).get("id")
        if src_algo_id != algo_id:
            raise ResumeError(
                f"Несовпадение алгоритма: исходный запуск обучен через «{src_algo_id}», "
                f"а этот запускается как «{algo_id}» — архитектуры не совпадут."
            )
        model_path = rdir / "model.zip"
    elif source == "checkpoint":
        safe_name = str(source_id)
        meta = _read_json(CHECKPOINTS_DIR / f"{safe_name}.meta.json")
        if meta is not None:
            src_algo_id = meta.get("algorithm_id")
            if src_algo_id and src_algo_id != algo_id:
                raise ResumeError(
                    f"Несовпадение алгоритма: чекпоинт «{safe_name}» сохранён для «{src_algo_id}», "
                    f"а этот запускается как «{algo_id}» — архитектуры не совпадут."
                )
        model_path = CHECKPOINTS_DIR / f"{safe_name}.zip"
    else:
        raise ResumeError(f"Неизвестный источник resume_from: {source}")

    if not model_path.exists():
        raise ResumeError(f"Файл модели для дообучения не найден: {model_path}")
    return model_path
