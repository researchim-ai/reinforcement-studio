"""Dynamic loading + interface validation for user-authored plugins (custom
Gym algorithms, custom AlphaZero trainers, custom reward functions).

Trust model: this is a local desktop app — plugin code runs with the same
level of trust as the app itself (no sandboxing/process isolation), which is
equivalent to hand-editing rl_core/algorithms/sb3_runner.py. The only
guardrail is `validate_*()`, a cheap dry-run smoke test so a broken plugin
fails fast with a readable traceback in the UI, rather than after a long
training run.
"""
from __future__ import annotations

import importlib.util
import inspect
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

from rl_core.paths import CUSTOM_ALPHAZERO_ALGOS_DIR, CUSTOM_GYM_ALGOS_DIR, CUSTOM_REWARDS_DIR


class PluginError(RuntimeError):
    """Raised when a plugin file doesn't satisfy its contract (missing
    attribute, wrong base class, ...). Caught by validate_*() and reported
    to the UI; if it escapes during an actual run, the runner's normal
    failure handling (error.log + status=failed) takes over."""


def _module_path(directory: Path, slug: str) -> Path:
    return directory / f"{slug}.py"


def _load_module_from_path(path: Path, module_name: str | None = None):
    if not path.exists():
        raise PluginError(f"Файл плагина не найден: {path}")
    name = module_name or f"rl_plugin_{path.stem}_{uuid.uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PluginError(f"Не удалось загрузить модуль плагина: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hyperparams_of(module: Any) -> list[dict[str, Any]]:
    hp = getattr(module, "HYPERPARAMS", [])
    if not isinstance(hp, list):
        raise PluginError("HYPERPARAMS должен быть списком спецификаций гиперпараметров")
    return hp


# --------------------------------------------------------------- Gym algorithms

def load_gym_algorithm_from_path(path: Path) -> tuple[type, list[dict[str, Any]]]:
    from stable_baselines3.common.base_class import BaseAlgorithm

    from rl_core.algorithms.base import CustomAlgorithm

    module = _load_module_from_path(path)
    cls = getattr(module, "ALGORITHM_CLASS", None)
    if cls is None or not inspect.isclass(cls):
        raise PluginError("Скрипт алгоритма должен определять ALGORITHM_CLASS (класс)")
    if not (issubclass(cls, BaseAlgorithm) or issubclass(cls, CustomAlgorithm)):
        raise PluginError(
            "ALGORITHM_CLASS должен наследоваться от stable_baselines3 BaseAlgorithm "
            "или от rl_core.algorithms.base.CustomAlgorithm"
        )
    return cls, _hyperparams_of(module)


def load_gym_algorithm(slug: str) -> tuple[type, list[dict[str, Any]]]:
    return load_gym_algorithm_from_path(_module_path(CUSTOM_GYM_ALGOS_DIR, slug))


def gym_algorithm_meta(slug: str) -> dict[str, Any]:
    path = _module_path(CUSTOM_GYM_ALGOS_DIR, slug)
    module = _load_module_from_path(path)
    return {
        "id": f"custom:{slug}",
        "slug": slug,
        "name": getattr(module, "NAME", slug),
        "kind": "gym",
        "description": getattr(module, "DESCRIPTION", ""),
        "hyperparams": _hyperparams_of(module),
        "is_custom": True,
        "supported_action_kinds": list(getattr(module, "SUPPORTED_ACTION_KINDS", ["discrete", "continuous"])),
    }


def list_gym_algorithm_slugs() -> list[str]:
    if not CUSTOM_GYM_ALGOS_DIR.exists():
        return []
    return sorted(p.stem for p in CUSTOM_GYM_ALGOS_DIR.glob("*.py"))


def validate_gym_algorithm_at(path: Path) -> dict[str, Any]:
    try:
        import gymnasium as gym
        from stable_baselines3.common.base_class import BaseAlgorithm
        from stable_baselines3.common.monitor import Monitor

        from rl_core.algorithms.base import TrainingCallback

        cls, _ = load_gym_algorithm_from_path(path)
        env = Monitor(gym.make("CartPole-v1"))
        obs, _ = env.reset()
        if issubclass(cls, BaseAlgorithm):
            model = cls("MlpPolicy", env, verbose=0)
            model.predict(obs, deterministic=True)
            model.learn(total_timesteps=8, progress_bar=False)
        else:
            algo = cls(env, {}, None, "cpu")
            action, _ = algo.predict(obs, deterministic=True)
            env.step(action)
            calls = {"n": 0}

            def writer(_num_timesteps: int, _episode_reward: float | None = None, _episode_length: float | None = None) -> bool:
                calls["n"] += 1
                return calls["n"] < 3

            algo.learn(total_timesteps=8, callback=TrainingCallback(writer))
        env.close()
        return {"ok": True, "error": None, "traceback": None}
    except Exception as exc:  # noqa: BLE001 - dry-run must report, never raise
        return {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}


def validate_gym_algorithm(slug: str) -> dict[str, Any]:
    return validate_gym_algorithm_at(_module_path(CUSTOM_GYM_ALGOS_DIR, slug))


# ----------------------------------------------------------- AlphaZero trainers

def load_alphazero_trainer_from_path(path: Path) -> type:
    from rl_core.alphazero.base import AlphaZeroTrainer

    module = _load_module_from_path(path)
    cls = getattr(module, "ALGORITHM_CLASS", None)
    if cls is None or not (inspect.isclass(cls) and issubclass(cls, AlphaZeroTrainer)):
        raise PluginError(
            "Скрипт AlphaZero должен определять ALGORITHM_CLASS, наследующий "
            "rl_core.alphazero.base.AlphaZeroTrainer"
        )
    return cls


def load_alphazero_trainer(slug: str) -> type:
    return load_alphazero_trainer_from_path(_module_path(CUSTOM_ALPHAZERO_ALGOS_DIR, slug))


def alphazero_trainer_meta(slug: str) -> dict[str, Any]:
    path = _module_path(CUSTOM_ALPHAZERO_ALGOS_DIR, slug)
    module = _load_module_from_path(path)
    return {
        "id": f"custom:{slug}",
        "slug": slug,
        "name": getattr(module, "NAME", slug),
        "kind": "alphazero",
        "description": getattr(module, "DESCRIPTION", ""),
        "hyperparams": _hyperparams_of(module),
        "is_custom": True,
    }


def list_alphazero_trainer_slugs() -> list[str]:
    if not CUSTOM_ALPHAZERO_ALGOS_DIR.exists():
        return []
    return sorted(p.stem for p in CUSTOM_ALPHAZERO_ALGOS_DIR.glob("*.py"))


def validate_alphazero_trainer_at(path: Path) -> dict[str, Any]:
    try:
        from rl_core.games import make_game

        cls = load_alphazero_trainer_from_path(path)
        game_cls = make_game("tic_tac_toe").__class__
        trainer = cls(
            game_cls,
            {"num_simulations": 2, "games_per_iteration": 1, "eval_games": 2, "eval_num_simulations": 2, "epochs": 1},
            "cpu",
        )
        trainer.run_iteration(1)
        return {"ok": True, "error": None, "traceback": None}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}


def validate_alphazero_trainer(slug: str) -> dict[str, Any]:
    return validate_alphazero_trainer_at(_module_path(CUSTOM_ALPHAZERO_ALGOS_DIR, slug))


# ------------------------------------------------------------- Reward functions

def load_reward_fn_from_path(path: Path) -> Callable[..., float]:
    module = _load_module_from_path(path)
    fn = getattr(module, "REWARD_FN", None) or getattr(module, "shape_reward", None)
    if fn is None or not callable(fn):
        raise PluginError("Скрипт reward-функции должен определять REWARD_FN или функцию shape_reward(...)")
    return fn


def load_reward_fn(slug: str) -> Callable[..., float]:
    return load_reward_fn_from_path(_module_path(CUSTOM_REWARDS_DIR, slug))


def reward_fn_meta(slug: str) -> dict[str, Any]:
    path = _module_path(CUSTOM_REWARDS_DIR, slug)
    module = _load_module_from_path(path)
    return {
        "id": slug,
        "slug": slug,
        "name": getattr(module, "NAME", slug),
        "kind": "reward",
        "description": getattr(module, "DESCRIPTION", ""),
        "hyperparams": [],
        "is_custom": True,
    }


def list_reward_fn_slugs() -> list[str]:
    if not CUSTOM_REWARDS_DIR.exists():
        return []
    return sorted(p.stem for p in CUSTOM_REWARDS_DIR.glob("*.py"))


def validate_reward_fn_at(path: Path) -> dict[str, Any]:
    try:
        fn = load_reward_fn_from_path(path)
        fake_obs = [0.0, 0.0, 0.0, 0.0]
        result = fn(fake_obs, 0, 1.0, fake_obs, False, False, {})
        float(result)
        return {"ok": True, "error": None, "traceback": None}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}


def validate_reward_fn(slug: str) -> dict[str, Any]:
    return validate_reward_fn_at(_module_path(CUSTOM_REWARDS_DIR, slug))
