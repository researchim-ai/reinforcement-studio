"""Contract for custom Gym-track algorithms plugged in via the in-app editor.

Two ways to write a custom algorithm (see rl_core/plugins/loader.py and the
templates surfaced on the Plugins page):

1. Subclass `stable_baselines3.common.base_class.BaseAlgorithm` (e.g. PPO)
   and override whatever you want (network, `train()`, ...). The runner
   detects this via `isinstance(obj, BaseAlgorithm)` and reuses the existing
   SB3 runner + MetricsCallback pipeline unchanged — the easiest path if you
   just want to tweak an existing algorithm.
2. Subclass `CustomAlgorithm` below for a fully from-scratch implementation
   (pure PyTorch, no SB3 dependency at all). The runner drives it through
   `rl_core/algorithms/custom_runner.py` using `TrainingCallback` instead of
   an SB3 callback.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable


class TrainingCallback:
    """Passed into `CustomAlgorithm.learn()`. Mirrors what SB3's
    `MetricsCallback` already does for built-in runs (write metrics.json,
    honor stop.flag) but without any SB3 dependency, so from-scratch
    algorithms stay lightweight and framework-agnostic.
    """

    def __init__(self, writer: Callable[[int, float | None, float | None], bool]) -> None:
        self._writer = writer

    def on_step(
        self,
        num_timesteps: int,
        episode_reward: float | None = None,
        episode_length: float | None = None,
    ) -> bool:
        """Call this regularly from inside `learn()` (e.g. once per env
        step). Pass the finished episode's reward/length when one just
        ended, so the Training Monitor can show running averages. Returns
        `False` once the run should stop (user pressed Stop) — break your
        loop when it does."""
        return self._writer(num_timesteps, episode_reward, episode_length)


class CustomAlgorithm(ABC):
    """Full from-scratch RL algorithm contract (Gym track).

    `env` is a fully wrapped `gymnasium.Env` (Monitor + any configured
    wrappers already applied by the runner) — use it like any Gym env.
    """

    def __init__(self, env: Any, hyperparams: dict[str, Any], seed: int | None, device: str) -> None:
        self.env = env
        self.hyperparams = hyperparams
        self.seed = seed
        self.device = device

    @abstractmethod
    def learn(self, total_timesteps: int, callback: TrainingCallback) -> None:
        """Run the full training loop for up to `total_timesteps` steps,
        calling `callback.on_step(...)` regularly and stopping early once it
        returns `False`."""

    @abstractmethod
    def predict(self, obs: Any, deterministic: bool = True) -> tuple[Any, Any]:
        """Returns `(action, state)` — mirrors SB3's
        `BaseAlgorithm.predict` signature so the same live-frame rendering
        code in the runner works for both SB3 and from-scratch algorithms."""

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path, env: Any) -> "CustomAlgorithm": ...
