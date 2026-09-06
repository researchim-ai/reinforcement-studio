"""Repeatable sample-efficiency benchmark for LatentImZero.

This is intentionally not collected by pytest.  It runs equal env-step
budgets and seeds for all algorithms and writes both learning curves and a
go/no-go summary:

    python -m rl_core.tests.benchmark_latentimzero --tier smoke
    python -m rl_core.tests.benchmark_latentimzero --tier tier0 --steps 100000

The default Tier-0 panel follows the LatentImZero design plan. Optional
dependencies are reported as skipped environments, never silently replaced.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

import rl_core.envs.registry  # noqa: F401 - registers Studio environments
from rl_core.algorithms.base import TrainingCallback
from rl_core.algorithms.native.efficientzero import (
    DEFAULT_HYPERPARAMS as EFFICIENTZERO_DEFAULTS,
    NativeEfficientZero,
)
from rl_core.algorithms.native.latentimzero import (
    DEFAULT_HYPERPARAMS as LATENTIMZERO_DEFAULTS,
    NativeLatentImZero,
)
from rl_core.algorithms.native.unizero import DEFAULT_HYPERPARAMS as UNIZERO_DEFAULTS, NativeUniZero


ALGORITHMS = {
    "latentimzero": (NativeLatentImZero, LATENTIMZERO_DEFAULTS),
    "unizero": (NativeUniZero, UNIZERO_DEFAULTS),
    "efficientzero": (NativeEfficientZero, EFFICIENTZERO_DEFAULTS),
}
TIERS = {
    "smoke": ["CartPole-v1", "Pendulum-v1"],
    "tier0": [
        "CartPole-v1",
        "Pendulum-v1",
        "MountainCar-v0",
        "VisualMemoryMaze-v0",
        "MiniGrid-DoorKey-8x8-Img-v0",
        "CarRacing-v3",
    ],
}
DEFAULT_THRESHOLDS = {
    "CartPole-v1": 475.0,
    "Pendulum-v1": -300.0,
    "MountainCar-v0": -110.0,
}
ABLATIONS = {
    "no_stochastic_state": {"use_stochastic_state": 0},
    "no_imagination_ac": {
        "actor_loss_coef": 0.0,
        "critic_loss_coef": 0.0,
        "imagination_horizon": 1,
    },
    "no_continue": {"use_continue": 0},
    "no_ensemble_bonus": {
        "ensemble_size": 1,
        "exploration_bonus_coef": 0.0,
        "uncertainty_scale": 0.0,
    },
    "no_planner_distillation": {"distill_coef": 0.0, "fresh_distill_fraction": 0.0},
}


@dataclass
class Curve:
    steps: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)

    def callback(
        self,
        step: int,
        reward: float | None = None,
        length: int | None = None,
        metrics: dict[str, float] | None = None,
    ) -> bool:
        del metrics
        if reward is not None:
            self.steps.append(int(step))
            self.rewards.append(float(reward))
            self.lengths.append(int(length or 0))
        return True


def _auc(curve: Curve, budget: int) -> float | None:
    if not curve.steps:
        return None
    x = np.asarray([0, *curve.steps, budget], dtype=np.float64)
    y = np.asarray([curve.rewards[0], *curve.rewards, curve.rewards[-1]], dtype=np.float64)
    x = np.maximum.accumulate(np.clip(x, 0, budget))
    return float(np.trapz(y, x) / max(1, budget))


def _steps_to_threshold(curve: Curve, threshold: float | None) -> int | None:
    if threshold is None:
        return None
    return next((step for step, reward in zip(curve.steps, curve.rewards) if reward >= threshold), None)


def run_one(
    algorithm_id: str,
    environment_id: str,
    seed: int,
    steps: int,
    device: str,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    algorithm_cls, defaults = ALGORITHMS[algorithm_id]
    env = gym.make(environment_id)
    curve = Curve()
    started = time.perf_counter()
    try:
        hyperparams = {
            **defaults,
            "learning_starts": min(int(defaults.get("learning_starts", 500)), max(2, steps // 20)),
            **overrides,
        }
        algorithm_cls(env, hyperparams, seed, device).learn(steps, TrainingCallback(curve.callback))
    finally:
        env.close()
    threshold = DEFAULT_THRESHOLDS.get(environment_id)
    return {
        "algorithm": algorithm_id,
        "environment": environment_id,
        "seed": seed,
        "budget": steps,
        "auc": _auc(curve, steps),
        "steps_to_threshold": _steps_to_threshold(curve, threshold),
        "final_reward": float(np.mean(curve.rewards[-10:])) if curve.rewards else None,
        "episodes": len(curve.rewards),
        "mean_episode_length": float(np.mean(curve.lengths)) if curve.lengths else None,
        "wall_seconds": time.perf_counter() - started,
        "curve": {"steps": curve.steps, "rewards": curve.rewards, "lengths": curve.lengths},
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"environments": {}, "criterion": {}}
    for env_id in sorted({row["environment"] for row in rows}):
        env_rows = [row for row in rows if row["environment"] == env_id]
        per_algorithm = {}
        for algorithm_id in ALGORITHMS:
            selected = [row for row in env_rows if row["algorithm"] == algorithm_id]
            if selected:
                aucs = [row["auc"] for row in selected if row["auc"] is not None]
                final_rewards = [
                    row["final_reward"] for row in selected if row["final_reward"] is not None
                ]
                per_algorithm[algorithm_id] = {
                    "median_auc": float(np.median(aucs)) if aucs else None,
                    "median_final_reward": (
                        float(np.median(final_rewards)) if final_rewards else None
                    ),
                    "median_wall_seconds": float(np.median([row["wall_seconds"] for row in selected])),
                    "threshold_successes": sum(row["steps_to_threshold"] is not None for row in selected),
                    "seeds": len(selected),
                }
        summary["environments"][env_id] = per_algorithm

    wins = 0
    losses = 0
    for values in summary["environments"].values():
        if "latentimzero" not in values or len(values) < 3:
            continue
        latent = values["latentimzero"]["median_auc"]
        baseline_values = [
            values["unizero"]["median_auc"],
            values["efficientzero"]["median_auc"],
        ]
        if latent is None or any(value is None for value in baseline_values):
            continue
        baseline = max(baseline_values)
        if not np.isfinite(latent) or not np.isfinite(baseline):
            continue
        scale = max(abs(baseline), 1.0)
        relative = (latent - baseline) / scale
        wins += int(relative >= 0.20)
        losses += int(relative < -0.10)
    summary["criterion"] = {
        "wins_at_least_20_percent": wins,
        "losses_over_10_percent": losses,
        "passed": wins >= 4 and losses <= 1,
        "required": "≥20% median AUC gain in 4/6 environments and >10% loss in at most one",
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=TIERS, default="smoke")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("latentimzero_benchmark.json"))
    parser.add_argument("--hyperparams", type=json.loads, default={})
    parser.add_argument("--ablations", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for environment_id in TIERS[args.tier]:
        for algorithm_id in ALGORITHMS:
            for seed in range(args.seeds):
                try:
                    row = run_one(algorithm_id, environment_id, seed, args.steps, args.device, args.hyperparams)
                    rows.append(row)
                    print(json.dumps({key: value for key, value in row.items() if key != "curve"}))
                except (gym.error.Error, ImportError, ModuleNotFoundError) as exc:
                    skipped.append({"environment": environment_id, "reason": str(exc)})
                    break
            if skipped and skipped[-1]["environment"] == environment_id:
                break
        if args.ablations and not (skipped and skipped[-1]["environment"] == environment_id):
            for ablation, ablation_overrides in ABLATIONS.items():
                for seed in range(args.seeds):
                    row = run_one(
                        "latentimzero",
                        environment_id,
                        seed,
                        args.steps,
                        args.device,
                        {**args.hyperparams, **ablation_overrides},
                    )
                    row["algorithm"] = f"latentimzero/{ablation}"
                    rows.append(row)
                    print(json.dumps({key: value for key, value in row.items() if key != "curve"}))
    output = {"tier": args.tier, "steps": args.steps, "rows": rows, "skipped": skipped, "summary": summarize(rows)}
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(json.dumps(output["summary"]["criterion"], ensure_ascii=False))


if __name__ == "__main__":
    main()
