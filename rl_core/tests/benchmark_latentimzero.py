"""Repeatable ResearchImZero-floor and model-call benchmark for LatentImZero v10.1.

This is intentionally not collected by pytest.  It runs equal env-step
budgets and seeds for all algorithms and writes both learning curves and a
go/no-go summary:

    python -m rl_core.tests.benchmark_latentimzero --tier smoke
    python -m rl_core.tests.benchmark_latentimzero --tier tier0 --steps 100000

The default panel compares the exact Research core against v10.1. Optional
dependencies are reported as skipped environments, never silently replaced;
no improvement is inferred without measured runs.
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
from rl_core.algorithms.native.latentimzero import (
    DEFAULT_HYPERPARAMS as LATENTIMZERO_DEFAULTS,
    NativeLatentImZero,
)
from rl_core.algorithms.native.researchimzero import (
    DEFAULT_HYPERPARAMS as RESEARCHIMZERO_DEFAULTS,
    NativeResearchImZero,
)


ALGORITHMS = {
    "researchimzero": (NativeResearchImZero, RESEARCHIMZERO_DEFAULTS),
    "latentimzero": (NativeLatentImZero, LATENTIMZERO_DEFAULTS),
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
    "research_pure": {
        "force_research_mode": 1, "uncertainty_enabled": 0,
        "adaptive_train_steps": 0, "replay_success_fraction": 0.0,
        "adaptive_closed_loop": 0, "path_consistency_coef": 0.0,
        "uncertainty_sve_beta": 0.0, "uncertainty_extra_simulations_max": 0,
        "learning_progress_priority_weight": 0.0,
    },
    "no_adaptive_replay": {"adaptive_train_steps": 0},
    "no_success_replay": {"replay_success_fraction": 0.0},
    "no_adaptive_horizon": {"adaptive_closed_loop": 0},
    "no_uncertainty": {"uncertainty_enabled": 0},
    "no_path_consistency": {"path_consistency_coef": 0.0},
    "no_uncertainty_sve": {"uncertainty_sve_beta": 0.0},
    "no_extra_sims": {"uncertainty_extra_simulations_max": 0},
    "no_learning_progress": {"learning_progress_priority_weight": 0.0},
    "fixed_max": {
        "voc_enabled": 1,
        "voc_safety_patience": 1_000_000_000,
        "voc_label_sample_fraction": 0.0,
        "voc_random_audit_fraction": 0.0,
        "voc_targeted_audit_fraction": 0.0,
    },
    "shadow_only": {"voc_safety_patience": 1_000_000_000},
    "no_common_eval_filter": {"voc_uncertainty_filter": 1.0},
    "full_shadow_labels": {"voc_label_sample_fraction": 1.0},
    "random_audit_only": {"voc_targeted_audit_fraction": 0.0},
    "no_voc_adaptive_scheduler": {"voc_enabled": 0},
}


@dataclass
class Curve:
    steps: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    model_calls: list[float] = field(default_factory=list)
    search_budgets: list[float] = field(default_factory=list)
    health_statuses: list[float] = field(default_factory=list)

    def callback(
        self,
        step: int,
        reward: float | None = None,
        length: int | None = None,
        metrics: dict[str, float] | None = None,
    ) -> bool:
        if metrics:
            if "total_model_calls_mean" in metrics:
                self.model_calls.append(float(metrics["total_model_calls_mean"]))
            elif "model_expansions_mean" in metrics:
                self.model_calls.append(float(metrics["model_expansions_mean"]))
            if "requested_budget_mean" in metrics:
                self.search_budgets.append(float(metrics["requested_budget_mean"]))
            elif "search_budget_mean" in metrics:
                self.search_budgets.append(float(metrics["search_budget_mean"]))
            if "voc_health_status" in metrics:
                self.health_statuses.append(float(metrics["voc_health_status"]))
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
        "mean_model_calls_per_action": (
            float(np.mean(curve.model_calls)) if curve.model_calls else None
        ),
        "mean_search_budget": (
            float(np.mean(curve.search_budgets)) if curve.search_budgets else None
        ),
        "final_voc_health_status": (
            curve.health_statuses[-1] if curve.health_statuses else None
        ),
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
                    "median_model_calls_per_action": (
                        float(np.median([
                            row.get("mean_model_calls_per_action")
                            for row in selected
                            if row.get("mean_model_calls_per_action") is not None
                        ]))
                        if any(row.get("mean_model_calls_per_action") is not None for row in selected)
                        else None
                    ),
                }
        summary["environments"][env_id] = per_algorithm

    regressions = 0
    comparisons = 0
    for values in summary["environments"].values():
        if "latentimzero" not in values or "researchimzero" not in values:
            continue
        latent = values["latentimzero"]["median_auc"]
        research = values["researchimzero"]["median_auc"]
        if latent is None or research is None:
            continue
        if not np.isfinite(latent) or not np.isfinite(research):
            continue
        scale = max(abs(research), 1.0)
        comparisons += 1
        regressions += int((latent - research) / scale < -0.10)
    summary["criterion"] = {
        "comparisons": comparisons,
        "regressions_over_10_percent": regressions,
        "passed": comparisons > 0 and regressions == 0,
        "required": "No >10% median-AUC regression versus ResearchImZero",
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
