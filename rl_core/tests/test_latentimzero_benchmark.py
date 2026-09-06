from __future__ import annotations

from rl_core.tests.benchmark_latentimzero import ABLATIONS, summarize


def _row(algorithm: str, environment: str, auc: float) -> dict:
    return {
        "algorithm": algorithm,
        "environment": environment,
        "auc": auc,
        "final_reward": auc,
        "wall_seconds": 1.0,
        "steps_to_threshold": 10,
    }


def test_floor_accepts_no_large_research_regressions() -> None:
    rows = []
    for index in range(3):
        environment = f"env-{index}"
        rows.extend([
            _row("researchimzero", environment, 100.0),
            _row("latentimzero", environment, 100.0),
        ])
    criterion = summarize(rows)["criterion"]
    assert criterion["comparisons"] == 3
    assert criterion["regressions_over_10_percent"] == 0
    assert criterion["passed"] is True


def test_floor_rejects_large_research_regression() -> None:
    rows = [
        _row("researchimzero", "env", 100.0),
        _row("latentimzero", "env", 80.0),
    ]
    assert summarize(rows)["criterion"]["passed"] is False


def test_incomplete_episodes_remain_missing_not_infinite() -> None:
    rows = [
        {**_row(algorithm, "slow-env", 0.0), "auc": None, "final_reward": None}
        for algorithm in ("latentimzero", "researchimzero")
    ]
    summary = summarize(rows)
    assert summary["environments"]["slow-env"]["latentimzero"]["median_auc"] is None
    assert summary["criterion"]["passed"] is False


def test_v7_ablations_are_registered() -> None:
    assert set(ABLATIONS) == {
        "research_pure", "uncertainty_disabled", "uncertainty_only",
    }
