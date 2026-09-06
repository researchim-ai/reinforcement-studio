from __future__ import annotations

from rl_core.tests.benchmark_latentimzero import summarize


def _row(algorithm: str, environment: str, auc: float) -> dict:
    return {
        "algorithm": algorithm,
        "environment": environment,
        "auc": auc,
        "final_reward": auc,
        "wall_seconds": 1.0,
        "steps_to_threshold": 10,
    }


def test_go_no_go_requires_four_wins_and_limits_regressions() -> None:
    rows = []
    for index in range(6):
        environment = f"env-{index}"
        rows.extend([
            _row("unizero", environment, 100.0),
            _row("efficientzero", environment, 90.0),
            _row("latentimzero", environment, 125.0 if index < 4 else 95.0),
        ])
    criterion = summarize(rows)["criterion"]
    assert criterion["wins_at_least_20_percent"] == 4
    assert criterion["losses_over_10_percent"] == 0
    assert criterion["passed"] is True


def test_go_no_go_rejects_two_large_regressions() -> None:
    rows = []
    for index in range(6):
        environment = f"env-{index}"
        rows.extend([
            _row("unizero", environment, 100.0),
            _row("efficientzero", environment, 90.0),
            _row("latentimzero", environment, 125.0 if index < 4 else 80.0),
        ])
    assert summarize(rows)["criterion"]["passed"] is False


def test_incomplete_episodes_remain_missing_not_infinite() -> None:
    rows = [
        {**_row(algorithm, "slow-env", 0.0), "auc": None, "final_reward": None}
        for algorithm in ("latentimzero", "unizero", "efficientzero")
    ]
    summary = summarize(rows)
    assert summary["environments"]["slow-env"]["latentimzero"]["median_auc"] is None
    assert summary["criterion"]["passed"] is False
