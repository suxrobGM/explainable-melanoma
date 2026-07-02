"""Tests for the multi-seed aggregation logic."""

from pathlib import Path

from melanomanet.evaluation.seeds import _SCALAR_KEYS, _report, _t_multiplier


def test_t_multiplier_small_samples() -> None:
    assert _t_multiplier(2) == 12.706  # df = 1
    assert _t_multiplier(3) == 4.303  # df = 2
    assert _t_multiplier(100) == 1.96  # normal approximation


def test_report_writes_per_run_and_summary_csvs(tmp_path: Path) -> None:
    metrics = {
        25: dict.fromkeys(_SCALAR_KEYS, 0.80),
        26: dict.fromkeys(_SCALAR_KEYS, 0.90),
    }

    _report(metrics, tmp_path)

    per_run = (tmp_path / "seeds_per_run.csv").read_text().strip().splitlines()
    assert per_run[0] == "seed," + ",".join(_SCALAR_KEYS)
    assert len(per_run) == 3  # header + 2 seeds

    summary = (tmp_path / "seeds_summary.csv").read_text().strip().splitlines()
    assert summary[0] == "metric,mean,std,ci95_halfwidth,n"
    accuracy_row = summary[1].split(",")
    assert accuracy_row[0] == "accuracy"
    assert float(accuracy_row[1]) == 0.85  # mean of 0.80 and 0.90


def test_report_handles_no_seeds(tmp_path: Path) -> None:
    _report({}, tmp_path)
    assert not (tmp_path / "seeds_summary.csv").exists()
