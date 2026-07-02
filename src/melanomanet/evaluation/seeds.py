"""
Multi-seed training/evaluation runner for confidence intervals.

Trains and evaluates MelanomaNet under several random seeds, collects the test
metrics from each run, and reports mean +/- 95% confidence interval. This backs
the "mean +/- CI" columns in the paper's results table.

Each seed gets its own checkpoint and output directory so runs do not clobber
one another. Training and evaluation run in subprocesses (``melanoma train`` /
``melanoma eval`` via ``python -m melanomanet.cli``) so CUDA memory is fully
released between seeds; per-seed metrics are recomputed from the
``eval_predictions.npz`` each eval writes.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from rich.table import Table

from ..utils.console import console
from ..utils.metrics import MetricsTracker

# t-multipliers for a two-sided 95% CI, indexed by degrees of freedom (n - 1).
# Falls back to the normal approximation (1.96) for larger samples.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}

# Scalar metrics aggregated across seeds (per-class arrays are handled separately).
_SCALAR_KEYS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "macro_f1",
    "macro_auc",
]


def _t_multiplier(n: int) -> float:
    """Two-sided 95% t-multiplier for ``n`` samples."""
    return _T95.get(n - 1, 1.96)


def _run_cli(args: list[str]) -> None:
    """Run a melanoma CLI subcommand in a subprocess and raise on failure."""
    cmd = [sys.executable, "-m", "melanomanet.cli", *args]
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    subprocess.run(cmd, check=True)


def _metrics_for_seed(pred_path: Path) -> dict:
    """Recompute the metrics dict from a saved eval_predictions.npz."""
    data = np.load(pred_path, allow_pickle=True)
    return MetricsTracker().calculate_metrics(data["y_true"], data["y_pred"], data["y_prob"])


def run_seeds(
    config_path: str,
    seeds: list[int],
    output_root: Path,
    skip_training: bool,
) -> None:
    """Train/evaluate across seeds and aggregate the test metrics.

    Per-seed configs are derived from the raw YAML mapping (seed and artifact
    paths overridden) so the child processes validate them via load_config.
    """
    with open(config_path) as f:
        base_config = yaml.safe_load(f)

    output_root.mkdir(parents=True, exist_ok=True)
    per_seed_metrics: dict[int, dict] = {}

    for seed in seeds:
        seed_dir = output_root / f"seed_{seed}"
        ckpt_dir = seed_dir / "checkpoints"
        out_dir = seed_dir / "outputs"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Per-seed config: fix the seed and redirect all artifacts.
        cfg = {**base_config}
        cfg["seed"] = seed
        cfg["paths"] = {
            **base_config.get("paths", {}),
            "checkpoint_dir": str(ckpt_dir),
            "output_dir": str(out_dir),
        }
        seed_config_path = seed_dir / "config.yaml"
        with open(seed_config_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        console.rule(f"[bold cyan]Seed {seed}")
        checkpoint = ckpt_dir / "best_model.pth"

        if not skip_training:
            _run_cli(["train", "--config", str(seed_config_path)])
        if not checkpoint.exists():
            console.print(f"[yellow]No checkpoint at {checkpoint}; skipping seed {seed}.[/yellow]")
            continue

        _run_cli(
            [
                "eval",
                "--config",
                str(seed_config_path),
                "--checkpoint",
                str(checkpoint),
            ]
        )

        per_seed_metrics[seed] = _metrics_for_seed(out_dir / "eval_predictions.npz")

    _report(per_seed_metrics, output_root)


def _report(per_seed_metrics: dict[int, dict], output_root: Path) -> None:
    """Write per-seed and summary CSVs and print a mean +/- CI table."""
    if not per_seed_metrics:
        console.print("[red]No completed seeds to aggregate.[/red]")
        return

    seeds = sorted(per_seed_metrics)
    n = len(seeds)
    t = _t_multiplier(n)

    # Per-seed CSV.
    per_seed_csv = output_root / "seeds_per_run.csv"
    with open(per_seed_csv, "w") as f:
        f.write("seed," + ",".join(_SCALAR_KEYS) + "\n")
        for seed in seeds:
            row = [f"{per_seed_metrics[seed][k]:.6f}" for k in _SCALAR_KEYS]
            f.write(f"{seed}," + ",".join(row) + "\n")

    # Summary CSV + table.
    table = Table(title=f"Mean +/- 95% CI over {n} seeds: {seeds}")
    table.add_column("Metric", style="cyan")
    table.add_column("Mean", style="green")
    table.add_column("Std", style="green")
    table.add_column("95% CI half-width", style="green")

    summary_csv = output_root / "seeds_summary.csv"
    with open(summary_csv, "w") as f:
        f.write("metric,mean,std,ci95_halfwidth,n\n")
        for key in _SCALAR_KEYS:
            values = np.array([per_seed_metrics[s][key] for s in seeds], dtype=np.float64)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if n > 1 else 0.0
            half = t * std / np.sqrt(n) if n > 1 else 0.0
            f.write(f"{key},{mean:.6f},{std:.6f},{half:.6f},{n}\n")
            table.add_row(key, f"{mean:.4f}", f"{std:.4f}", f"{half:.4f}")

    console.print(table)
    console.print(f"[green]Per-run metrics: {per_seed_csv}[/green]")
    console.print(f"[green]Summary: {summary_csv}[/green]")
