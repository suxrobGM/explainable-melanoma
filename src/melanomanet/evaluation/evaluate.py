"""Test-set evaluation for a trained MelanomaNet checkpoint."""

from pathlib import Path

import numpy as np
from rich.table import Table

from ..config import Config
from ..data.dataloader import create_data_loaders
from ..engine.predict import collect_predictions
from ..inference.loaders import load_model
from ..utils.console import console
from ..utils.env import resolve_device
from ..utils.metrics import MetricsTracker
from .plots import plot_confusion_matrix, plot_roc_curves


def evaluate(config: Config, checkpoint_path: str) -> dict:
    """
    Evaluate a trained model on the test set.

    Prints metric tables, persists per-sample predictions to
    ``eval_predictions.npz``, and saves confusion-matrix / ROC plots.

    Args:
        config: Configuration
        checkpoint_path: Path to model checkpoint

    Returns:
        The calculated metrics dictionary
    """
    device = resolve_device(config.device)
    console.print(f"[bold green]Using device: {device}[/bold green]")

    console.print("[bold]Loading data...[/bold]")
    _, _, test_loader, _ = create_data_loaders(config)

    model = load_model(config, checkpoint_path, device)

    console.print("\n[bold cyan]Evaluating on test set...[/bold cyan]\n")
    y_true, y_pred, y_prob, _ = collect_predictions(
        model, test_loader, device, description="Testing"
    )

    class_names = config.data.class_names
    metrics_tracker = MetricsTracker()
    metrics = metrics_tracker.calculate_metrics(y_true, y_pred, y_prob)

    # Print overall results
    table = Table(title="Test Set Results (Multi-class)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Accuracy", f"{metrics['accuracy']:.4f}")
    table.add_row("Balanced accuracy", f"{metrics['balanced_accuracy']:.4f}")
    table.add_row("Precision (weighted)", f"{metrics['precision']:.4f}")
    table.add_row("Recall (weighted)", f"{metrics['recall']:.4f}")
    table.add_row("F1 Score (weighted)", f"{metrics['f1']:.4f}")
    table.add_row("F1 Score (macro)", f"{metrics['macro_f1']:.4f}")
    if metrics["macro_auc"] is not None:
        table.add_row("ROC-AUC (macro, OvR)", f"{metrics['macro_auc']:.4f}")
    console.print(table)
    console.print()

    # Per-class sensitivity / specificity / AUC
    per_class_table = Table(title="Per-class Sensitivity / Specificity / AUC")
    per_class_table.add_column("Class", style="cyan")
    per_class_table.add_column("Sensitivity", style="green")
    per_class_table.add_column("Specificity", style="green")
    per_class_table.add_column("AUC (OvR)", style="green")
    per_class_auc = metrics["per_class_auc"]
    for c in np.unique(y_true):
        per_class_table.add_row(
            class_names[c],
            f"{metrics['per_class_sensitivity'][c]:.4f}",
            f"{metrics['per_class_specificity'][c]:.4f}",
            f"{per_class_auc[c]:.4f}" if per_class_auc is not None else "-",
        )
    console.print(per_class_table)
    console.print()

    metrics_tracker.print_classification_report(y_true, y_pred, class_names)

    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Persist per-sample predictions so AUC / faithfulness / plots can be
    # recomputed without re-running the model.
    np.savez_compressed(
        output_dir / "eval_predictions.npz",
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=np.array(class_names),
    )
    console.print(
        f"[green]Per-sample predictions saved to {output_dir / 'eval_predictions.npz'}[/green]"
    )

    if config.evaluation.save_confusion_matrix:
        plot_confusion_matrix(y_true, y_pred, class_names, output_dir / "confusion_matrix.png")
    if metrics["per_class_auc"] is not None:
        plot_roc_curves(
            y_true,
            y_prob,
            class_names,
            metrics["per_class_auc"],
            output_dir / "roc_curves.png",
        )

    console.print("\n[bold green]Evaluation complete![/bold green]")
    return metrics
