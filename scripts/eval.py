# Author: Sukhrobbek Ilyosbekov
# Date: 2025-12-09

"""
Evaluation script for trained MelanomaNet.

Args:
    --config: Path to configuration YAML file
    --checkpoint: Path to model checkpoint file

Usage:
    python scripts/eval.py --checkpoint checkpoints/best_model.pth --config config.yaml
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import typer
import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from sklearn.metrics import confusion_matrix, roc_curve

from melanomanet.data.dataloader import create_data_loaders
from melanomanet.models.melanomanet import create_model
from melanomanet.utils.checkpoint import load_checkpoint
from melanomanet.utils.metrics import MetricsTracker

console = Console()


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list, save_path: Path
) -> None:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    console.print(f"[green]Confusion matrix saved to {save_path}[/green]")


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list,
    per_class_auc: np.ndarray,
    save_path: Path,
) -> None:
    """Plot per-class one-vs-rest ROC curves on a single axis."""
    present = np.unique(y_true)

    plt.figure(figsize=(8, 6))
    for c in present:
        fpr, tpr, _ = roc_curve((y_true == c).astype(int), y_prob[:, c])
        auc_c = per_class_auc[c]
        plt.plot(fpr, tpr, linewidth=1.5, label=f"{class_names[c]} (AUC={auc_c:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("Per-class ROC Curves (one-vs-rest)")
    plt.legend(fontsize=8, loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    console.print(f"[green]ROC curves saved to {save_path}[/green]")


def evaluate(config_path: str, checkpoint_path: str) -> None:
    """
    Evaluate trained model on test set.

    Args:
        config_path: Path to config file
        checkpoint_path: Path to model checkpoint
    """
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    console.print(f"[bold green]Using device: {device}[/bold green]")

    # Create data loaders
    console.print("[bold]Loading data...[/bold]")
    _, _, test_loader, _ = create_data_loaders(config)

    # Load model
    console.print("[bold]Loading model...[/bold]")
    model = create_model(config).to(device)
    load_checkpoint(Path(checkpoint_path), model, device=device)
    model.eval()

    # Evaluate
    console.print("\n[bold cyan]Evaluating on test set...[/bold cyan]\n")

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        with Progress(
            TextColumn("[bold cyan]Testing"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Testing", total=len(test_loader))

            for images, labels, _ in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

                progress.update(task, advance=1)

    # Calculate metrics
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.concatenate(all_probs, axis=0)  # (N, C)

    class_names = config["data"]["class_names"]
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
    present = np.unique(y_true)
    per_class_table = Table(title="Per-class Sensitivity / Specificity / AUC")
    per_class_table.add_column("Class", style="cyan")
    per_class_table.add_column("Sensitivity", style="green")
    per_class_table.add_column("Specificity", style="green")
    per_class_table.add_column("AUC (OvR)", style="green")
    for c in present:
        auc_c = metrics["per_class_auc"]
        auc_str = f"{auc_c[c]:.4f}" if auc_c is not None else "-"
        per_class_table.add_row(
            class_names[c],
            f"{metrics['per_class_sensitivity'][c]:.4f}",
            f"{metrics['per_class_specificity'][c]:.4f}",
            auc_str,
        )
    console.print(per_class_table)
    console.print()

    # Print classification report
    metrics_tracker.print_classification_report(y_true, y_pred, class_names)

    # Create output directory
    output_dir = Path(config["paths"]["output_dir"])
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
        f"[green]Per-sample predictions saved to "
        f"{output_dir / 'eval_predictions.npz'}[/green]"
    )

    # Plot and save visualizations
    if config["evaluation"]["save_confusion_matrix"]:
        plot_confusion_matrix(
            y_true, y_pred, class_names, output_dir / "confusion_matrix.png"
        )
    if metrics["per_class_auc"] is not None:
        plot_roc_curves(
            y_true,
            y_prob,
            class_names,
            metrics["per_class_auc"],
            output_dir / "roc_curves.png",
        )

    console.print("\n[bold green]Evaluation complete![/bold green]")


def main(
    config: Annotated[str, typer.Option(help="Path to config file")] = "config.yaml",
    checkpoint: Annotated[
        str, typer.Option(help="Path to checkpoint")
    ] = "checkpoints/best_model.pth",
):
    """Evaluate MelanomaNet."""
    evaluate(config, checkpoint)


if __name__ == "__main__":
    typer.run(main)
