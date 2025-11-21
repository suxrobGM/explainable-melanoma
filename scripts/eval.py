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

import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from rich.console import Console
from rich.table import Table
from sklearn.metrics import confusion_matrix, roc_curve
from tqdm import tqdm

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


def plot_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray, save_path: Path, auc_score: float
) -> None:
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    console.print(f"[green]ROC curve saved to {save_path}[/green]")


def evaluate(config_path: str, checkpoint_path: str) -> None:
    """
    Evaluate trained model on test set.

    Args:
        config_path: Path to config file
        checkpoint_path: Path to model checkpoint
    """
    # Load config
    with open(config_path, "r") as f:
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
        for images, labels, _ in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    # Calculate metrics
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics_tracker = MetricsTracker()
    metrics = metrics_tracker.calculate_metrics(y_true, y_pred, y_prob)

    # Print results
    table = Table(title="Test Set Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Accuracy", f"{metrics['accuracy']:.4f}")
    table.add_row("Sensitivity (Recall)", f"{metrics['sensitivity']:.4f}")
    table.add_row("Specificity", f"{metrics['specificity']:.4f}")
    table.add_row("Precision", f"{metrics['precision']:.4f}")
    table.add_row("F1 Score", f"{metrics['f1']:.4f}")
    table.add_row("AUC-ROC", f"{metrics['auc']:.4f}")
    table.add_row("", "")
    table.add_row("True Positives", str(metrics["tp"]))
    table.add_row("True Negatives", str(metrics["tn"]))
    table.add_row("False Positives", str(metrics["fp"]))
    table.add_row("False Negatives", str(metrics["fn"]))

    console.print(table)
    console.print()

    # Print classification report
    metrics_tracker.print_classification_report(
        y_true, y_pred, config["data"]["class_names"]
    )

    # Create output directory
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot and save visualizations
    if config["evaluation"]["save_confusion_matrix"]:
        plot_confusion_matrix(
            y_true,
            y_pred,
            config["data"]["class_names"],
            output_dir / "confusion_matrix.png",
        )

    plot_roc_curve(y_true, y_prob, output_dir / "roc_curve.png", metrics["auc"])

    console.print("\n[bold green]Evaluation complete![/bold green]")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MelanomaNet")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint"
    )
    args = parser.parse_args()

    evaluate(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
