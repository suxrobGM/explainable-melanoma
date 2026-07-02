"""Evaluation plots: confusion matrix and per-class ROC curves."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

from ..utils.console import console


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], save_path: Path
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
    class_names: list[str],
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
