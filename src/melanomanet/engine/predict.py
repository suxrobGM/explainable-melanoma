"""Shared batch-prediction loop used by validation and evaluation."""

import numpy as np
import torch
import torch.nn as nn
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader

from ..utils.console import console


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
    description: str = "Predicting",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run the model over a loader and collect labels/predictions/probabilities.

    Args:
        model: Model to evaluate (set to eval mode by the caller)
        loader: Data loader yielding (images, labels, _) batches
        device: Computation device
        criterion: Optional loss to accumulate
        description: Progress bar label

    Returns:
        Tuple of (y_true, y_pred, y_prob, avg_loss); avg_loss is 0.0 when no
        criterion is given.
    """
    model.eval()
    total_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    with (
        torch.no_grad(),
        Progress(
            TextColumn(f"[bold cyan]{description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress,
    ):
        task = progress.add_task(description, total=len(loader))

        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            if criterion is not None:
                total_loss += criterion(outputs, labels).item()

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())  # full (batch, C) matrix

            progress.update(task, advance=1)

    avg_loss = total_loss / len(loader) if criterion is not None else 0.0
    return (
        np.array(all_labels),
        np.array(all_preds),
        np.concatenate(all_probs, axis=0),
        avg_loss,
    )
