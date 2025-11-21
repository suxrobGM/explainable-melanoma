"""
Model checkpoint saving and loading utilities.
"""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    filepath: Path,
    scheduler: Any = None,
    **kwargs: Any,
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Validation metrics
        filepath: Path to save checkpoint
        scheduler: Optional learning rate scheduler
        **kwargs: Additional items to save in checkpoint
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    # Save scheduler state if provided
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    # Add any additional items
    checkpoint.update(kwargs)

    # Create parent directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        filepath: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        scheduler: Optional learning rate scheduler to load state
        device: Device to load checkpoint on

    Returns:
        Checkpoint dictionary
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
