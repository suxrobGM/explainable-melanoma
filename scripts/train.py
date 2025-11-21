"""
Training script for MelanomaNet.

Args:
    --config: Path to configuration YAML file
    --resume: (Optional) Path to checkpoint file to resume training from

Usage:
    python scripts/train.py --config config.yaml
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from rich.console import Console
from rich.table import Table
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from melanomanet.data.dataloader import create_data_loaders
from melanomanet.models.losses import create_criterion
from melanomanet.models.melanomanet import create_model
from melanomanet.utils.checkpoint import load_checkpoint, save_checkpoint
from melanomanet.utils.metrics import MetricsTracker

console = Console()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: dict[str, Any],
) -> float:
    """
    Train for one epoch.

    Args:
        model: MelanomaNet model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (cuda/cpu)
        epoch: Current epoch number
        config: Configuration dictionary

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0

    # Mixed precision training
    use_amp = config.get("mixed_precision", False)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")

    for batch_idx, (images, labels, _) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass with mixed precision
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """
    Validate model.

    Args:
        model: MelanomaNet model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device
        epoch: Current epoch number

    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    total_loss = 0.0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")

        for images, labels, _ in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Melanoma probability

    # Calculate metrics
    metrics_tracker = MetricsTracker()
    metrics = metrics_tracker.calculate_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs)
    )

    metrics["val_loss"] = total_loss / len(val_loader)

    return metrics


def train(config: dict[str, Any], resume_checkpoint: str | None = None) -> None:
    """
    Main training loop.

    Args:
        config: Configuration dictionary
        resume_checkpoint: Path to checkpoint file to resume training from
    """
    # Setup
    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    console.print(f"[bold green]Using device: {device}[/bold green]")

    # Create directories
    Path(config["paths"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["log_dir"]).mkdir(parents=True, exist_ok=True)

    # Create data loaders
    console.print("[bold]Loading data...[/bold]")
    train_loader, val_loader, test_loader, class_weights = create_data_loaders(config)

    # Create model
    console.print("[bold]Creating model...[/bold]")
    model = create_model(config).to(device)
    console.print(f"Model: {config['model']['backbone']}")
    console.print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = create_criterion(config, class_weights.to(device))

    if config["training"]["optimizer"] == "adam":
        optimizer = Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

    # Learning rate scheduler
    if config["training"]["scheduler"] == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    else:
        scheduler = None

    # Initialize training state
    start_epoch = 0
    best_val_auc = 0.0
    patience_counter = 0

    # Resume from checkpoint if provided
    if resume_checkpoint:
        console.print(
            f"[bold yellow]Resuming from checkpoint: {resume_checkpoint}[/bold yellow]"
        )
        checkpoint = load_checkpoint(
            Path(resume_checkpoint),
            model,
            optimizer,
            scheduler,
            device,
        )
        start_epoch = checkpoint["epoch"] + 1
        best_val_auc = checkpoint["metrics"].get("auc", 0.0)
        console.print(
            f"[bold green]Resumed from epoch {checkpoint['epoch'] + 1}[/bold green]"
        )
        console.print(f"[bold green]Best AUC so far: {best_val_auc:.4f}[/bold green]\n")

    patience = config["training"]["early_stopping_patience"]

    console.print("\n[bold cyan]Starting training...[/bold cyan]\n")

    for epoch in range(start_epoch, config["training"]["epochs"]):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch)

        # Update learning rate
        if scheduler:
            scheduler.step()

        # Print metrics
        table = Table(title=f"Epoch {epoch+1} Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Train Loss", f"{train_loss:.4f}")
        table.add_row("Val Loss", f"{val_metrics['val_loss']:.4f}")
        table.add_row("Accuracy", f"{val_metrics['accuracy']:.4f}")
        table.add_row("Sensitivity", f"{val_metrics['sensitivity']:.4f}")
        table.add_row("Specificity", f"{val_metrics['specificity']:.4f}")
        table.add_row("AUC", f"{val_metrics['auc']:.4f}")
        table.add_row("F1", f"{val_metrics['f1']:.4f}")

        console.print(table)
        console.print()

        # Save checkpoint at every epoch (for resuming)
        save_checkpoint(
            model,
            optimizer,
            epoch,
            val_metrics,
            Path(config["paths"]["checkpoint_dir"]) / "last_checkpoint.pth",
            scheduler=scheduler,
        )

        # Save best model
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_metrics,
                Path(config["paths"]["checkpoint_dir"]) / "best_model.pth",
                scheduler=scheduler,
            )
            console.print(
                f"[bold green]New best model saved! AUC: {best_val_auc:.4f}[/bold green]\n"
            )
            patience_counter = 0
        else:
            patience_counter += 1

        # Save periodic checkpoint (every N epochs, if configured)
        save_interval = config.get("training", {}).get("checkpoint_save_interval", 0)

        if save_interval > 0 and (epoch + 1) % save_interval == 0:
            checkpoint_path = (
                Path(config["paths"]["checkpoint_dir"])
                / f"checkpoint_epoch_{epoch+1}.pth"
            )
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_metrics,
                checkpoint_path,
                scheduler=scheduler,
            )
            console.print(
                f"[bold cyan]Checkpoint saved at epoch {epoch+1}[/bold cyan]\n"
            )

        # Early stopping
        if patience_counter >= patience:
            console.print(
                f"[bold yellow]Early stopping triggered after {epoch+1} epochs[/bold yellow]"
            )
            break

    console.print(
        f"\n[bold green]Training complete! Best AUC: {best_val_auc:.4f}[/bold green]"
    )


def main():
    parser = argparse.ArgumentParser(description="Train MelanomaNet")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train(config, resume_checkpoint=args.resume)


if __name__ == "__main__":
    main()
