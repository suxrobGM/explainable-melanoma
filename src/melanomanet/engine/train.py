"""Training loop for MelanomaNet."""

from pathlib import Path

import torch
import torch.nn as nn
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from torch.amp.grad_scaler import GradScaler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ..config import Config
from ..data.dataloader import create_data_loaders
from ..models.losses import create_criterion
from ..models.melanomanet import create_model
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.console import console
from ..utils.env import resolve_device, set_seed
from ..utils.metrics import MetricsTracker
from .predict import collect_predictions


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: GradScaler | None,
) -> float:
    """Train for one epoch and return the average training loss."""
    model.train()
    total_loss = 0.0

    with Progress(
        TextColumn("[bold blue]Epoch {task.fields[epoch]} [Train]"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("loss: {task.fields[loss]:.4f}"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Training", total=len(train_loader), epoch=epoch + 1, loss=0.0
        )

        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass with mixed precision
            if scaler is not None:
                with torch.amp.autocast_mode.autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            progress.update(task, advance=1, loss=loss.item())

    return total_loss / len(train_loader)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """Validate the model and return a metrics dictionary."""
    y_true, y_pred, y_prob, val_loss = collect_predictions(
        model, val_loader, device, criterion, description=f"Epoch {epoch + 1} [Val]"
    )
    metrics = MetricsTracker().calculate_metrics(y_true, y_pred, y_prob)
    metrics["val_loss"] = val_loss
    return metrics


def train(config: Config, resume_checkpoint: str | None = None) -> None:
    """
    Main training loop.

    Args:
        config: Configuration
        resume_checkpoint: Path to checkpoint file to resume training from
    """
    set_seed(config.seed)
    device = resolve_device(config.device)
    console.print(f"[bold green]Using device: {device}[/bold green]")

    checkpoint_dir = Path(config.paths.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Loading data...[/bold]")
    train_loader, val_loader, _, class_weights = create_data_loaders(config)

    console.print("[bold]Creating model...[/bold]")
    model = create_model(config.model, config.data.num_classes).to(device)
    console.print(f"Model: {config.model.backbone}")
    console.print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss, optimizer, scheduler
    training = config.training
    criterion = create_criterion(training, class_weights.to(device))
    optimizer_cls = Adam if training.optimizer == "adam" else AdamW
    optimizer = optimizer_cls(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    scheduler = (
        CosineAnnealingLR(optimizer, T_max=training.epochs)
        if training.scheduler == "cosine"
        else None
    )
    scaler = GradScaler("cuda") if config.mixed_precision else None

    start_epoch = 0
    best_val_f1 = 0.0

    if resume_checkpoint:
        console.print(
            f"[bold yellow]Resuming from checkpoint: {resume_checkpoint}[/bold yellow]"
        )
        checkpoint = load_checkpoint(
            Path(resume_checkpoint), model, optimizer, scheduler, device
        )
        start_epoch = checkpoint["epoch"] + 1
        best_val_f1 = checkpoint["metrics"].get("f1", 0.0)
        console.print(
            f"[bold green]Resumed from epoch {checkpoint['epoch'] + 1}[/bold green]"
        )
        console.print(f"[bold green]Best F1 so far: {best_val_f1:.4f}[/bold green]\n")

    console.print("\n[bold cyan]Starting training...[/bold cyan]")
    console.print(f"[bold]Total epochs: {training.epochs}[/bold]")
    console.print(f"[bold]Starting from epoch: {start_epoch + 1}[/bold]")
    console.print(f"[bold]Epochs to train: {training.epochs - start_epoch}[/bold]\n")

    for epoch in range(start_epoch, training.epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler
        )
        val_metrics = validate(model, val_loader, criterion, device, epoch)

        if scheduler:
            scheduler.step()

        table = Table(title=f"Epoch {epoch + 1}/{training.epochs}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Train Loss", f"{train_loss:.4f}")
        table.add_row("Val Loss", f"{val_metrics['val_loss']:.4f}")
        table.add_row("Accuracy", f"{val_metrics['accuracy']:.4f}")
        table.add_row("Precision", f"{val_metrics['precision']:.4f}")
        table.add_row("Recall", f"{val_metrics['recall']:.4f}")
        table.add_row("F1 Score", f"{val_metrics['f1']:.4f}")
        console.print(table)
        console.print()

        # Save checkpoint at every epoch (for resuming)
        save_checkpoint(
            model,
            optimizer,
            epoch,
            val_metrics,
            checkpoint_dir / "last_checkpoint.pth",
            scheduler=scheduler,
        )

        # Save best model (based on F1 score for multi-class)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_metrics,
                checkpoint_dir / "best_model.pth",
                scheduler=scheduler,
            )
            console.print(
                f"[bold green]New best model saved! F1: {best_val_f1:.4f}"
                "[/bold green]\n"
            )

        # Save periodic checkpoint (every N epochs, if configured)
        save_interval = training.checkpoint_save_interval
        if save_interval > 0 and (epoch + 1) % save_interval == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_metrics,
                checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth",
                scheduler=scheduler,
            )
            console.print(
                f"[bold cyan]Checkpoint saved at epoch {epoch + 1}[/bold cyan]\n"
            )

    console.print(
        f"\n[bold green]Training complete! Best F1: {best_val_f1:.4f}[/bold green]"
    )
