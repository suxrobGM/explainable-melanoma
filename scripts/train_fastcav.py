# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-12-09

"""
Train FastCAV concept vectors using a pre-trained model checkpoint.

This script loads a trained MelanomaNet model and trains Concept Activation
Vectors (CAVs) for explainability using the FastCAV methodology.

Args:
    --config: Path to configuration YAML file
    --checkpoint: Path to model checkpoint file

Usage:
    python scripts/train_fastcav.py --config config.yaml --checkpoint checkpoints/best_model.pth
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
from typing import Any

import torch
import yaml
from rich.console import Console
from rich.table import Table

from melanomanet.data.transforms import get_val_transforms
from melanomanet.explainability.fastcav import FastCAV
from melanomanet.models.melanomanet import create_model
from melanomanet.utils.checkpoint import load_checkpoint

console = Console()


def train_fastcav(
    config: dict[str, Any],
    checkpoint_path: str,
) -> None:
    """
    Train FastCAV concept vectors.

    Args:
        config: Configuration dictionary
        checkpoint_path: Path to model checkpoint
    """
    # Setup device
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    console.print(f"[bold green]Using device: {device}[/bold green]")

    # Get FastCAV config
    fastcav_config = config.get("fastcav", {})
    concepts_dir = Path(fastcav_config.get("concepts_dir", "./data/concepts"))

    if not concepts_dir.exists():
        console.print(
            f"[bold red]Error: Concepts directory not found: {concepts_dir}[/bold red]"
        )
        console.print(
            "[yellow]Run 'pdm run generate-concepts' first to create concept examples.[/yellow]"
        )
        return

    # Create and load model
    console.print("[bold]Loading model...[/bold]")
    model = create_model(config).to(device)

    # Load checkpoint
    checkpoint = load_checkpoint(
        Path(checkpoint_path),
        model,
        device=device,
    )
    epoch = checkpoint.get("epoch", "unknown")
    metrics = checkpoint.get("metrics", {})
    console.print(f"[bold green]Loaded checkpoint from epoch {epoch}[/bold green]")
    if "f1" in metrics:
        console.print(f"[bold green]Model F1 score: {metrics['f1']:.4f}[/bold green]")

    model.eval()

    # Initialize FastCAV
    console.print("\n[bold cyan]Initializing FastCAV...[/bold cyan]")
    fastcav = FastCAV(
        model=model,
        concepts_dir=concepts_dir,
        device=device,
    )

    console.print(f"Found {len(fastcav.available_concepts)} concepts:")
    for concept in fastcav.available_concepts:
        console.print(f"  - {concept}")

    if not fastcav.available_concepts:
        console.print("[bold red]No concepts found. Exiting.[/bold red]")
        return

    # Get transforms
    transform = get_val_transforms(config)

    # Train all CAVs
    console.print("\n[bold cyan]Training CAVs...[/bold cyan]")
    batch_size = fastcav_config.get("batch_size", 32)
    accuracies = fastcav.train_all_cavs(transform=transform, batch_size=batch_size)

    # Save CAVs
    cavs_path = Path(fastcav_config.get("cavs_path", "./checkpoints/cavs.pth"))
    cavs_path.parent.mkdir(parents=True, exist_ok=True)
    fastcav.save_cavs(cavs_path)

    console.print(f"\n[bold green]CAVs saved to: {cavs_path}[/bold green]")

    # Print summary table
    table = Table(title="CAV Training Summary")
    table.add_column("Concept", style="cyan")
    table.add_column("Accuracy", style="green")
    table.add_column("Status", style="bold")

    for concept, acc in accuracies.items():
        status = "[green]Good[/green]" if acc > 0.6 else "[yellow]Low[/yellow]"
        table.add_row(concept, f"{acc:.3f}", status)

    console.print(table)

    # Summary statistics
    valid_accs = [a for a in accuracies.values() if a > 0]
    if valid_accs:
        avg_acc = sum(valid_accs) / len(valid_accs)
        console.print(f"\n[bold]Average CAV accuracy: {avg_acc:.3f}[/bold]")
        good_cavs = sum(1 for a in valid_accs if a > 0.6)
        console.print(
            f"[bold]CAVs with good accuracy (>0.6): {good_cavs}/{len(valid_accs)}[/bold]"
        )


def main():
    parser = argparse.ArgumentParser(description="Train FastCAV concept vectors")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint file",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train_fastcav(config, args.checkpoint)


if __name__ == "__main__":
    main()
