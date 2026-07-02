"""Train FastCAV concept vectors using a pre-trained model checkpoint."""

from pathlib import Path

from rich.table import Table

from ..config import Config
from ..data.transforms import get_val_transforms
from ..inference.loaders import load_model
from ..utils.console import console
from ..utils.env import resolve_device
from .fastcav import FastCAV


def train_fastcav(config: Config, checkpoint_path: str) -> None:
    """
    Train FastCAV concept vectors.

    Args:
        config: Configuration
        checkpoint_path: Path to model checkpoint
    """
    device = resolve_device(config.device)
    console.print(f"[bold green]Using device: {device}[/bold green]")

    concepts_dir = Path(config.fastcav.concepts_dir)
    if not concepts_dir.exists():
        console.print(f"[bold red]Error: Concepts directory not found: {concepts_dir}[/bold red]")
        console.print(
            "[yellow]Run 'poe generate-concepts' first to create concept examples.[/yellow]"
        )
        return

    model = load_model(config, checkpoint_path, device)

    # Initialize FastCAV
    console.print("\n[bold cyan]Initializing FastCAV...[/bold cyan]")
    fastcav = FastCAV(model=model, concepts_dir=concepts_dir, device=device)

    console.print(f"Found {len(fastcav.available_concepts)} concepts:")
    for concept in fastcav.available_concepts:
        console.print(f"  - {concept}")

    if not fastcav.available_concepts:
        console.print("[bold red]No concepts found. Exiting.[/bold red]")
        return

    # Train all CAVs
    console.print("\n[bold cyan]Training CAVs...[/bold cyan]")
    transform = get_val_transforms(config.data.image_size)
    accuracies = fastcav.train_all_cavs(transform=transform, batch_size=config.fastcav.batch_size)

    # Save CAVs
    cavs_path = Path(config.fastcav.cavs_path)
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
        console.print(f"[bold]CAVs with good accuracy (>0.6): {good_cavs}/{len(valid_accs)}[/bold]")
