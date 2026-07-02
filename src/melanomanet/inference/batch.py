"""Batch inference over multiple images with a success/failure summary."""

from pathlib import Path

from ..config import Config
from ..utils.console import console
from ..utils.images import SUPPORTED_IMAGE_EXTENSIONS, iter_image_files
from .artifacts import result_png_path
from .core import run_inference


def collect_image_paths(input_paths: list[str] | None, input_dir: str | None) -> list[Path]:
    """Collect all valid image paths from inputs.

    Args:
        input_paths: List of individual image paths
        input_dir: Directory containing images

    Returns:
        Sorted list of unique valid image paths
    """
    image_paths = []

    if input_paths:
        for path_str in input_paths:
            path = Path(path_str)
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                image_paths.append(path)
            elif not path.exists():
                console.print(f"[yellow]Warning: File not found: {path}[/yellow]")
            else:
                console.print(f"[yellow]Warning: Unsupported format: {path}[/yellow]")

    if input_dir:
        dir_path = Path(input_dir)
        if dir_path.is_dir():
            image_paths.extend(iter_image_files(dir_path))
        else:
            console.print(f"[yellow]Warning: Directory not found: {dir_path}[/yellow]")

    return sorted(set(image_paths))


def run_batch_inference(config: Config, checkpoint_path: str, image_paths: list[Path]) -> None:
    """Run inference on each image and print a processing summary."""
    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold]{'=' * 70}[/bold]")
    console.print(f"[bold]Found {len(image_paths)} image(s) to process[/bold]")
    console.print(f"[bold]Output directory: {output_dir}[/bold]")
    console.print(f"[bold]{'=' * 70}[/bold]\n")

    results = []
    for idx, image_path in enumerate(image_paths, 1):
        console.print(
            f"\n[bold cyan][{idx}/{len(image_paths)}] Processing: {image_path.name}[/bold cyan]"
        )
        console.print("-" * 70)

        output_path = result_png_path(output_dir, image_path.stem)

        try:
            run_inference(config, checkpoint_path, str(image_path), str(output_path))
            results.append({"image": image_path.name, "status": "success"})
        except Exception as e:
            console.print(f"[red]Error processing {image_path.name}: {e}[/red]")
            results.append({"image": image_path.name, "status": "failed", "error": str(e)})

    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful

    console.print(f"\n[bold]{'=' * 70}[/bold]")
    console.print("[bold]PROCESSING SUMMARY[/bold]")
    console.print(f"[bold]{'=' * 70}[/bold]")
    console.print(
        f"Total: {len(results)} | [green]Success: {successful}[/green] "
        f"| [red]Failed: {failed}[/red]"
    )

    if failed > 0:
        console.print("\n[red]Failed images:[/red]")
        for r in results:
            if r["status"] == "failed":
                console.print(f"  - {r['image']}: {r['error']}")

    if successful > 0:
        console.print(f"\n[green]All outputs saved to: {output_dir}[/green]")
