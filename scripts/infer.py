"""
Inference script with comprehensive explainability features.

Features:
- GradCAM++ attention visualization
- ABCDE criterion analysis
- MC Dropout uncertainty quantification
- FastCAV concept-based explanations

Usage:
    pdm run infer
    python scripts/infer.py --config config.yaml --checkpoint checkpoints/best_model.pth --input image.jpg
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse

import yaml
from rich.console import Console

from melanomanet.inference import run_inference

console = Console()

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_image_paths(
    input_paths: list[str] | None, input_dir: str | None
) -> list[Path]:
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
            if path.is_file() and path.suffix.lower() in SUPPORTED_FORMATS:
                image_paths.append(path)
            elif not path.exists():
                console.print(f"[yellow]Warning: File not found: {path}[/yellow]")
            else:
                console.print(f"[yellow]Warning: Unsupported format: {path}[/yellow]")

    if input_dir:
        dir_path = Path(input_dir)
        if dir_path.is_dir():
            for ext in SUPPORTED_FORMATS:
                image_paths.extend(dir_path.glob(f"*{ext}"))
                image_paths.extend(dir_path.glob(f"*{ext.upper()}"))
        else:
            console.print(f"[yellow]Warning: Directory not found: {dir_path}[/yellow]")

    return sorted(set(image_paths))


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with comprehensive explainability features"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint"
    )
    parser.add_argument(
        "--input",
        type=str,
        action="append",
        help="Path to input image(s) - can be specified multiple times",
    )
    parser.add_argument("--input-dir", type=str, help="Directory containing images")
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.error("Either --input or --input-dir must be specified")

    # Load config and setup output directory
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    image_paths = collect_image_paths(args.input, args.input_dir)

    if not image_paths:
        console.print("[red]Error: No valid images found![/red]")
        return

    console.print(f"\n[bold]{'='*70}[/bold]")
    console.print(f"[bold]Found {len(image_paths)} image(s) to process[/bold]")
    console.print(f"[bold]Output directory: {output_dir}[/bold]")
    console.print(f"[bold]{'='*70}[/bold]\n")

    # Process images
    results = []
    for idx, image_path in enumerate(image_paths, 1):
        console.print(
            f"\n[bold cyan][{idx}/{len(image_paths)}] Processing: {image_path.name}[/bold cyan]"
        )
        console.print("-" * 70)

        output_path = output_dir / f"{image_path.stem}_result.png"

        try:
            run_inference(
                args.config, args.checkpoint, str(image_path), str(output_path)
            )
            results.append({"image": image_path.name, "status": "success"})
        except Exception as e:
            console.print(f"[red]Error processing {image_path.name}: {e}[/red]")
            results.append(
                {"image": image_path.name, "status": "failed", "error": str(e)}
            )

    # Print summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful

    console.print(f"\n[bold]{'='*70}[/bold]")
    console.print("[bold]PROCESSING SUMMARY[/bold]")
    console.print(f"[bold]{'='*70}[/bold]")
    console.print(
        f"Total: {len(results)} | [green]Success: {successful}[/green] | [red]Failed: {failed}[/red]"
    )

    if failed > 0:
        console.print("\n[red]Failed images:[/red]")
        for r in results:
            if r["status"] == "failed":
                console.print(f"  - {r['image']}: {r['error']}")

    if successful > 0:
        console.print(f"\n[green]All outputs saved to: {output_dir}[/green]")


if __name__ == "__main__":
    main()
