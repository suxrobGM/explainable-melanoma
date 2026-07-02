"""
Concept dataset generation for FastCAV using ABCDE feature extraction.

Analyzes each ISIC 2019 training image with the ABCDE feature extractors and
categorizes it into positive/negative examples for each concept.
"""

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from ..abcde import ABCDEAnalyzer
from ..config import Config
from ..utils.console import console
from ..utils.images import iter_image_files

CONCEPTS = {
    "asymmetry": {
        "flag_column": "asymmetry_flag",
        "description": "Asymmetric vs symmetric lesions",
    },
    "irregular_border": {
        "flag_column": "border_flag",
        "description": "Irregular vs regular borders",
    },
    "multicolor": {
        "flag_column": "color_flag",
        "description": "Multi-colored vs uniform lesions",
    },
    "large_diameter": {
        "flag_column": "diameter_flag",
        "description": "Large vs small lesions",
    },
}


def analyze_image_abcde(
    image_path: Path, analyzer: ABCDEAnalyzer
) -> dict[str, Any] | None:
    """
    Analyze a single image using ABCDE criteria.

    Args:
        image_path: Path to the image
        analyzer: Configured ABCDE analyzer

    Returns:
        Dictionary with scores and flags for each criterion, or None if the
        lesion mask is too small or the analysis fails.
    """
    try:
        image = np.array(Image.open(image_path).convert("RGB"))
        result = analyzer.analyze_image(image)

        if result["details"]["lesion_area_pixels"] < 100:  # Mask too small
            return None

        return {
            "asymmetry_score": result["scores"]["asymmetry"],
            "asymmetry_flag": result["flags"]["asymmetry_flag"],
            "border_score": result["scores"]["border"],
            "border_flag": result["flags"]["border_flag"],
            "color_score": result["scores"]["color"],
            "num_colors": result["details"]["num_colors"],
            "color_flag": result["flags"]["color_flag"],
            "diameter_score": result["scores"]["diameter"],
            "diameter_flag": result["flags"]["diameter_flag"],
        }
    except Exception as e:
        console.print(
            f"[yellow]Warning: Failed to analyze {image_path.name}: {e}[/yellow]"
        )
        return None


def generate_concept_dataset(
    config: Config,
    output_dir: Path,
    max_samples_per_concept: int = 200,
    min_samples_per_class: int = 50,
) -> None:
    """
    Generate concept datasets from ISIC 2019 training data.

    Creates positive/negative example folders for each ABCDE concept:
    - asymmetry: Asymmetric vs symmetric lesions
    - irregular_border: Irregular vs regular borders
    - multicolor: Multi-colored vs uniform lesions
    - large_diameter: Large vs small lesions

    Args:
        config: Configuration
        output_dir: Output directory for concept datasets
        max_samples_per_concept: Maximum samples per positive/negative class
        min_samples_per_class: Minimum samples required
    """
    data_dir = Path(config.data.data_dir)
    train_dir = data_dir / "train"

    analyzer = ABCDEAnalyzer(
        asymmetry_threshold=config.abcde.asymmetry_threshold,
        border_threshold=config.abcde.border_threshold,
        color_threshold=config.abcde.color_threshold,
        diameter_threshold_px=config.abcde.diameter_threshold_px,
    )

    console.print("[bold cyan]ABCDE Concept Dataset Generator[/bold cyan]")
    console.print(f"Data directory: {data_dir}")
    console.print(f"Output directory: {output_dir}")
    console.print(f"Thresholds: {config.abcde}")
    console.print()

    image_paths = list(iter_image_files(train_dir))
    console.print(f"Found {len(image_paths)} training images")

    # Analyze all images
    console.print("\n[bold]Analyzing images with ABCDE criteria...[/bold]")
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing images...", total=len(image_paths))

        for image_path in image_paths:
            analysis = analyze_image_abcde(image_path, analyzer)
            if analysis:
                analysis["image_path"] = image_path
                analysis["image_id"] = image_path.stem
                results.append(analysis)
            progress.advance(task)

    console.print(f"Successfully analyzed {len(results)} images")
    results_df = pd.DataFrame(results)

    console.print("\n[bold]Creating concept datasets...[/bold]")
    for concept_name, concept_info in CONCEPTS.items():
        console.print(f"\n[cyan]{concept_name}[/cyan]: {concept_info['description']}")

        flags = results_df[concept_info["flag_column"]].astype(bool)
        positive_df = results_df[flags]
        negative_df = results_df[~flags]

        console.print(
            f"  Raw counts: {len(positive_df)} positive, {len(negative_df)} negative"
        )

        if len(positive_df) < min_samples_per_class:
            console.print(
                f"  [yellow]Warning: Not enough positive samples "
                f"({len(positive_df)} < {min_samples_per_class})[/yellow]"
            )
            continue
        if len(negative_df) < min_samples_per_class:
            console.print(
                f"  [yellow]Warning: Not enough negative samples "
                f"({len(negative_df)} < {min_samples_per_class})[/yellow]"
            )
            continue

        # Balance classes, capped at max_samples_per_concept
        n_samples = min(len(positive_df), len(negative_df), max_samples_per_concept)
        positive_samples = positive_df.sample(n=n_samples, random_state=42)
        negative_samples = negative_df.sample(n=n_samples, random_state=42)

        pos_dir = output_dir / concept_name / "positive"
        neg_dir = output_dir / concept_name / "negative"
        pos_dir.mkdir(parents=True, exist_ok=True)
        neg_dir.mkdir(parents=True, exist_ok=True)

        for _, row in positive_samples.iterrows():
            shutil.copy2(row["image_path"], pos_dir / row["image_path"].name)
        for _, row in negative_samples.iterrows():
            shutil.copy2(row["image_path"], neg_dir / row["image_path"].name)

        console.print(f"  Created: {n_samples} positive, {n_samples} negative samples")

    # Save analysis results for reference
    results_df.to_csv(output_dir / "abcde_analysis.csv", index=False)
    console.print(
        f"\n[bold green]Concept datasets created at: {output_dir}[/bold green]"
    )
    console.print(f"Analysis results saved to: {output_dir / 'abcde_analysis.csv'}")

    # Print summary
    console.print("\n[bold]Summary:[/bold]")
    for concept_name in CONCEPTS:
        concept_dir = output_dir / concept_name
        if concept_dir.exists():
            pos_count = len(list((concept_dir / "positive").glob("*")))
            neg_count = len(list((concept_dir / "negative").glob("*")))
            console.print(
                f"  {concept_name}: {pos_count} positive, {neg_count} negative"
            )
