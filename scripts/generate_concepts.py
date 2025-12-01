"""
Generate concept datasets for FastCAV analysis using ABCDE feature extraction.

This script automatically creates concept datasets from the ISIC 2019 training set
by analyzing each image with the ABCDE feature extractors and categorizing them
into positive/negative examples for each concept.

Usage:
    python scripts/generate_concepts.py --config config.yaml --output data/concepts
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import random
import shutil
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from melanomanet.abcde.features import (
    analyze_asymmetry,
    analyze_border,
    analyze_color,
    analyze_diameter,
)
from melanomanet.abcde.segmentation import extract_lesion_mask

console = Console()


def load_ground_truth(data_dir: Path) -> pd.DataFrame:
    """Load and process ground truth labels."""
    gt_path = data_dir / "ISIC_2019_Training_GroundTruth.csv"
    df = pd.read_csv(gt_path)

    # Convert one-hot to class index
    class_columns = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
    df["target"] = df[class_columns].values.argmax(axis=1)
    df = df.rename(columns={"image": "image_id"})

    return df[["image_id", "target"]]


def analyze_image_abcde(image_path: Path, thresholds: dict) -> dict[str, Any] | None:
    """
    Analyze a single image using ABCDE criteria.

    Args:
        image_path: Path to the image
        thresholds: Dictionary of thresholds for each criterion

    Returns:
        Dictionary with scores and flags for each criterion
    """
    try:
        # Load image
        image = np.array(Image.open(image_path).convert("RGB"))

        # Extract lesion mask
        mask = extract_lesion_mask(image)

        if mask.sum() < 100:  # Skip if mask is too small
            return None

        # Analyze each criterion
        asymmetry_score, _ = analyze_asymmetry(image, mask)
        border_score, _ = analyze_border(mask, image)
        color_score, num_colors, _ = analyze_color(image, mask)
        diameter_score, _ = analyze_diameter(mask, image)

        return {
            "asymmetry_score": asymmetry_score,
            "asymmetry_flag": asymmetry_score > thresholds["asymmetry"],
            "border_score": border_score,
            "border_flag": border_score > thresholds["border"],
            "color_score": color_score,
            "num_colors": num_colors,
            "color_flag": num_colors >= thresholds["color"],
            "diameter_score": diameter_score,
            "diameter_flag": diameter_score > thresholds["diameter"],
        }
    except Exception as e:
        console.print(
            f"[yellow]Warning: Failed to analyze {image_path.name}: {e}[/yellow]"
        )
        return None


def generate_concept_dataset(
    config: dict,
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
        config: Configuration dictionary
        output_dir: Output directory for concept datasets
        max_samples_per_concept: Maximum samples per positive/negative class
        min_samples_per_class: Minimum samples required
    """
    data_dir = Path(config["data"]["data_dir"])
    train_dir = data_dir / "train"
    output_dir = Path(output_dir)

    # ABCDE thresholds from config
    abcde_config = config.get("abcde", {})
    thresholds = {
        "asymmetry": abcde_config.get("asymmetry_threshold", 0.3),
        "border": abcde_config.get("border_threshold", 0.4),
        "color": abcde_config.get("color_threshold", 3),
        "diameter": abcde_config.get("diameter_threshold_px", 114),
    }

    console.print("[bold cyan]ABCDE Concept Dataset Generator[/bold cyan]")
    console.print(f"Data directory: {data_dir}")
    console.print(f"Output directory: {output_dir}")
    console.print(f"Thresholds: {thresholds}")
    console.print()

    # Load ground truth
    console.print("[bold]Loading ground truth...[/bold]")
    gt_df = load_ground_truth(data_dir)

    # Get list of training images
    image_extensions = {".jpg", ".jpeg", ".png"}
    image_paths = [
        p for p in train_dir.iterdir() if p.suffix.lower() in image_extensions
    ]

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
            analysis = analyze_image_abcde(image_path, thresholds)
            if analysis:
                analysis["image_path"] = image_path
                analysis["image_id"] = image_path.stem
                results.append(analysis)
            progress.advance(task)

    console.print(f"Successfully analyzed {len(results)} images")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Create concept directories
    concepts = {
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

    console.print("\n[bold]Creating concept datasets...[/bold]")

    for concept_name, concept_info in concepts.items():
        console.print(f"\n[cyan]{concept_name}[/cyan]: {concept_info['description']}")

        flag_col = concept_info["flag_column"]

        # Get positive and negative samples
        positive_df = results_df[results_df[flag_col] == True]
        negative_df = results_df[results_df[flag_col] == False]

        console.print(
            f"  Raw counts: {len(positive_df)} positive, {len(negative_df)} negative"
        )

        # Check minimum samples
        if len(positive_df) < min_samples_per_class:
            console.print(
                f"  [yellow]Warning: Not enough positive samples ({len(positive_df)} < {min_samples_per_class})[/yellow]"
            )
            continue
        if len(negative_df) < min_samples_per_class:
            console.print(
                f"  [yellow]Warning: Not enough negative samples ({len(negative_df)} < {min_samples_per_class})[/yellow]"
            )
            continue

        # Sample to max_samples_per_concept
        n_positive = min(len(positive_df), max_samples_per_concept)
        n_negative = min(len(negative_df), max_samples_per_concept)

        # Balance classes
        n_samples = min(n_positive, n_negative)

        positive_samples = positive_df.sample(n=n_samples, random_state=42)
        negative_samples = negative_df.sample(n=n_samples, random_state=42)

        # Create directories
        concept_dir = output_dir / concept_name
        pos_dir = concept_dir / "positive"
        neg_dir = concept_dir / "negative"

        pos_dir.mkdir(parents=True, exist_ok=True)
        neg_dir.mkdir(parents=True, exist_ok=True)

        # Copy images
        for _, row in positive_samples.iterrows():
            src = row["image_path"]
            dst = pos_dir / src.name
            shutil.copy2(src, dst)

        for _, row in negative_samples.iterrows():
            src = row["image_path"]
            dst = neg_dir / src.name
            shutil.copy2(src, dst)

        console.print(f"  Created: {n_samples} positive, {n_samples} negative samples")

    # Save analysis results for reference
    results_df.to_csv(output_dir / "abcde_analysis.csv", index=False)
    console.print(
        f"\n[bold green]Concept datasets created at: {output_dir}[/bold green]"
    )
    console.print(f"Analysis results saved to: {output_dir / 'abcde_analysis.csv'}")

    # Print summary
    console.print("\n[bold]Summary:[/bold]")
    for concept_name in concepts.keys():
        concept_dir = output_dir / concept_name
        if concept_dir.exists():
            pos_count = len(list((concept_dir / "positive").glob("*")))
            neg_count = len(list((concept_dir / "negative").glob("*")))
            console.print(
                f"  {concept_name}: {pos_count} positive, {neg_count} negative"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Generate concept datasets for FastCAV analysis"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/concepts",
        help="Output directory for concept datasets",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Maximum samples per concept class",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=50,
        help="Minimum samples required per class",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    generate_concept_dataset(
        config=config,
        output_dir=Path(args.output),
        max_samples_per_concept=args.max_samples,
        min_samples_per_class=args.min_samples,
    )


if __name__ == "__main__":
    main()
