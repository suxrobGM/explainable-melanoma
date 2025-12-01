"""
Inference script with comprehensive explainability features.

Features:
- GradCAM++ attention visualization
- ABCDE criterion analysis
- MC Dropout uncertainty quantification
- FastCAV concept-based explanations

Args:
    --config: Path to configuration YAML file
    --checkpoint: Path to model checkpoint file
    --input: Path(s) to input dermoscopic image(s) - can specify multiple times
    --input-dir: Path to directory containing images to process

Usage:
    # Single image
    python scripts/infer.py --checkpoint checkpoints/best_model.pth \
                            --input image.jpg \
                            --config config.yaml

    # Multiple images
    python scripts/infer.py --checkpoint checkpoints/best_model.pth \
                            --input image1.jpg --input image2.jpg \
                            --config config.yaml

    # Folder of images
    python scripts/infer.py --checkpoint checkpoints/best_model.pth \
                            --input-dir ./images \
                            --config config.yaml
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

from melanomanet.abcde import ABCDEAnalyzer, create_abcde_report
from melanomanet.data.transforms import get_val_transforms
from melanomanet.explainability import FastCAV, create_fastcav_report
from melanomanet.models.melanomanet import create_model
from melanomanet.uncertainty import MCDropoutEstimator, get_uncertainty_interpretation
from melanomanet.utils.checkpoint import load_checkpoint
from melanomanet.utils.gradcam import MelanomaGradCAM

# Supported image formats
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_image_paths(input_paths: list[str], input_dir: str) -> list[Path]:
    """
    Collect all valid image paths from individual files and/or directory.

    Args:
        input_paths: List of individual image file paths
        input_dir: Directory containing images

    Returns:
        List of valid image file paths
    """
    image_paths = []

    # Collect from individual input paths
    if input_paths:
        for path_str in input_paths:
            path = Path(path_str)
            if path.is_file() and path.suffix.lower() in SUPPORTED_FORMATS:
                image_paths.append(path)
            elif not path.exists():
                print(f"Warning: File not found: {path}")
            else:
                print(f"Warning: Unsupported file format: {path}")

    # Collect from input directory
    if input_dir:
        dir_path = Path(input_dir)
        if dir_path.is_dir():
            for ext in SUPPORTED_FORMATS:
                image_paths.extend(dir_path.glob(f"*{ext}"))
                image_paths.extend(dir_path.glob(f"*{ext.upper()}"))
        else:
            print(f"Warning: Directory not found: {dir_path}")

    # Remove duplicates and sort
    image_paths = sorted(set(image_paths))

    return image_paths


def run_inference(
    config_path: str,
    checkpoint_path: str,
    image_path: str,
    output_path: str,
) -> None:
    """
    Run inference on a single image with comprehensive explainability.

    Includes:
    - Model prediction with confidence
    - GradCAM++ attention visualization
    - ABCDE criterion analysis
    - MC Dropout uncertainty estimation
    - FastCAV concept-based explanations

    Args:
        config_path: Path to config file
        checkpoint_path: Path to model checkpoint
        image_path: Path to input dermoscopic image
        output_path: Path to save visualization
    """
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Read feature flags from config
    enable_abcde = config.get("abcde", {}).get("enable", True)
    enable_uncertainty = config.get("uncertainty", {}).get("enable", True)
    enable_fastcav = config.get("fastcav", {}).get("enable", True)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = create_model(config).to(device)
    load_checkpoint(Path(checkpoint_path), model, device=device)
    model.eval()

    # Load and preprocess image
    print(f"Loading image from {image_path}...")
    original_image = Image.open(image_path).convert("RGB")

    # Resize for visualization using configured image size
    img_size = config["data"]["image_size"]
    original_image_np = np.array(original_image.resize((img_size, img_size))) / 255.0

    # Transform for model
    transform = get_val_transforms(config)
    image_tensor = transform(original_image).unsqueeze(0)

    # Initialize GradCAM
    gradcam = MelanomaGradCAM(model, device=device)

    # Get prediction
    pred_class, confidence = gradcam.get_prediction(image_tensor)
    class_names = config["data"]["class_names"]

    print(f"\nPrediction: {class_names[pred_class]}")
    print(f"Confidence: {confidence:.4f}")

    # Generate attention visualization
    print("\nGenerating attention map...")
    attention_map = gradcam.generate_attention_map(image_tensor, pred_class)
    visualization = gradcam.visualize_attention(
        image_tensor, original_image_np, target_class=pred_class
    )

    # MC Dropout Uncertainty Estimation
    uncertainty_result = None
    if enable_uncertainty:
        print("\nEstimating prediction uncertainty (MC Dropout)...")
        n_samples = config.get("uncertainty", {}).get("n_samples", 10)
        threshold = config.get("uncertainty", {}).get("uncertainty_threshold", 0.5)

        estimator = MCDropoutEstimator(
            model=model,
            n_samples=n_samples,
            uncertainty_threshold=threshold,
            device=device,
        )
        uncertainty_result = estimator.estimate(image_tensor)
        print(get_uncertainty_interpretation(uncertainty_result))

    # FastCAV Concept Analysis
    fastcav_result = None
    if enable_fastcav:
        print("\nAnalyzing concept importance (FastCAV)...")
        concepts_dir = Path(
            config.get("fastcav", {}).get("concepts_dir", "./data/concepts")
        )
        cavs_path = Path(
            config.get("fastcav", {}).get("cavs_path", "./checkpoints/cavs.pth")
        )

        if cavs_path.exists():
            fastcav = FastCAV(
                model=model,
                concepts_dir=concepts_dir,
                device=device,
            )
            fastcav.load_cavs(cavs_path)

            fastcav_result = fastcav.analyze_image(
                image_tensor,
                target_class=pred_class,
                class_name=class_names[pred_class],
            )
            print(create_fastcav_report(fastcav_result))
        else:
            print(f"Warning: CAVs file not found at {cavs_path}. Run training first.")

    # ABCDE Analysis
    abcde_result = None
    alignment_scores = None

    if enable_abcde:
        print("\nPerforming ABCDE criterion analysis...")
        abcde_config = config.get("abcde", {})
        analyzer = ABCDEAnalyzer(
            asymmetry_threshold=abcde_config.get("asymmetry_threshold", 0.3),
            border_threshold=abcde_config.get("border_threshold", 0.4),
            color_threshold=abcde_config.get("color_threshold", 3),
            diameter_threshold_px=abcde_config.get("diameter_threshold_px", 50),
        )

        # Convert image to uint8 for ABCDE analysis
        image_uint8 = (original_image_np * 255).astype(np.uint8)
        abcde_result = analyzer.analyze_image(image_uint8, return_visualizations=True)

        # Analyze GradCAM-ABCDE alignment
        if abcde_config.get("enable_alignment_analysis", True):
            alignment_scores = analyzer.align_with_gradcam(abcde_result, attention_map)

        # Print ABCDE report
        report = create_abcde_report(abcde_result, alignment_scores)
        print("\n" + report)

    # Create comprehensive visualization
    if enable_abcde and abcde_result:
        # Determine figure size based on features enabled
        n_rows = 3 if (enable_uncertainty or enable_fastcav) else 3
        fig = plt.figure(figsize=(20, 4 * n_rows))
        gs = fig.add_gridspec(n_rows, 4, hspace=0.3, wspace=0.3)

        # Row 1: Main prediction and GradCAM
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(original_image_np)
        ax1.set_title("Original Image", fontsize=12, fontweight="bold")
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(attention_map, cmap="jet")
        ax2.set_title("GradCAM++ Heatmap", fontsize=12, fontweight="bold")
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(visualization)
        ax3.set_title(
            f"Prediction: {class_names[pred_class]}\nConfidence: {confidence:.2%}",
            fontsize=12,
            fontweight="bold",
        )
        ax3.axis("off")

        # Risk assessment
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.axis("off")
        risk_color = {
            "Low": "green",
            "Medium": "orange",
            "High": "red",
        }.get(abcde_result["overall_risk"], "gray")
        ax4.text(
            0.5,
            0.5,
            f"ABCDE Risk:\n{abcde_result['overall_risk']}",
            fontsize=16,
            fontweight="bold",
            ha="center",
            va="center",
            color=risk_color,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor=risk_color,
                linewidth=3,
            ),
        )

        # Row 2: ABCDE Visualizations
        viz = abcde_result["visualizations"]

        ax5 = fig.add_subplot(gs[1, 0])
        ax5.imshow(viz["asymmetry"])
        asym_status = (
            "[!] PRESENT" if abcde_result["flags"]["asymmetry_flag"] else "[OK] Absent"
        )
        ax5.set_title(
            f"A - Asymmetry: {asym_status}\nScore: {abcde_result['scores']['asymmetry']:.3f}",
            fontsize=10,
        )
        ax5.axis("off")

        ax6 = fig.add_subplot(gs[1, 1])
        ax6.imshow(viz["border"])
        border_status = (
            "[!] IRREGULAR" if abcde_result["flags"]["border_flag"] else "[OK] Regular"
        )
        ax6.set_title(
            f"B - Border: {border_status}\nScore: {abcde_result['scores']['border']:.3f}",
            fontsize=10,
        )
        ax6.axis("off")

        ax7 = fig.add_subplot(gs[1, 2])
        ax7.imshow(viz["color"])
        color_status = (
            "[!] VARIED" if abcde_result["flags"]["color_flag"] else "[OK] Uniform"
        )
        ax7.set_title(
            f"C - Color: {color_status}\nColors: {abcde_result['details']['num_colors']}",
            fontsize=10,
        )
        ax7.axis("off")

        ax8 = fig.add_subplot(gs[1, 3])
        ax8.imshow(viz["diameter"])
        diam_status = (
            "[!] LARGE" if abcde_result["flags"]["diameter_flag"] else "[OK] Small"
        )
        ax8.set_title(
            f"D - Diameter: {diam_status}\n{abcde_result['scores']['diameter']:.1f}px",
            fontsize=10,
        )
        ax8.axis("off")

        # Row 3: Uncertainty and Concept Analysis
        ax9 = fig.add_subplot(gs[2, :2])
        ax9.axis("off")

        # Uncertainty panel
        if uncertainty_result:
            reliability = "RELIABLE" if uncertainty_result.is_reliable else "UNCERTAIN"
            unc_color = (
                "lightgreen" if uncertainty_result.is_reliable else "lightyellow"
            )
            unc_text = "UNCERTAINTY ANALYSIS\n" + "=" * 40 + "\n"
            unc_text += f"Predictive: {uncertainty_result.predictive_uncertainty:.3f}"
            unc_text += " (total model uncertainty)\n"
            unc_text += f"Epistemic:  {uncertainty_result.epistemic_uncertainty:.3f}"
            unc_text += " (model knowledge gaps)\n"
            unc_text += f"Aleatoric:  {uncertainty_result.aleatoric_uncertainty:.3f}"
            unc_text += " (inherent data noise)\n"
            unc_text += "-" * 40 + "\n"
            unc_text += f"Reliability: {reliability}\n"
            if uncertainty_result.is_reliable:
                unc_text += "(Low uncertainty = confident prediction)"
            else:
                unc_text += "(High uncertainty = review recommended)"
            ax9.text(
                0.1,
                0.5,
                unc_text,
                fontsize=10,
                fontfamily="monospace",
                va="center",
                bbox=dict(boxstyle="round,pad=1", facecolor=unc_color, alpha=0.7),
            )
        else:
            summary_text = "ABCDE CRITERION SUMMARY\n" + "=" * 40 + "\n"
            for criterion, flag_key in [
                ("Asymmetry", "asymmetry_flag"),
                ("Border", "border_flag"),
                ("Color", "color_flag"),
                ("Diameter", "diameter_flag"),
            ]:
                status = "[!] CONCERN" if abcde_result["flags"][flag_key] else "[OK] OK"
                summary_text += f"{criterion}: {status}\n"

            ax9.text(
                0.1,
                0.5,
                summary_text,
                fontsize=11,
                fontfamily="monospace",
                va="center",
                bbox=dict(boxstyle="round,pad=1", facecolor="lightgray", alpha=0.5),
            )

        # Concept importance panel
        ax10 = fig.add_subplot(gs[2, 2:])
        ax10.axis("off")

        if fastcav_result and fastcav_result.concept_scores:
            concept_text = "CONCEPT IMPORTANCE (FastCAV)\n" + "=" * 40 + "\n"
            sorted_concepts = sorted(
                fastcav_result.concept_scores.values(),
                key=lambda x: abs(x.tcav_score),
                reverse=True,
            )
            for cs in sorted_concepts[:4]:
                direction = "+" if cs.tcav_score > 0 else "-"
                concept_text += (
                    f"{cs.concept_name}: {direction}{abs(cs.tcav_score):.2f}\n"
                )
            concept_text += "-" * 40 + "\n"
            concept_text += "+ supports prediction\n"
            concept_text += "- opposes prediction\n"
            concept_text += "(higher magnitude = stronger influence)"
            ax10.text(
                0.1,
                0.5,
                concept_text,
                fontsize=10,
                fontfamily="monospace",
                va="center",
                bbox=dict(boxstyle="round,pad=1", facecolor="lightblue", alpha=0.5),
            )
        elif fastcav_result and not fastcav_result.concept_scores:
            concept_text = "CONCEPT IMPORTANCE (FastCAV)\n" + "=" * 40 + "\n"
            concept_text += "No concept scores available.\n"
            concept_text += "Run 'pdm run train-fastcav' to train CAVs.\n"
            ax10.text(
                0.1,
                0.5,
                concept_text,
                fontsize=11,
                fontfamily="monospace",
                va="center",
                bbox=dict(boxstyle="round,pad=1", facecolor="lightyellow", alpha=0.5),
            )
        elif alignment_scores:
            alignment_text = "GradCAM ALIGNMENT\n" + "=" * 40 + "\n"
            alignment_text += (
                f"Border Alignment: {alignment_scores['border_alignment']:.3f}\n"
            )
            alignment_text += (
                f"Overall Attention: {alignment_scores['overall_alignment']:.3f}\n"
            )
            alignment_text += f"Mean Lesion Attention: {alignment_scores['mean_lesion_attention']:.3f}\n"
            ax10.text(
                0.1,
                0.5,
                alignment_text,
                fontsize=11,
                fontfamily="monospace",
                va="center",
                bbox=dict(boxstyle="round,pad=1", facecolor="lightblue", alpha=0.5),
            )

        plt.suptitle(
            "MelanomaNet: Explainable Melanoma Detection with Uncertainty & Concept Analysis",
            fontsize=14,
            fontweight="bold",
        )

    else:
        # Simple visualization without ABCDE
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original_image_np)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(attention_map, cmap="jet")
        axes[1].set_title("GradCAM++ Heatmap")
        axes[1].axis("off")

        axes[2].imshow(visualization)
        axes[2].set_title(
            f"Prediction: {class_names[pred_class]}\nConfidence: {confidence:.2%}"
        )
        axes[2].axis("off")

        plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nVisualization saved to {output_path}")

    # Save comprehensive report to text file
    report_path = Path(output_path).with_suffix(".txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("MELANOMANET COMPREHENSIVE ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Image: {image_path}\n")
        f.write(f"Prediction: {class_names[pred_class]}\n")
        f.write(f"Confidence: {confidence:.4f}\n\n")

        if uncertainty_result:
            f.write("UNCERTAINTY ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(
                f"Predictive Uncertainty: {uncertainty_result.predictive_uncertainty:.4f} "
                "(total model uncertainty)\n"
            )
            f.write(
                f"Epistemic Uncertainty:  {uncertainty_result.epistemic_uncertainty:.4f} "
                "(model knowledge gaps - can improve with more data)\n"
            )
            f.write(
                f"Aleatoric Uncertainty:  {uncertainty_result.aleatoric_uncertainty:.4f} "
                "(inherent data noise - cannot be reduced)\n"
            )
            f.write(
                f"Reliability: {'RELIABLE' if uncertainty_result.is_reliable else 'UNCERTAIN'}\n\n"
            )

        if enable_abcde and abcde_result:
            f.write(create_abcde_report(abcde_result, alignment_scores))
            f.write("\n")

        if fastcav_result:
            f.write(create_fastcav_report(fastcav_result))

    print(f"Report saved to {report_path}")


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
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing images to process",
    )
    args = parser.parse_args()

    # Validate inputs
    if not args.input and not args.input_dir:
        parser.error("Either --input or --input-dir must be specified")

    # Load config to get output directory
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Create output directory if it doesn't exist
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all image paths
    image_paths = collect_image_paths(args.input, args.input_dir)

    if not image_paths:
        print("Error: No valid images found!")
        return

    # Process all images
    print(f"\n{'='*70}")
    print(f"Found {len(image_paths)} image(s) to process")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")

    results = []
    for idx, image_path in enumerate(image_paths, 1):
        print(f"\n[{idx}/{len(image_paths)}] Processing: {image_path.name}")
        print("-" * 70)

        # Generate output path with unique name
        output_name = f"{image_path.stem}_result.png"
        output_path = output_dir / output_name

        try:
            run_inference(
                args.config, args.checkpoint, str(image_path), str(output_path)
            )
            results.append(
                {"image": image_path.name, "status": "success", "output": output_path}
            )
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            results.append(
                {"image": image_path.name, "status": "failed", "error": str(e)}
            )

    # Print summary
    print(f"\n{'='*70}")
    print("PROCESSING SUMMARY")
    print(f"{'='*70}")
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")

    print(f"Total images: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed images:")
        for r in results:
            if r["status"] == "failed":
                print(f"  - {r['image']}: {r['error']}")

    if successful > 0:
        print(f"\nAll outputs saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
