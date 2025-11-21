"""
Inference script with GradCAM++ visualization and ABCDE analysis.

Args:
    --config: Path to configuration YAML file
    --checkpoint: Path to model checkpoint file
    --input: Path to input dermoscopic image
    --output: Path to save visualization

Usage:
    python scripts/infer.py --checkpoint checkpoints/best_model.pth \
                            --input image.jpg \
                            --output result.png \
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
from melanomanet.models.melanomanet import create_model
from melanomanet.utils.checkpoint import load_checkpoint
from melanomanet.utils.gradcam import MelanomaGradCAM, denormalize_image


def run_inference(
    config_path: str,
    checkpoint_path: str,
    image_path: str,
    output_path: str,
) -> None:
    """
    Run inference on a single image with attention visualization and ABCDE analysis.

    Args:
        config_path: Path to config file
        checkpoint_path: Path to model checkpoint
        image_path: Path to input dermoscopic image
        output_path: Path to save visualization
    """
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Read ABCDE enable flag from config
    enable_abcde = config.get("abcde", {}).get("enable", True)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = create_model(config).to(device)
    load_checkpoint(Path(checkpoint_path), model, device=device)
    model.eval()

    # Load and preprocess image
    print(f"Loading image from {image_path}...")
    original_image = Image.open(image_path).convert("RGB")

    # Resize for visualization
    original_image_np = np.array(original_image.resize((224, 224))) / 255.0

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
            alignment_scores = analyzer.align_with_gradcam(
                abcde_result, attention_map, image_uint8
            )

        # Print ABCDE report
        report = create_abcde_report(abcde_result, alignment_scores)
        print("\n" + report)

    # Create comprehensive visualization
    if enable_abcde and abcde_result:
        # Create larger figure with ABCDE visualizations
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

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
            "⚠ PRESENT" if abcde_result["flags"]["asymmetry_flag"] else "✓ Absent"
        )
        ax5.set_title(
            f"A - Asymmetry: {asym_status}\nScore: {abcde_result['scores']['asymmetry']:.3f}",
            fontsize=10,
        )
        ax5.axis("off")

        ax6 = fig.add_subplot(gs[1, 1])
        ax6.imshow(viz["border"])
        border_status = (
            "⚠ IRREGULAR" if abcde_result["flags"]["border_flag"] else "✓ Regular"
        )
        ax6.set_title(
            f"B - Border: {border_status}\nScore: {abcde_result['scores']['border']:.3f}",
            fontsize=10,
        )
        ax6.axis("off")

        ax7 = fig.add_subplot(gs[1, 2])
        ax7.imshow(viz["color"])
        color_status = (
            "⚠ VARIED" if abcde_result["flags"]["color_flag"] else "✓ Uniform"
        )
        ax7.set_title(
            f"C - Color: {color_status}\nColors: {abcde_result['details']['num_colors']}",
            fontsize=10,
        )
        ax7.axis("off")

        ax8 = fig.add_subplot(gs[1, 3])
        ax8.imshow(viz["diameter"])
        diam_status = "⚠ LARGE" if abcde_result["flags"]["diameter_flag"] else "✓ Small"
        ax8.set_title(
            f"D - Diameter: {diam_status}\n{abcde_result['scores']['diameter']:.1f}px",
            fontsize=10,
        )
        ax8.axis("off")

        # Row 3: Alignment metrics and summary
        ax9 = fig.add_subplot(gs[2, :2])
        ax9.axis("off")
        summary_text = "ABCDE CRITERION SUMMARY\n" + "=" * 40 + "\n"
        for criterion, flag_key in [
            ("Asymmetry", "asymmetry_flag"),
            ("Border", "border_flag"),
            ("Color", "color_flag"),
            ("Diameter", "diameter_flag"),
        ]:
            status = "⚠ CONCERN" if abcde_result["flags"][flag_key] else "✓ OK"
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

        if alignment_scores:
            ax10 = fig.add_subplot(gs[2, 2:])
            ax10.axis("off")
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
            "MelanomaNet: Explainable Melanoma Detection with ABCDE Analysis",
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

    # Save ABCDE report to text file
    if enable_abcde and abcde_result:
        report_path = Path(output_path).with_suffix(".txt")
        with open(report_path, "w") as f:
            f.write(create_abcde_report(abcde_result, alignment_scores))
        print(f"ABCDE report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with GradCAM++ and ABCDE analysis"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint"
    )
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, required=True, help="Path to save output")
    args = parser.parse_args()

    run_inference(args.config, args.checkpoint, args.input, args.output)


if __name__ == "__main__":
    main()
