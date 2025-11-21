"""
Inference script with GradCAM++ visualization.

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

from melanomanet.data.transforms import get_val_transforms
from melanomanet.models.melanomanet import create_model
from melanomanet.utils.checkpoint import load_checkpoint
from melanomanet.utils.gradcam import MelanomaGradCAM, denormalize_image


def run_inference(
    config_path: str, checkpoint_path: str, image_path: str, output_path: str
) -> None:
    """
    Run inference on a single image with attention visualization.

    Args:
        config_path: Path to config file
        checkpoint_path: Path to model checkpoint
        image_path: Path to input dermoscopic image
        output_path: Path to save visualization
    """
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

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
    visualization = gradcam.visualize_attention(
        image_tensor, original_image_np, target_class=pred_class
    )

    # Create figure with original and attention
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(original_image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Attention heatmap
    attention_map = gradcam.generate_attention_map(image_tensor, pred_class)
    axes[1].imshow(attention_map, cmap="jet")
    axes[1].set_title("GradCAM++ Heatmap")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(visualization)
    axes[2].set_title(
        f"Prediction: {class_names[pred_class]}\n" f"Confidence: {confidence:.2%}"
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nResult saved to {output_path}")

    # Also save just the overlay
    overlay_path = Path(output_path).with_stem(f"{Path(output_path).stem}_overlay")
    plt.imsave(overlay_path, visualization)
    print(f"Overlay saved to {overlay_path}")


def main():
    parser = argparse.ArgumentParser(description="Run inference with GradCAM++")
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
