"""Model and image loading utilities for inference."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from rich.console import Console

from ..models.melanomanet import MelanomaNet, create_model
from ..utils.checkpoint import load_checkpoint

console = Console()


def load_model(config: dict, checkpoint_path: str, device: torch.device) -> MelanomaNet:
    """Load model from checkpoint.

    Args:
        config: Configuration dictionary
        checkpoint_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Loaded and evaluated model
    """
    console.print(f"[bold]Loading model from {checkpoint_path}...[/bold]")
    model = create_model(config).to(device)
    load_checkpoint(Path(checkpoint_path), model, device=device)
    model.eval()
    return model


def load_image(image_path: str, img_size: int) -> tuple[Image.Image, np.ndarray]:
    """Load and preprocess image.

    Args:
        image_path: Path to input image
        img_size: Target image size

    Returns:
        Tuple of (original PIL image, resized numpy array normalized to [0,1])
    """
    console.print(f"[bold]Loading image from {image_path}...[/bold]")
    original = Image.open(image_path).convert("RGB")
    resized_np = np.array(original.resize((img_size, img_size))) / 255.0
    return original, resized_np
