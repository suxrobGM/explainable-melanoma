# Author: Sukhrobbek Ilyosbekov
# Date: 2025-12-09

"""Model and image loading utilities for inference."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..config import Config
from ..models.melanomanet import MelanomaNet, create_model
from ..utils.checkpoint import load_checkpoint
from ..utils.console import console


def load_model(config: Config, checkpoint_path: str, device: torch.device) -> MelanomaNet:
    """Load model from checkpoint.

    Args:
        config: Configuration
        checkpoint_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Loaded and evaluated model
    """
    console.print(f"[bold]Loading model from {checkpoint_path}...[/bold]")
    model = create_model(config.model, config.data.num_classes).to(device)
    checkpoint = load_checkpoint(Path(checkpoint_path), model, device=device)

    epoch = checkpoint.get("epoch")
    f1 = checkpoint.get("metrics", {}).get("f1")
    if epoch is not None:
        console.print(f"[green]Loaded checkpoint from epoch {epoch}[/green]")
    if f1 is not None:
        console.print(f"[green]Model F1 score: {f1:.4f}[/green]")

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
