"""
Data preprocessing and augmentation transforms for dermoscopic images.
"""

from typing import Any

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


def get_train_transforms(config: dict[str, Any]) -> T.Compose:
    """
    Get training transforms with augmentation.

    Augmentation strategy:
    - Geometric: rotation, flips, affine
    - Color: brightness, contrast, saturation adjustments
    - Normalization: ImageNet statistics

    Args:
        config: Configuration dictionary with augmentation parameters

    Returns:
        Composed transforms for training
    """
    image_size = config["data"]["image_size"]
    aug_config = config["training"]["augmentation"]

    transforms = [
        # Resize to square
        T.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        # Geometric augmentation
        T.RandomHorizontalFlip(p=aug_config["horizontal_flip"]),
        T.RandomVerticalFlip(p=aug_config["vertical_flip"]),
        T.RandomRotation(
            degrees=aug_config["rotation_degrees"],
            interpolation=InterpolationMode.BILINEAR,
        ),
    ]

    # Optional affine transformation
    if aug_config.get("random_affine", False):
        transforms.append(
            T.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                interpolation=InterpolationMode.BILINEAR,
            )
        )

    # Color augmentation
    color_jitter = aug_config.get("color_jitter", {})
    if color_jitter:
        transforms.append(
            T.ColorJitter(
                brightness=color_jitter.get("brightness", 0.2),
                contrast=color_jitter.get("contrast", 0.2),
                saturation=color_jitter.get("saturation", 0.2),
                hue=color_jitter.get("hue", 0.1),
            )
        )

    # Normalize and convert to tensor
    transforms.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return T.Compose(transforms)


def get_val_transforms(config: dict[str, Any]) -> T.Compose:
    """
    Get validation/test transforms without augmentation.

    Args:
        config: Configuration dictionary

    Returns:
        Composed transforms for validation/testing
    """
    image_size = config["data"]["image_size"]

    return T.Compose(
        [
            T.Resize(
                (image_size, image_size), interpolation=InterpolationMode.BILINEAR
            ),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
