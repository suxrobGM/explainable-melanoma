# Author: Sukhrobbek Ilyosbekov
# Date: 2025-12-09

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from ..config import AugmentationConfig

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(
    image_size: int, augmentation: AugmentationConfig
) -> T.Compose:
    """
    Get training transforms with augmentation.

    Augmentation strategy:
    - Geometric: rotation, flips, affine
    - Color: brightness, contrast, saturation adjustments
    - Normalization: ImageNet statistics

    Args:
        image_size: Target square image size
        augmentation: Augmentation parameters

    Returns:
        Composed transforms for training
    """
    transforms = [
        # Resize to square
        T.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        # Geometric augmentation
        T.RandomHorizontalFlip(p=augmentation.horizontal_flip),
        T.RandomVerticalFlip(p=augmentation.vertical_flip),
        T.RandomRotation(
            degrees=augmentation.rotation_degrees,
            interpolation=InterpolationMode.BILINEAR,
        ),
    ]

    # Optional affine transformation
    if augmentation.random_affine:
        transforms.append(
            T.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                interpolation=InterpolationMode.BILINEAR,
            )
        )

    # Color augmentation
    jitter = augmentation.color_jitter
    transforms.append(
        T.ColorJitter(
            brightness=jitter.brightness,
            contrast=jitter.contrast,
            saturation=jitter.saturation,
            hue=jitter.hue,
        )
    )

    # Normalize and convert to tensor
    transforms.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return T.Compose(transforms)


def get_val_transforms(image_size: int) -> T.Compose:
    """
    Get validation/test transforms without augmentation.

    Args:
        image_size: Target square image size

    Returns:
        Composed transforms for validation/testing
    """
    return T.Compose(
        [
            T.Resize(
                (image_size, image_size), interpolation=InterpolationMode.BILINEAR
            ),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
