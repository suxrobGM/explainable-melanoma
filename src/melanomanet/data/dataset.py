from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class MelanomaDataset(Dataset):
    """
    Custom dataset for skin lesion classification from dermoscopic images.

    Supports ISIC2019 dataset with configurable preprocessing
    and augmentation pipelines.

    Args:
        image_dir: Directory containing dermoscopic images
        labels_df: DataFrame with columns ['image_id', 'target']
                  where target: class index (0-8 for ISIC 2019 9 classes)
        transform: Optional torchvision transforms to apply

    Returns:
        Tuple of (image_tensor, label, image_id)
    """

    def __init__(
        self,
        image_dir: str | Path,
        labels_df: pd.DataFrame,
        transform: Callable | None = None,
    ):
        self.image_dir = Path(image_dir)
        self.labels_df = labels_df.reset_index(drop=True)
        self.transform = transform

        # Validate that images exist
        self._validate_dataset()

    def _validate_dataset(self) -> None:
        """Check that image directory exists and contains images."""
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Check first 10 images exist
        for idx in range(min(10, len(self.labels_df))):
            image_id = self.labels_df.iloc[idx]["image_id"]
            image_path = self._get_image_path(image_id)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

    def _get_image_path(self, image_id: str) -> Path:
        """
        Get full path to image file.
        Handles different extensions (.jpg, .png, .jpeg)
        """
        for ext in [".jpg", ".jpeg", ".png"]:
            path = self.image_dir / f"{image_id}{ext}"
            if path.exists():
                return path
        raise FileNotFoundError(f"Image {image_id} not found with any extension")

    def __len__(self) -> int:
        return len(self.labels_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        """
        Load and return a single sample.

        Returns:
            image: Transformed image tensor (C, H, W)
            label: Multi-class label (0-8 for ISIC 2019)
            image_id: String identifier for the image
        """
        # Get image metadata
        row = self.labels_df.iloc[idx]
        image_id = row["image_id"]
        label = int(row["target"])

        # Load image
        image_path = self._get_image_path(image_id)
        image = Image.open(image_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label, image_id


def get_class_weights(labels_df: pd.DataFrame, num_classes: int = 9) -> torch.Tensor:
    """
    Calculate class weights for imbalanced dataset.
    Uses inverse frequency weighting.

    Args:
        labels_df: DataFrame with 'target' column
        num_classes: Total number of classes (default: 9 for ISIC 2019)

    Returns:
        Tensor of class weights for all classes
        Classes with 0 samples get weight 0.0 (won't affect loss)
    """
    # Initialize weights for all classes
    weights = torch.zeros(num_classes, dtype=torch.float32)

    # Get counts for all classes (including those with 0 samples)
    total = len(labels_df)

    for class_idx in range(num_classes):
        count = (labels_df["target"] == class_idx).sum()
        if count > 0:
            # Inverse frequency weighting for classes with samples
            weights[class_idx] = total / (num_classes * count)
        else:
            # Classes with no samples get 0 weight
            weights[class_idx] = 0.0

    return weights
