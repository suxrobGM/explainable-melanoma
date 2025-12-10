# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-12-09

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset


class ConceptDataset(Dataset):
    """Dataset for loading concept examples."""

    def __init__(
        self,
        positive_dir: Path | str,
        negative_dir: Path | str,
        transform: Any,
    ):
        """
        Initialize concept dataset.

        Args:
            positive_dir: Directory with positive concept examples
            negative_dir: Directory with negative concept examples
            transform: Image transforms to apply
        """
        self.positive_dir = Path(positive_dir)
        self.negative_dir = Path(negative_dir)
        self.transform = transform

        # Collect image paths
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

        self.positive_paths = [
            p
            for p in self.positive_dir.iterdir()
            if p.suffix.lower() in image_extensions
        ]
        self.negative_paths = [
            p
            for p in self.negative_dir.iterdir()
            if p.suffix.lower() in image_extensions
        ]

        self.all_paths = self.positive_paths + self.negative_paths
        self.labels = [1] * len(self.positive_paths) + [0] * len(self.negative_paths)

    def __len__(self) -> int:
        return len(self.all_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image_path = self.all_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")
        image_torch: torch.Tensor = self.transform(image)

        return image_torch, label
