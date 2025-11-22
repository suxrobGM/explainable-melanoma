"""
DataLoader factory for creating train/val/test data loaders.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .dataset import MelanomaDataset, get_class_weights
from .transforms import get_train_transforms, get_val_transforms


def prepare_isic2019_labels(data_dir: str) -> pd.DataFrame:
    """
    Prepare ISIC2019 labels DataFrame for multi-class classification.

    Expected file structure:
    data_dir/
        train/
            ISIC_0000001.jpg
            ISIC_0000002.jpg
            ...
        ISIC_2019_Training_GroundTruth.csv

    CSV format: image,MEL,NV,BCC,AK,BKL,DF,VASC,SCC,UNK
    One-hot encoded: exactly one column = 1.0, others = 0.0

    Args:
        data_dir: Root directory containing ISIC2019 data

    Returns:
        DataFrame with columns ['image_id', 'target']
        target: 0=MEL, 1=NV, 2=BCC, 3=AK, 4=BKL, 5=DF, 6=VASC, 7=SCC, 8=UNK
    """
    data_path = Path(data_dir)
    csv_path = data_path / "ISIC_2019_Training_GroundTruth.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {csv_path}")

    # Load labels
    df = pd.read_csv(csv_path)

    # Multi-class classification: Convert one-hot to class index
    class_columns = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]

    # Find which column has 1.0 for each row (argmax across class columns)
    df["target"] = df[class_columns].values.argmax(axis=1)

    df = df.rename(columns={"image": "image_id"})
    df = df[["image_id", "target"]]

    print(f"Loaded {len(df)} images")
    print(f"Class distribution:")
    class_names = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]

    for i, name in enumerate(class_names):
        count = (df["target"] == i).sum()
        print(f"  {name}: {count} ({count/len(df)*100:.2f}%)")

    return df


def create_data_loaders(
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Create train, validation, and test data loaders.

    Performs stratified split to maintain class balance across splits.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_weights)
    """
    # Load labels for ISIC2019
    data_dir = config["data"]["data_dir"]
    labels_df = prepare_isic2019_labels(data_dir)
    image_dir = Path(data_dir) / "train"

    # Calculate class weights
    num_classes = config["data"]["num_classes"]
    class_weights = get_class_weights(labels_df, num_classes)

    # Stratified train/val/test split
    train_split = config["data"]["train_split"]
    val_split = config["data"]["val_split"]
    test_split = config["data"]["test_split"]

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        labels_df,
        train_size=train_split,
        stratify=labels_df["target"],
        random_state=config["seed"],
    )

    # Second split: val vs test
    val_ratio = val_split / (val_split + test_split)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio,
        stratify=temp_df["target"],
        random_state=config["seed"],
    )

    print(f"\nDataset splits:")
    print(f"Train: {len(train_df)} images")
    print(f"Val: {len(val_df)} images")
    print(f"Test: {len(test_df)} images")

    # Create datasets
    train_transforms = get_train_transforms(config)
    val_transforms = get_val_transforms(config)

    train_dataset = MelanomaDataset(image_dir, train_df, train_transforms)
    val_dataset = MelanomaDataset(image_dir, val_df, val_transforms)
    test_dataset = MelanomaDataset(image_dir, test_df, val_transforms)

    # Create data loaders
    batch_size = config["training"]["batch_size"]
    num_workers = config["data"]["num_workers"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # For stable batch norm
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_weights
