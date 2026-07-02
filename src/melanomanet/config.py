"""Typed configuration mirroring config.yaml.

Sections that inference treats as optional (model, evaluation, uncertainty,
fastcav, abcde, paths) are fully defaulted so a sparse YAML still loads;
unknown keys raise TypeError at construction.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ColorJitterConfig:
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.1


@dataclass(frozen=True)
class AugmentationConfig:
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.5
    rotation_degrees: float = 20
    random_affine: bool = False
    color_jitter: ColorJitterConfig = field(default_factory=ColorJitterConfig)


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str
    data_dir: str
    train_split: float
    val_split: float
    test_split: float
    num_classes: int
    class_names: list[str]
    image_size: int
    num_workers: int


@dataclass(frozen=True)
class ModelConfig:
    backbone: str = "efficientnet_v2_m"
    pretrained: bool = True
    dropout_rate: float = 0.3


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    optimizer: str
    scheduler: str
    augmentation: AugmentationConfig
    checkpoint_save_interval: int = 0
    use_class_weights: bool = True
    focal_loss: bool = False
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


@dataclass(frozen=True)
class EvaluationConfig:
    metrics: list[str] = field(default_factory=list)
    save_confusion_matrix: bool = True


@dataclass(frozen=True)
class UncertaintyConfig:
    enable: bool = True
    n_samples: int = 10
    uncertainty_threshold: float = 0.5
    temperature_scaling: bool = False


@dataclass(frozen=True)
class FastCAVConfig:
    enable: bool = True
    concepts_dir: str = "./data/concepts"
    cavs_path: str = "./checkpoints/cavs.pth"
    concepts: list[str] = field(default_factory=list)
    batch_size: int = 32


@dataclass(frozen=True)
class ABCDEConfig:
    enable: bool = True
    asymmetry_threshold: float = 0.3
    border_threshold: float = 0.4
    color_threshold: int = 3
    diameter_threshold_px: float = 114
    enable_alignment_analysis: bool = True


@dataclass(frozen=True)
class PathsConfig:
    checkpoint_dir: str = "./checkpoints"
    output_dir: str = "./outputs"


@dataclass(frozen=True)
class Config:
    data: DataConfig
    training: TrainingConfig
    seed: int
    device: str
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    fastcav: FastCAVConfig = field(default_factory=FastCAVConfig)
    abcde: ABCDEConfig = field(default_factory=ABCDEConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    mixed_precision: bool = False


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file into a Config."""
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    training_raw = dict(raw["training"])
    augmentation_raw = dict(training_raw.pop("augmentation", {}))
    color_jitter = ColorJitterConfig(**augmentation_raw.pop("color_jitter", {}))
    augmentation = AugmentationConfig(color_jitter=color_jitter, **augmentation_raw)

    return Config(
        data=DataConfig(**raw["data"]),
        training=TrainingConfig(augmentation=augmentation, **training_raw),
        model=ModelConfig(**raw.get("model", {})),
        evaluation=EvaluationConfig(**raw.get("evaluation", {})),
        uncertainty=UncertaintyConfig(**raw.get("uncertainty", {})),
        fastcav=FastCAVConfig(**raw.get("fastcav", {})),
        abcde=ABCDEConfig(**raw.get("abcde", {})),
        paths=PathsConfig(**raw.get("paths", {})),
        seed=raw["seed"],
        device=raw["device"],
        mixed_precision=raw.get("mixed_precision", False),
    )
