"""Tests for the typed config loader."""

from pathlib import Path

import pytest

from melanomanet.config import load_config

REPO_ROOT = Path(__file__).parents[1]


def test_load_real_config() -> None:
    cfg = load_config(REPO_ROOT / "config.yaml")

    assert cfg.data.num_classes == 9
    assert cfg.data.class_names[0] == "MEL"
    assert cfg.training.augmentation.color_jitter.hue == 0.1
    assert cfg.training.augmentation.random_affine is True
    assert cfg.fastcav.batch_size == 32  # default: absent from config.yaml
    assert cfg.abcde.diameter_threshold_px == 114
    assert cfg.seed == 25
    assert cfg.mixed_precision is True


def test_unknown_key_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        _minimal_yaml() + "model:\n  backbon: efficientnet_v2_s\n",
        encoding="utf-8",
    )
    with pytest.raises(TypeError):
        load_config(bad)


def test_missing_optional_sections_use_defaults(tmp_path: Path) -> None:
    sparse = tmp_path / "sparse.yaml"
    sparse.write_text(_minimal_yaml(), encoding="utf-8")

    cfg = load_config(sparse)

    assert cfg.model.backbone == "efficientnet_v2_m"
    assert cfg.uncertainty.enable is True
    assert cfg.fastcav.concepts_dir == "./data/concepts"
    assert cfg.paths.output_dir == "./outputs"
    assert cfg.training.augmentation.horizontal_flip == 0.5
    assert cfg.mixed_precision is False


def _minimal_yaml() -> str:
    return (
        "data:\n"
        "  dataset_name: ISIC2019\n"
        "  data_dir: ./data/isic_2019\n"
        "  train_split: 0.7\n"
        "  val_split: 0.15\n"
        "  test_split: 0.15\n"
        "  num_classes: 9\n"
        "  class_names: [MEL, NV, BCC, AK, BKL, DF, VASC, SCC, UNK]\n"
        "  image_size: 384\n"
        "  num_workers: 4\n"
        "training:\n"
        "  batch_size: 8\n"
        "  epochs: 1\n"
        "  learning_rate: 0.0001\n"
        "  weight_decay: 0.0001\n"
        "  optimizer: adam\n"
        "  scheduler: cosine\n"
        "seed: 25\n"
        "device: cpu\n"
    )
