"""Shared image-file helpers."""

from collections.abc import Iterator
from pathlib import Path

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def iter_image_files(directory: Path) -> Iterator[Path]:
    """Yield supported image files in a directory, sorted by name."""
    yield from sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
