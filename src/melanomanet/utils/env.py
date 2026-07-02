"""Runtime environment helpers: device resolution and seeding."""

import random

import numpy as np
import torch


def resolve_device(device: str) -> torch.device:
    """Resolve the configured device, falling back to CPU without CUDA."""
    return torch.device(device if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # Disable nondeterministic algorithms
    torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmark for performance
