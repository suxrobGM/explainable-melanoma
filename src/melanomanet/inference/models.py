"""Data models for inference results."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..explainability.models import FastCAVResult
from ..uncertainty.models import UncertaintyResult


@dataclass
class InferenceResult:
    """Container for all inference results."""

    pred_class: int
    confidence: float
    attention_map: np.ndarray
    visualization: np.ndarray
    uncertainty: UncertaintyResult | None = None
    fastcav: FastCAVResult | None = None
    abcde: dict[str, Any] | None = None
    alignment_scores: dict[str, float] | None = None
