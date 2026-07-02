# Author: Sukhrobbek Ilyosbekov
# Date: 2025-12-09

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


@dataclass
class PaperFigureData:
    """Data container for paper figure generation."""

    prediction: str
    confidence: float
    risk_level: str

    # Images (as numpy arrays); gradcam is the raw 2-D attention heatmap
    original: np.ndarray
    gradcam: np.ndarray
    overlay: np.ndarray
    asymmetry_img: np.ndarray
    border_img: np.ndarray
    color_img: np.ndarray
    diameter_img: np.ndarray

    # ABCDE scores and analyzer-resolved flags
    asymmetry_score: float
    border_score: float
    n_colors: int
    diameter: float
    asymmetry_flag: bool
    border_flag: bool
    color_flag: bool
    diameter_flag: bool

    # Uncertainty
    predictive_unc: float
    epistemic_unc: float
    aleatoric_unc: float
    is_reliable: bool

    # FastCAV
    concepts: list[str]
    scores: list[float]
