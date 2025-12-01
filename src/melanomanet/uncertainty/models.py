from dataclasses import dataclass

import numpy as np


@dataclass
class UncertaintyResult:
    """Container for uncertainty estimation results."""

    predicted_class: int
    confidence: float
    mean_probs: np.ndarray
    predictive_uncertainty: float  # Total uncertainty (entropy)
    epistemic_uncertainty: float  # Model uncertainty (reducible with more data)
    aleatoric_uncertainty: float  # Data uncertainty (irreducible)
    is_reliable: bool  # True if uncertainty is below threshold
