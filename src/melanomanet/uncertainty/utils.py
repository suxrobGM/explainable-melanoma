# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-12-09

"""
Utility functions for uncertainty estimation.
"""

import numpy as np

from .models import UncertaintyResult


def compute_calibration_metrics(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> dict:
    """
    Compute calibration metrics (ECE, MCE).

    Args:
        probs: Predicted probabilities (N, C)
        labels: True labels (N,)
        n_bins: Number of bins for calibration

    Returns:
        Dictionary with ECE, MCE, and per-bin statistics
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    # Bin boundaries
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    bin_stats = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            bin_error = np.abs(avg_accuracy - avg_confidence)

            ece += prop_in_bin * bin_error
            mce = max(mce, bin_error)

            bin_stats.append(
                {
                    "bin_lower": bin_lower,
                    "bin_upper": bin_upper,
                    "accuracy": avg_accuracy,
                    "confidence": avg_confidence,
                    "count": in_bin.sum(),
                }
            )

    return {
        "ece": ece,
        "mce": mce,
        "bin_stats": bin_stats,
    }


def get_uncertainty_interpretation(result: UncertaintyResult) -> str:
    """
    Generate human-readable interpretation of uncertainty.

    Args:
        result: UncertaintyResult from MC Dropout estimation

    Returns:
        Interpretation string
    """
    lines = []

    # Confidence level
    if result.confidence > 0.9:
        conf_level = "Very High"
    elif result.confidence > 0.7:
        conf_level = "High"
    elif result.confidence > 0.5:
        conf_level = "Moderate"
    else:
        conf_level = "Low"

    lines.append(f"Confidence Level: {conf_level} ({result.confidence:.1%})")

    # Uncertainty interpretation
    if result.predictive_uncertainty < 0.3:
        unc_level = "Low uncertainty - prediction is reliable"
    elif result.predictive_uncertainty < 0.7:
        unc_level = "Moderate uncertainty - consider additional review"
    else:
        unc_level = "High uncertainty - manual review recommended"

    lines.append(f"Predictive Uncertainty: {unc_level}")

    # Source of uncertainty
    if result.epistemic_uncertainty > result.aleatoric_uncertainty:
        source = "Model uncertainty dominates - more training data may help"
    else:
        source = "Data uncertainty dominates - image quality may be limiting"

    lines.append(f"Uncertainty Source: {source}")

    # Reliability flag
    if result.is_reliable:
        lines.append("Status: RELIABLE - suitable for clinical decision support")
    else:
        lines.append("Status: UNCERTAIN - requires expert verification")

    return "\n".join(lines)
