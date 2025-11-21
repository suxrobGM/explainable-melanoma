"""
Evaluation metrics for melanoma detection.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


class MetricsTracker:
    """Calculate and track evaluation metrics."""

    def calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> dict[str, float]:
        """
        Calculate all metrics.

        Args:
            y_true: Ground truth labels (N,)
            y_pred: Predicted labels (N,)
            y_prob: Predicted probabilities for positive class (N,)

        Returns:
            Dictionary of metrics
        """
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # F1 score
        if precision + sensitivity > 0:
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
        else:
            f1 = 0.0

        # AUC
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.0

        return {
            "accuracy": accuracy,
            "sensitivity": sensitivity,  # TPR, Recall
            "specificity": specificity,  # TNR
            "precision": precision,
            "f1": f1,
            "auc": auc,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }

    def print_classification_report(
        self, y_true: np.ndarray, y_pred: np.ndarray, class_names: list
    ) -> None:
        """Print detailed classification report."""
        print("\nClassification Report:")
        print("=" * 60)
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
