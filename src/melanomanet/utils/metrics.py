"""
Evaluation metrics for multi-class skin lesion classification.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class MetricsTracker:
    """Calculate and track evaluation metrics for multi-class classification."""

    def calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> dict:
        """
        Calculate all metrics for multi-class classification.

        Args:
            y_true: Ground truth labels (N,)
            y_pred: Predicted labels (N,)
            y_prob: Predicted probabilities/confidence scores (N,) [currently unused]

        Returns:
            Dictionary of metrics (using weighted averaging for multi-class)
        """
        # Calculate metrics with weighted averaging (accounts for class imbalance)
        accuracy = float(accuracy_score(y_true, y_pred))
        precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        # Confusion matrix for reference
        cm = confusion_matrix(y_true, y_pred)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm,
        }

    def print_classification_report(
        self, y_true: np.ndarray, y_pred: np.ndarray, class_names: list
    ) -> None:
        """Print detailed classification report."""
        print("\nClassification Report:")
        print("=" * 60)
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
