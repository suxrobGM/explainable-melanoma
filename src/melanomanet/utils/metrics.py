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

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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
        precision = float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        recall = float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        )
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

        # Get unique labels present in the data
        unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))

        # Filter class names to only include present classes
        present_class_names = [class_names[i] for i in unique_labels]

        print(
            classification_report(
                y_true,
                y_pred,
                labels=unique_labels,
                target_names=present_class_names,
                digits=4,
                zero_division=0,
            )
        )
