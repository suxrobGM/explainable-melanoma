# Author: Sukhrobbek Ilyosbekov
# Date: 2025-12-09

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class MetricsTracker:
    """Calculate and track evaluation metrics for multi-class classification."""

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None = None,
    ) -> dict:
        """
        Calculate classification metrics for multi-class evaluation.

        Args:
            y_true: Ground truth labels, shape (N,).
            y_pred: Predicted labels, shape (N,).
            y_prob: Predicted class probabilities, shape (N, C). Required for
                ROC-AUC; when omitted, AUC entries are set to ``None``.

        Returns:
            Dictionary of metrics. Weighted averages account for class imbalance;
            macro averages and balanced accuracy weight every class equally and
            are the more informative headline numbers on this imbalanced dataset.
        """
        # Fix the class axis so every per-class array is index-aligned with the
        # class label, regardless of which classes happen to appear in a split.
        if y_prob is not None:
            n_classes = y_prob.shape[1]
        else:
            n_classes = int(max(y_true.max(), y_pred.max())) + 1
        labels = list(range(n_classes))

        accuracy = float(accuracy_score(y_true, y_pred))
        balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
        precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        sensitivity, specificity = self._per_class_sensitivity_specificity(cm)

        macro_auc, per_class_auc = self._roc_auc(y_true, y_prob)

        return {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "macro_f1": macro_f1,
            "macro_auc": macro_auc,
            "per_class_auc": per_class_auc,
            "per_class_sensitivity": sensitivity,
            "per_class_specificity": specificity,
            "confusion_matrix": cm,
        }

    @staticmethod
    def _per_class_sensitivity_specificity(
        cm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Derive per-class sensitivity and specificity from a confusion matrix.

        For class ``i`` (one-vs-rest): sensitivity = TP / (TP + FN) is the
        recall on that class, and specificity = TN / (TN + FP) is the recall on
        every other class combined.
        """
        cm = cm.astype(np.float64)
        total = cm.sum()
        tp = np.diag(cm)
        fn = cm.sum(axis=1) - tp
        fp = cm.sum(axis=0) - tp
        tn = total - tp - fn - fp

        with np.errstate(divide="ignore", invalid="ignore"):
            sensitivity = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
            specificity = np.where((tn + fp) > 0, tn / (tn + fp), 0.0)

        return sensitivity, specificity

    @staticmethod
    def _roc_auc(
        y_true: np.ndarray, y_prob: np.ndarray | None
    ) -> tuple[float | None, np.ndarray | None]:
        """
        Compute macro and per-class one-vs-rest ROC-AUC.

        Classes absent from ``y_true`` (e.g. the unused UNK category) cannot have
        a defined AUC and are reported as NaN in the per-class array while being
        skipped in the macro average.
        """
        if y_prob is None:
            return None, None

        n_classes = y_prob.shape[1]
        present = np.unique(y_true)

        per_class = np.full(n_classes, np.nan, dtype=np.float64)
        for c in present:
            per_class[c] = roc_auc_score((y_true == c).astype(int), y_prob[:, c])

        macro_auc = float(np.nanmean(per_class)) if present.size > 0 else None
        return macro_auc, per_class

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
