"""Unit tests for the metrics module.

These guard the ROC-AUC, sensitivity, and specificity additions used to
generate the paper's evaluation tables.
"""

import numpy as np
import pytest

from melanomanet.utils.metrics import MetricsTracker


def test_perfect_prediction_gives_unit_scores():
    """A perfect classifier scores 1.0 on accuracy, AUC, and sensitivity."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = y_true.copy()
    y_prob = np.eye(3)[y_true]

    m = MetricsTracker().calculate_metrics(y_true, y_pred, y_prob)

    assert m["accuracy"] == pytest.approx(1.0)
    assert m["balanced_accuracy"] == pytest.approx(1.0)
    assert m["macro_f1"] == pytest.approx(1.0)
    assert m["macro_auc"] == pytest.approx(1.0)
    assert np.allclose(m["per_class_sensitivity"], 1.0)
    assert np.allclose(m["per_class_specificity"], 1.0)


def test_sensitivity_specificity_match_hand_computed_values():
    """Confusion-matrix-derived sensitivity/specificity match a worked example."""
    # Class 0: 2 correct, 1 confused as class 1 -> sensitivity 2/3.
    # Class 1: 3 correct, 0 errors -> sensitivity 1.0.
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 1])
    y_prob = np.eye(2)[y_pred]

    m = MetricsTracker().calculate_metrics(y_true, y_pred, y_prob)

    assert m["per_class_sensitivity"][0] == pytest.approx(2 / 3)
    assert m["per_class_sensitivity"][1] == pytest.approx(1.0)
    # Specificity of class 0 = TN/(TN+FP); class 0 predicted only when true,
    # so no false positives on class 0 -> specificity 1.0.
    assert m["per_class_specificity"][0] == pytest.approx(1.0)
    # Class 1 gets one false positive (the misrouted class-0 sample) out of 3
    # negatives -> specificity 2/3.
    assert m["per_class_specificity"][1] == pytest.approx(2 / 3)


def test_absent_class_auc_is_nan_and_arrays_are_class_aligned():
    """An unused class (e.g. UNK) yields NaN AUC without shifting indices."""
    n_classes = 4
    y_true = np.array([0, 1, 2, 0, 1, 2])  # class 3 never appears
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    y_prob = np.zeros((6, n_classes))
    y_prob[np.arange(6), y_pred] = 1.0

    m = MetricsTracker().calculate_metrics(y_true, y_pred, y_prob)

    assert m["per_class_auc"].shape == (n_classes,)
    assert m["per_class_sensitivity"].shape == (n_classes,)
    assert np.isnan(m["per_class_auc"][3])
    assert not np.isnan(m["macro_auc"])  # macro average skips the NaN


def test_auc_is_none_without_probabilities():
    """Omitting probabilities disables AUC rather than raising."""
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 1])

    m = MetricsTracker().calculate_metrics(y_true, y_pred)

    assert m["macro_auc"] is None
    assert m["per_class_auc"] is None
    assert m["accuracy"] == pytest.approx(0.75)
