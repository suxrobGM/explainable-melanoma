"""Tests for faithfulness metric helpers."""

import numpy as np

from melanomanet.evaluation.faithfulness import _deletion_aopc, _insertion_aopc, _iou


def test_aopc_deletion_rewards_fast_probability_drop() -> None:
    fast_drop = np.array([0.9, 0.1, 0.05, 0.0])
    slow_drop = np.array([0.9, 0.8, 0.7, 0.6])
    assert _deletion_aopc(fast_drop) > _deletion_aopc(slow_drop)


def test_aopc_insertion_rewards_fast_probability_rise() -> None:
    fast_rise = np.array([0.0, 0.8, 0.9, 0.9])
    slow_rise = np.array([0.0, 0.1, 0.2, 0.3])
    assert _insertion_aopc(fast_rise) > _insertion_aopc(slow_rise)


def test_aopc_flat_curve_is_zero() -> None:
    flat = np.array([0.5, 0.5, 0.5])
    assert _deletion_aopc(flat) == 0.0
    assert _insertion_aopc(flat) == 0.0


def test_iou_perfect_overlap() -> None:
    attention = np.zeros((4, 4))
    attention[:2, :2] = 1.0
    lesion = np.zeros((4, 4))
    lesion[:2, :2] = 1
    # Top quantile of attention selects exactly the lesion quadrant.
    assert _iou(attention, lesion, quantile=0.75) == 1.0


def test_iou_no_overlap() -> None:
    attention = np.zeros((4, 4))
    attention[:2, :2] = 1.0
    lesion = np.zeros((4, 4))
    lesion[2:, 2:] = 1
    assert _iou(attention, lesion, quantile=0.75) == 0.0


def test_iou_empty_union_is_zero() -> None:
    assert _iou(np.zeros((4, 4)), np.zeros((4, 4)), quantile=1.0) >= 0.0
