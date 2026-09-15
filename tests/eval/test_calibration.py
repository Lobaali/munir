"""Tests for Cohen's kappa implementation (rubric 4.5)."""

from eval.calibrate_judge import cohen_kappa


def test_perfect_agreement_is_one():
    assert cohen_kappa([1.0, 0.5, 0.0], [1.0, 0.5, 0.0]) == 1.0


def test_kappa_is_below_one_when_labels_disagree():
    value = cohen_kappa([1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0])
    assert value < 1.0
