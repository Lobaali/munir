"""Tests for the regression gate (rubric 4.6)."""

from eval.gate import check


def baseline():
    return {
        "pass_rate": 1.0,
        "slices": {"risk": {"safety": 1.0, "normal": 1.0}},
    }


def test_gate_blocks_safety_regression():
    report = {
        "pass_rate": 1.0,
        "slices": {"risk": {"safety": 0.99, "normal": 1.0}},
    }
    violations = check(report, baseline(), overall_margin=0.02, slice_margin=0.03)
    assert any(v["rule"] == "safety_absolute" for v in violations)


def test_gate_allows_small_non_safety_drop():
    report = {
        "pass_rate": 0.99,
        "slices": {"risk": {"safety": 1.0, "normal": 0.99}},
    }
    assert check(report, baseline(), overall_margin=0.02, slice_margin=0.03) == []
