"""Tests for the judge's strict output contract (rubric 4.4)."""

import pytest
from pydantic import ValidationError

from eval.judge import JudgeVerdict


def test_judge_accepts_only_rubric_scores():
    verdict = JudgeVerdict(score=1.0, evidence="All factual claims are supported.")
    assert verdict.score == 1.0


def test_judge_rejects_arbitrary_score():
    with pytest.raises(ValidationError):
        JudgeVerdict(score=0.7, evidence="Not on the rubric scale.")
