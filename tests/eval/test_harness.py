"""Tests for the declarative assertion dispatcher (rubric 4.2/4.3)."""

from types import SimpleNamespace

from eval.harness import evaluate_assert, load_cases


def test_golden_set_is_stratified_and_nonempty():
    cases = load_cases()
    assert len(cases) >= 40
    assert {case["strata"]["language"] for case in cases} >= {"en", "ar"}
    assert {case["strata"]["risk"] for case in cases} >= {"normal", "safety"}


def test_blocked_assertion_is_deterministic():
    blocked = SimpleNamespace(text="refused", blocked=True)
    passed, _ = evaluate_assert({"type": "blocked"}, blocked, {})
    assert passed


def test_llm_rubric_assertion_does_not_gate_by_itself():
    reply = SimpleNamespace(text="answer")
    passed, detail = evaluate_assert({"type": "llm-rubric"}, reply, {})
    assert passed
    assert "tracked" in detail
