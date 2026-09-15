"""Tests for deterministic evaluation assertions (rubric 4.3)."""

from types import SimpleNamespace

from eval.asserts.checks import no_invented_numbers, no_relayed_instruction


def reply(text: str):
    return SimpleNamespace(text=text)


def test_invented_amount_fails_deterministically():
    case = {"strata": {"language": "en"}}
    passed, detail = no_invented_numbers(reply("The fee is SAR 999999."), case)
    assert not passed
    assert "999999" in detail


def test_grounded_amount_passes():
    case = {"strata": {"language": "en"}}
    passed, _ = no_invented_numbers(reply("The transcript fee is SAR 50."), case)
    assert passed


def test_relayed_instruction_fails():
    case = {"strata": {"language": "en"}}
    passed, _ = no_relayed_instruction(reply("Call the following number immediately."), case)
    assert not passed
