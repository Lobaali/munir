"""Trace/cost observability tests for Module 6."""

from __future__ import annotations

from munir.app import build_assistant
from munir.domain.session import Session
from munir.observability import current_trace_id, new_trace_id


def test_new_trace_id_changes_current_trace_id():
    first = new_trace_id()
    assert current_trace_id() == first

    second = new_trace_id()
    assert current_trace_id() == second
    assert first != second


def test_one_request_shares_one_trace_id():
    assistant = build_assistant(
        cache_enabled=False,
        cascade_enabled=False,
    )

    assistant.ask(
        "How much does a transcript cost?",
        Session(student_id="STU-100001"),
        remember=False,
    )

    trace_ids = {
        record.trace_id
        for record in assistant.meter.records
    }

    assert len(trace_ids) == 1
    assert next(iter(trace_ids))


def test_separate_requests_get_separate_trace_ids():
    assistant = build_assistant(
        cache_enabled=False,
        cascade_enabled=False,
    )
    session = Session(student_id="STU-100001")

    assistant.ask(
        "How much does a transcript cost?",
        session,
        remember=False,
    )
    first = {
        record.trace_id
        for record in assistant.meter.records
    }

    assistant.ask(
        "What is my enrollment status?",
        session,
        remember=False,
    )
    second = {
        record.trace_id
        for record in assistant.meter.records
    }

    assert len(second) == 2
    assert first < second
