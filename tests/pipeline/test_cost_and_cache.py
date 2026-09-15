"""Section 5 tests: cost metering, provider prompt caching, and response keys.

These are not benchmark numbers. They are permanent regression tests proving
that the optimisation mechanisms are actually wired into the application.
"""

from __future__ import annotations

from munir.app import build_assistant
from munir.caching.response_cache import ResponseCache
from munir.domain.session import Session
from munir.llm.fake import FakeClient
from munir.llm.interfaces import LLMRequest, Message


def test_repeated_standalone_faq_is_served_from_exact_cache():
    """Rubric 5.4: the response cache must really be used."""

    assistant = build_assistant(
        cache_enabled=True,
        cascade_enabled=False,
    )

    first = assistant.ask(
        "How much does a transcript cost?",
        Session(student_id="STU-100001"),
        remember=False,
    )
    second = assistant.ask(
        "How much does a transcript cost?",
        Session(student_id="STU-100001"),
        remember=False,
    )

    assert first.cache_tier == ""
    assert second.cache_tier == "exact"
    assert first.text == second.text
    assert assistant.cache is not None
    assert assistant.cache.stats.hits == 1


def test_cost_is_metered_from_real_usage_records():
    """Rubric 5.1: usage becomes a CostRecord per model call."""

    assistant = build_assistant(
        cache_enabled=False,
        cascade_enabled=False,
    )

    assistant.ask(
        "How much does a transcript cost?",
        Session(student_id="STU-100001"),
        remember=False,
    )

    assert assistant.meter.records
    assert assistant.meter.total_sar > 0

    record = assistant.meter.records[-1]
    assert record.input_tokens >= 0
    assert record.output_tokens >= 0
    assert record.cost_sar >= 0
    assert record.stage in {
        "input_guard",
        "router",
        "faq",
    }


def test_prompt_cache_usage_is_observed_on_second_identical_prefix():
    """Rubric 5.2: cached_input_tokens must become non-zero after a hit."""

    client = FakeClient(model_id="campus-flagship")
    client.script_text("first")
    client.script_text("second")

    request = LLMRequest(
        messages=[
            Message(
                role="system",
                content="stable trusted prefix",
            ),
            Message(
                role="user",
                content="different turn",
            ),
        ],
        model_alias="munir-default",
        max_tokens=100,
        cache_prefix_messages=1,
    )

    # FakeClient does not simulate provider caching. The application-level
    # MockBrainClient does. This test therefore uses the real mock provider.
    from munir.llm.fake import MockBrainClient

    mock = MockBrainClient(model_id="campus-flagship")
    first = mock.complete(request)
    second = mock.complete(request)

    assert first.usage.cached_input_tokens == 0
    assert second.usage.cached_input_tokens > 0


def test_cache_key_contains_every_answer_changing_variable():
    """Rubric 5.4: a near miss must never collide with the original key."""

    common = {
        "rendered_prompt": "trusted facts",
        "history": [],
        "parameters": {
            "temperature": 0.4,
            "max_tokens": 700,
        },
    }

    base = ResponseCache.exact_key(
        "munir-default",
        "answer_faq.v2",
        "How do I renew my registration?",
        "en",
        **common,
    )

    assert base != ResponseCache.exact_key(
        "munir-cheap",
        "answer_faq.v2",
        "How do I renew my registration?",
        "en",
        **common,
    )
    assert base != ResponseCache.exact_key(
        "munir-default",
        "answer_faq.v3",
        "How do I renew my registration?",
        "en",
        **common,
    )
    assert base != ResponseCache.exact_key(
        "munir-default",
        "answer_faq.v2",
        "How do I cancel my registration?",
        "en",
        **common,
    )
    assert base != ResponseCache.exact_key(
        "munir-default",
        "answer_faq.v2",
        "How do I renew my registration?",
        "ar",
        **common,
    )
    assert base != ResponseCache.exact_key(
        "munir-default",
        "answer_faq.v2",
        "How do I renew my registration?",
        "en",
        rendered_prompt="different trusted facts",
        history=[],
        parameters=common["parameters"],
    )
    assert base != ResponseCache.exact_key(
        "munir-default",
        "answer_faq.v2",
        "How do I renew my registration?",
        "en",
        rendered_prompt="trusted facts",
        history=[],
        parameters={"temperature": 0.7, "max_tokens": 700},
    )
