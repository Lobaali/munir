"""
Tests for Munir's token-counting utility.

SECTION 5 — COST / LATENCY ENGINEERING
"""

from munir.llm.tokens import TokenCounter


def test_token_counter_is_deterministic():
    """
    The same input should always produce the same local
    token count.

    This gives us deterministic measurements for experiments.
    """
    counter = TokenCounter()

    text = "How much is the tuition?"

    first = counter.count(text)
    second = counter.count(text)

    assert first.tokens == second.tokens
    assert first.tokens >= 0


def test_empty_text_has_zero_tokens():
    """
    Empty text should produce zero tokens.
    """
    counter = TokenCounter()

    result = counter.count("")

    assert result.tokens == 0


def test_message_count_includes_roles_and_content():
    """
    Message counting should include both the role and content
    of each chat message.
    """
    counter = TokenCounter()

    messages = [
        {
            "role": "system",
            "content": "You are Munir.",
        },
        {
            "role": "user",
            "content": "What is the tuition?",
        },
    ]

    result = counter.count_messages(messages)

    assert result > 0