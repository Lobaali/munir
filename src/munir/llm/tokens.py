"""
Token counting utilities for Munir.

SECTION 5 — COST / LATENCY ENGINEERING

This module provides local token counting for measurement.

Important:
- Local token counts help us measure prompt size.
- Provider-reported usage remains the authoritative source
  for actual billing/cost whenever available.
- The tokenizer is not used as a replacement for provider usage.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from tokenizers import Tokenizer
except ImportError:  # pragma: no cover
    Tokenizer = None


@dataclass(frozen=True)
class TokenCount:
    """
    Result of a token-counting operation.

    text:
        The text that was measured.

    tokens:
        Number of tokens counted.
    """

    text: str
    tokens: int


class TokenCounter:
    """
    Provider-independent wrapper around the tokenizers library.

    Keeping the tokenizer behind our own class prevents the rest
    of Munir from depending directly on the tokenizers package.
    """

    def __init__(self, tokenizer_path: str | None = None) -> None:
        """
        Create a token counter.

        If tokenizer_path is provided, it must point to a compatible
        tokenizer.json file.

        When no tokenizer is configured, a deterministic whitespace
        approximation is used for development/testing only.
        """
        self._tokenizer = None

        if tokenizer_path and Tokenizer is not None:
            self._tokenizer = Tokenizer.from_file(tokenizer_path)

    def count(self, text: str) -> TokenCount:
        """
        Count tokens in a piece of text.

        A real tokenizer is preferred when configured.

        The fallback is only an approximation and must not be treated
        as provider billing data.
        """
        if self._tokenizer is not None:
            encoded = self._tokenizer.encode(text)
            token_count = len(encoded.ids)
        else:
            token_count = len(text.split()) if text.strip() else 0

        return TokenCount(
            text=text,
            tokens=token_count,
        )

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        """
        Count the text contained in chat messages.

        This includes the role and content of every message.

        This is a local approximation because providers can apply
        different chat-message overhead rules.
        """
        total = 0

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            total += self.count(role).tokens
            total += self.count(content).tokens

        return total