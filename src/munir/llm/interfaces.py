r"""Provider-neutral model boundary.

Module 6 adds two fields that matter for economics:
- ``Usage.cached_input_tokens`` tells us what the provider reused from its
  prompt cache.
- ``LLMRequest.cache_prefix_messages`` tells an adapter which leading
  messages are intended to be stable/cacheable.

The business logic never imports a provider SDK. It only sees these types.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

FinishReason = Literal[
    "stop",
    "length",
    "tool_calls",
    "refusal",
    "error",
]


class ToolCall(BaseModel):
    """A tool request emitted by the model and executed by the application."""

    id: str
    name: str
    arguments: str


class Message(BaseModel):
    """Provider-neutral conversation message."""

    role: Literal[
        "system",
        "user",
        "assistant",
        "tool",
    ]
    content: str
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] = Field(
        default_factory=list
    )


class LLMRequest(BaseModel):
    """Everything the provider adapter needs to make one model call."""

    messages: list[Message]

    # This is an alias, not a concrete model ID. Config resolves it.
    model_alias: str = "munir-default"

    temperature: float = Field(
        0.2,
        ge=0.0,
        le=2.0,
    )

    # Every call is bounded. This also belongs in the response-cache key.
    max_tokens: int = Field(
        1024,
        gt=0,
    )

    tools: list[dict] | None = None
    response_format: dict | None = None

    # Module 6: the application marks how many leading messages are stable.
    # The adapter/provider reports the actual reused token count in Usage.
    cache_prefix_messages: int = Field(
        0,
        ge=0,
    )


class Usage(BaseModel):
    """Measured usage returned by the model boundary."""

    input_tokens: int = 0
    output_tokens: int = 0

    # This is the evidence used by the prompt-cache part of Section 5.
    cached_input_tokens: int = 0


class LLMResponse(BaseModel):
    """Provider-neutral model response including measured usage."""

    text: str | None = None
    tool_calls: list[ToolCall] = Field(
        default_factory=list
    )
    finish_reason: FinishReason = "stop"
    model_id: str = ""
    usage: Usage = Field(
        default_factory=Usage
    )
    latency_ms: float = 0.0
    route: str = ""


class StreamChunk(BaseModel):
    """One streaming frame; final frame carries final usage."""

    delta: str = ""
    final: bool = False
    usage: Usage | None = None
    ttft_ms: float | None = None
    total_ms: float | None = None
    model_id: str = ""
    finish_reason: FinishReason | None = None


@runtime_checkable
class LLMClient(Protocol):
    """The only interface application code uses to call a model."""

    def complete(
        self,
        request: LLMRequest,
    ) -> LLMResponse: ...

    def stream(
        self,
        request: LLMRequest,
    ) -> Iterator[StreamChunk]: ...


class LLMError(RuntimeError):
    """Normalised provider failure used by the retry/fallback boundary."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
        raw: Any = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.retry_after = retry_after
        self.raw = raw

    def __repr__(self) -> str:
        return (
            f"LLMError(status={self.status}, "
            f"retryable={self.retryable}, msg={self!s})"
        )
