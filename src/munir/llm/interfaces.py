"""
Munir LLM interfaces.

RUBRIC 1B:
    Typed model boundary.

The application talks to LLMClient rather than directly talking
to OpenAI or another provider SDK.

RUBRIC 2D:
    ToolCall is Munir's provider-independent representation of a
    function/tool call.

RUBRIC 5A:
    Usage stores token information needed for cost measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol


@dataclass
class LLMRequest:
    """
    Request sent through Munir's model boundary.

    This is NOT an OpenAI request object.

    It is Munir's own typed representation.
    """

    messages: list[dict[str, Any]]

    # The actual model is resolved from configuration before this request
    # reaches the provider.
    model: str

    temperature: float = 0.0

    # RUBRIC 2B:
    # Structured-output schema sent to the provider.
    response_format: dict[str, Any] | None = None

    # RUBRIC 2D:
    # Tool definitions sent to the provider.
    tools: list[dict[str, Any]] | None = None


@dataclass
class Usage:
    """
    Token usage returned by a model provider.

    RUBRIC 5A:
        Per-request usage -> cost record.

    cached_input_tokens is included for RUBRIC 5B:
        Prompt-cache usage.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ToolCall:
    """
    Provider-independent representation of a tool call.

    RUBRIC 2D:
        The provider adapter converts provider-specific tool-call
        objects into this structure.
    """

    id: str
    name: str
    arguments: str


@dataclass
class LLMResponse:
    """
    Provider-independent response returned to Munir.
    """

    text: str | None
    model_id: str
    finish_reason: str | None

    usage: Usage = field(default_factory=Usage)

    tool_calls: list[ToolCall] = field(default_factory=list)

    latency_ms: float = 0.0

    # Identifies the route that produced the response.
    route: str = ""


class LLMError(Exception):
    """
    Normalized model error.

    Provider-specific exceptions are converted to this class inside
    the provider adapter.

    RUBRIC 1D:
        Allows the retry/fallback layer to determine whether an error
        should be retried.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
        raw: Exception | None = None,
    ) -> None:
        super().__init__(message)

        self.status = status
        self.retryable = retryable
        self.retry_after = retry_after
        self.raw = raw


class LLMClient(Protocol):
    """
    RUBRIC 1B:
        Typed model boundary.

    Every model implementation must provide this interface.

    Examples:
        FakeLLMClient
        OpenAICompatClient
        future local open-weight client
    """

    @property
    def route_name(self) -> str:
        ...

    def complete(self, request: LLMRequest) -> LLMResponse:
        ...

    def stream(self, request: LLMRequest) -> Iterator[Any]:
        ...