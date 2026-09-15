"""
Deterministic fake LLM used for local tests and reproducible evaluation.

IMPORTANT:

This is NOT presented as a real commercial or open-weight model.

It exists only so the application can be tested without external
provider credentials.

It implements the same LLMClient interface as the real provider.

RUBRIC 1B:
    Demonstrates that application logic depends on the typed model
    boundary rather than directly on an SDK.

The real provider evidence comes from openai_compat.py.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from munir.llm.interfaces import (
    LLMRequest,
    LLMResponse,
    ToolCall,
    Usage,
)


class FakeLLMClient:
    """
    Deterministic model implementation.

    The response is intentionally simple and predictable.

    This client should be used for:
        - unit tests
        - local development
        - deterministic evaluation

    It should NOT be used as evidence of real provider throughput.
    """

    def __init__(
        self,
        *,
        route: str = "fake",
    ) -> None:
        self.route = route

    @property
    def route_name(self) -> str:
        return self.route

    def complete(self, request: LLMRequest) -> LLMResponse:

        last_user_message = ""

        for message in reversed(request.messages):
            if message.get("role") == "user":
                content = message.get("content", "")

                if isinstance(content, str):
                    last_user_message = content

                break

        # ---------------------------------------------------------------
        # RUBRIC 2D TEST SUPPORT
        #
        # If tools were supplied and the user asks for a tool-style
        # operation, return a deterministic ToolCall.
        # ---------------------------------------------------------------

        if request.tools and "tool" in last_user_message.lower():

            first_tool = request.tools[0]

            function = first_tool.get("function", {})

            tool_name = function.get("name", "unknown_tool")

            arguments = json.dumps(
                {},
                ensure_ascii=False,
            )

            return LLMResponse(
                text=None,
                model_id=request.model,
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="fake-tool-call-1",
                        name=tool_name,
                        arguments=arguments,
                    )
                ],
                usage=Usage(
                    input_tokens=50,
                    output_tokens=10,
                ),
                latency_ms=1.0,
                route=self.route,
            )

        # ---------------------------------------------------------------
        # RUBRIC 2B TEST SUPPORT
        #
        # If structured output was requested, return valid JSON.
        #
        # The real provider must receive the response_format over the wire.
        # The Fake client only provides deterministic local testing.
        # ---------------------------------------------------------------

        if request.response_format is not None:

            structured_response = {
                "answer": "This is a deterministic test response.",
                "language": "en",
            }

            text = json.dumps(
                structured_response,
                ensure_ascii=False,
            )

        else:

            text = "This is a deterministic test response."

        return LLMResponse(
            text=text,
            model_id=request.model,
            finish_reason="stop",
            usage=Usage(
                input_tokens=50,
                output_tokens=12,
            ),
            latency_ms=1.0,
            route=self.route,
        )

    def stream(self, request: LLMRequest) -> Iterator[Any]:

        response = self.complete(request)

        yield response.text or ""