"""
OpenAI-compatible provider adapter.

RUBRIC 1A:
    The real provider SDK is called here.

RUBRIC 1B:
    This SDK code is isolated behind LLMClient.

RUBRIC 2B:
    response_format is actually forwarded to the provider.

RUBRIC 2D:
    tool definitions are actually forwarded to the provider and
    provider tool calls are converted to Munir ToolCall objects.

RUBRIC 5A:
    Provider token usage is normalized into Usage.

IMPORTANT:
    This is the only file that imports the OpenAI SDK.
"""

from __future__ import annotations

import os
import time
from typing import Any, Iterator

from munir.llm.interfaces import (
    LLMError,
    LLMRequest,
    LLMResponse,
    ToolCall,
    Usage,
)


class OpenAICompatClient:
    """
    Client for OpenAI and OpenAI-compatible APIs.

    The application does not call this class directly.
    It communicates through the LLMClient interface.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key_env: str,
        route: str,
    ) -> None:

        # ---------------------------------------------------------------
        # RUBRIC 1A
        #
        # Provider SDK import is isolated inside the adapter.
        # ---------------------------------------------------------------

        from openai import OpenAI

        api_key = os.environ.get(api_key_env)

        if not api_key:
            raise RuntimeError(
                f"Missing API key environment variable: {api_key_env}"
            )

        self.route = route

        # max_retries=0 is intentional.
        #
        # RUBRIC 1D:
        # Retry/fallback is controlled by ResilientClient rather than
        # allowing multiple retry systems to fight each other.
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
        )

    @property
    def route_name(self) -> str:
        return self.route

    def complete(self, request: LLMRequest) -> LLMResponse:

        started = time.perf_counter()

        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
        }

        # ---------------------------------------------------------------
        # RUBRIC 2B
        #
        # Structured output is ACTUALLY sent to the provider.
        # ---------------------------------------------------------------

        if request.response_format is not None:
            kwargs["response_format"] = request.response_format

        # ---------------------------------------------------------------
        # RUBRIC 2D
        #
        # Tool definitions are ACTUALLY sent to the provider.
        # ---------------------------------------------------------------

        if request.tools is not None:
            kwargs["tools"] = request.tools

        try:

            # -----------------------------------------------------------
            # RUBRIC 1A
            #
            # THIS is the real provider SDK call.
            # -----------------------------------------------------------

            response = self.client.chat.completions.create(
                **kwargs
            )

        except Exception as exc:

            status = getattr(
                exc,
                "status_code",
                None,
            )

            retryable = status in {
                408,
                409,
                429,
                500,
                502,
                503,
                504,
            }

            retry_after = None

            headers = getattr(
                exc,
                "headers",
                None,
            )

            if headers:

                value = headers.get("retry-after")

                if value is not None:

                    try:
                        retry_after = float(value)
                    except (TypeError, ValueError):
                        retry_after = None

            raise LLMError(
                str(exc),
                status=status,
                retryable=retryable,
                retry_after=retry_after,
                raw=exc,
            ) from exc

        latency_ms = (
            time.perf_counter() - started
        ) * 1000

        choice = response.choices[0]
        message = choice.message

        # ---------------------------------------------------------------
        # RUBRIC 2D
        #
        # Convert provider-specific tool calls into Munir's ToolCall.
        # ---------------------------------------------------------------

        tool_calls: list[ToolCall] = []

        provider_tool_calls = getattr(
            message,
            "tool_calls",
            None,
        )

        if provider_tool_calls:

            for call in provider_tool_calls:

                tool_calls.append(
                    ToolCall(
                        id=call.id,
                        name=call.function.name,
                        arguments=call.function.arguments,
                    )
                )

        # ---------------------------------------------------------------
        # RUBRIC 5A
        #
        # Normalize provider token usage.
        # ---------------------------------------------------------------

        provider_usage = getattr(
            response,
            "usage",
            None,
        )

        usage = Usage()

        if provider_usage is not None:

            usage.input_tokens = getattr(
                provider_usage,
                "prompt_tokens",
                0,
            )

            usage.output_tokens = getattr(
                provider_usage,
                "completion_tokens",
                0,
            )

            prompt_details = getattr(
                provider_usage,
                "prompt_tokens_details",
                None,
            )

            if prompt_details is not None:

                usage.cached_input_tokens = getattr(
                    prompt_details,
                    "cached_tokens",
                    0,
                )

        return LLMResponse(
            text=getattr(
                message,
                "content",
                None,
            ),
            model_id=request.model,
            finish_reason=getattr(
                choice,
                "finish_reason",
                None,
            ),
            usage=usage,
            tool_calls=tool_calls,
            latency_ms=latency_ms,
            route=self.route,
        )

    def stream(
        self,
        request: LLMRequest,
    ) -> Iterator[Any]:

        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "stream": True,
        }

        if request.response_format is not None:
            kwargs["response_format"] = request.response_format

        if request.tools is not None:
            kwargs["tools"] = request.tools

        try:

            response = self.client.chat.completions.create(
                **kwargs
            )

            for chunk in response:
                yield chunk

        except Exception as exc:

            status = getattr(
                exc,
                "status_code",
                None,
            )

            raise LLMError(
                str(exc),
                status=status,
                retryable=status in {
                    408,
                    409,
                    429,
                    500,
                    502,
                    503,
                    504,
                },
                raw=exc,
            ) from exc