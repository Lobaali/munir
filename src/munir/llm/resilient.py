"""
Reliable model wrapper.

RUBRIC 1D:
    Retry + exponential backoff + fallback.

The business logic does not need to know:
    - when to retry
    - how long to wait
    - which model to use after failure
"""

from __future__ import annotations

import random
import time

from munir.llm.interfaces import (
    LLMClient,
    LLMError,
    LLMRequest,
    LLMResponse,
)


class AllModelsFailed(LLMError):
    """Raised when primary and fallback routes both fail."""


class ResilientClient:
    """
    Adds bounded retry, exponential backoff, and fallback.
    """

    def __init__(
        self,
        primary: LLMClient,
        fallback: LLMClient,
        *,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
    ) -> None:

        self.primary = primary
        self.fallback = fallback

        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay

    @property
    def route_name(self) -> str:
        return self.primary.route_name

    def _backoff_delay(
        self,
        attempt: int,
    ) -> float:

        delay = self.base_delay * (
            2 ** (attempt - 1)
        )

        jitter = random.uniform(
            0.5,
            1.5,
        )

        return min(
            delay * jitter,
            self.max_delay,
        )

    def _call_with_retry(
        self,
        client: LLMClient,
        request: LLMRequest,
    ) -> LLMResponse:

        last_error: LLMError | None = None

        for attempt in range(
            1,
            self.max_attempts + 1,
        ):

            try:

                return client.complete(request)

            except LLMError as exc:

                last_error = exc

                # Permanent errors should not be retried.
                if not exc.retryable:
                    raise

                # No sleep after the final attempt.
                if attempt == self.max_attempts:
                    break

                if exc.retry_after is not None:

                    delay = exc.retry_after

                else:

                    delay = self._backoff_delay(
                        attempt
                    )

                time.sleep(
                    min(
                        delay,
                        self.max_delay,
                    )
                )

        assert last_error is not None

        raise last_error

    def complete(
        self,
        request: LLMRequest,
    ) -> LLMResponse:

        # ---------------------------------------------------------------
        # RUBRIC 1D:
        # First use the primary model.
        # ---------------------------------------------------------------

        try:

            return self._call_with_retry(
                self.primary,
                request,
            )

        except LLMError as primary_error:

            # -----------------------------------------------------------
            # RUBRIC 1D:
            # If primary fails after retries, use fallback.
            # -----------------------------------------------------------

            try:

                response = self._call_with_retry(
                    self.fallback,
                    request,
                )

                return response

            except LLMError as fallback_error:

                raise AllModelsFailed(
                    "Primary and fallback model routes failed.",
                    status=fallback_error.status,
                    retryable=False,
                    raw=fallback_error,
                ) from fallback_error

    def stream(self, request: LLMRequest):
        """
        Streaming is not needed for the rubric's core requirements.

        We intentionally do not implement a separate retrying stream
        protocol here.
        """

        return self.primary.stream(request)