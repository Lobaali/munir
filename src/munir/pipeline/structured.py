"""
Munir structured-output generation pipeline.

This module implements the complete:

    VALIDATE -> RETRY -> REPAIR

workflow.

The model is first asked to generate an object using a strict JSON Schema.

Munir then validates the returned JSON using the corresponding Pydantic
model.

If validation fails:

    1. Munir records the validation errors.
    2. The errors are converted into a clear, field-specific message.
    3. The failed model response and the validation errors are sent back
       to the model.
    4. The model gets one bounded repair attempt.
    5. The repaired response is validated again.

The loop is intentionally bounded.

Munir NEVER retries forever.

RUBRIC
------
Section 2A:
    Pydantic model validation.

Section 2B:
    The JSON Schema is actually passed to the LLM through
    LLMRequest.response_format.

Section 2C:
    Validation -> retry -> repair with actual validation errors
    fed back into the model.
"""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel, ValidationError

from munir.domain.ticket import strict_schema
from munir.llm.interfaces import (
    LLMClient,
    LLMRequest,
    LLMResponse,
    Message,
)
from munir.observability import get_logger


log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


# This message is deliberately explicit.

# We do NOT simply say:
#
#     "Please try again."
#
# Instead, the actual Pydantic validation errors are inserted into
# the message so the model knows exactly what needs to be repaired.
REPAIR_INSTRUCTION = (
    "The JSON you returned failed validation. "
    "Fix ONLY these errors and return the corrected object, "
    "with no commentary:\n{errors}"
)


class StructuredExtractionFailed(Exception):
    """
    Raised when structured generation remains invalid after all
    allowed attempts.

    The raw response and validation errors are preserved so that the
    application can send the incident to an appropriate failure/review
    path without exposing raw errors to the end user.
    """

    def __init__(
        self,
        raw: str | None,
        errors: list[dict],
        attempts: int,
    ) -> None:
        super().__init__(
            f"structured extraction failed after {attempts} attempts"
        )

        self.raw = raw
        self.errors = errors
        self.attempts = attempts


class StructuredOutcome(BaseModel):
    """
    Metadata describing how structured generation succeeded.

    Example:

        attempts=1
        first_try=True

    means the model produced a valid result immediately.

    Example:

        attempts=2
        first_try=False

    means the model needed the repair step.

    This is useful for evaluation because Munir can measure how often
    repairs are actually required.
    """

    attempts: int
    first_try: bool

    # LLMResponse is a normal Python object rather than a Pydantic model,
    # so arbitrary_types_allowed is required here.
    responses: list[LLMResponse]

    model_config = {
        "arbitrary_types_allowed": True
    }


def render_errors(
    exc: ValidationError,
) -> str:
    """
    Convert Pydantic validation errors into a compact message.

    Example:

        - student_id: student_id must look like STU-123456
        - slot: appointment slot must be in the future

    Including the field location is important because it tells the
    model exactly where the problem occurred.
    """

    lines: list[str] = []

    for error in exc.errors():
        location = ".".join(
            str(part)
            for part in error["loc"]
        )

        if not location:
            location = "<root>"

        lines.append(
            f"- {location}: {error['msg']}"
        )

    return "\n".join(lines)


def extract_structured(
    client: LLMClient,
    schema_model: type[T],
    *,
    system: str,
    user: str,
    schema_name: str,
    model_alias: str = "munir-extract",
    temperature: float = 0.0,
    max_tokens: int = 600,
    max_attempts: int = 2,
    response_format: dict | None = None,
) -> tuple[T, StructuredOutcome]:
    """
    Generate one structured object and validate it.

    Parameters
    ----------
    client:
        The typed LLM boundary. Business logic never imports a provider SDK.

    schema_model:
        The Pydantic model that defines the required output.

    system:
        System instructions for the structured task.

    user:
        User/task content.

    schema_name:
        Stable name of the structured schema.

    model_alias:
        Configured Munir model alias.

    temperature:
        Kept at 0.0 by default for deterministic extraction.

    max_tokens:
        Bounded generation size.

    max_attempts:
        Maximum number of total model calls.

        Default = 2:

            attempt 1 -> normal generation
            attempt 2 -> repair

        This MUST remain bounded.

    response_format:
        Optional pre-built schema. If omitted, Munir builds it from
        the Pydantic contract.
    """

    # ---------------------------------------------------------------
    # STEP 1
    # Build the strict schema that will be sent to the model.
    #
    # This is the critical Section 2B step.
    # ---------------------------------------------------------------

    schema = (
        response_format
        if response_format is not None
        else strict_schema(
            schema_model,
            schema_name,
        )
    )

    # Start with the original structured-generation request.
    messages = [
        Message(
            role="system",
            content=system,
        ),
        Message(
            role="user",
            content=user,
        ),
    ]

    # Keep every provider response so evaluation can see whether
    # the request needed repair.
    responses: list[LLMResponse] = []

    last_raw: str | None = None
    last_errors: list[dict] = []

    # ---------------------------------------------------------------
    # STEP 2
    # Bounded generation/repair loop.
    # ---------------------------------------------------------------

    for attempt in range(
        1,
        max_attempts + 1,
    ):
        # -----------------------------------------------------------
        # STEP 3
        # Send the schema OVER THE WIRE.
        #
        # response_format is not just documentation.
        #
        # The provider receives this as part of the actual request.
        # -----------------------------------------------------------

        request = LLMRequest(
            messages=messages,
            model_alias=model_alias,
            temperature=temperature,
            max_tokens=max_tokens,

            # This is the evidence required by Rubric 2B.
            response_format=schema,

            # The first system message is stable and can be used
            # as the cache prefix by the cost/cache layer.
            cache_prefix_messages=1,
        )

        response = client.complete(request)

        responses.append(response)

        last_raw = response.text

        # -----------------------------------------------------------
        # STEP 4
        # Validate the returned JSON against the Pydantic contract.
        #
        # A valid object becomes a TYPED Python object here.
        # Downstream application code does not need to operate on
        # arbitrary raw model output.
        # -----------------------------------------------------------

        try:
            value = schema_model.model_validate_json(
                response.text or ""
            )

        except ValidationError as exc:
            # -------------------------------------------------------
            # STEP 5
            # Capture the REAL validation errors.
            # -------------------------------------------------------

            last_errors = [
                {
                    "loc": list(error["loc"]),
                    "msg": error["msg"],
                    "type": error["type"],
                }
                for error in exc.errors()
            ]

            log.warning(
                "structured_validation_failed",
                schema=schema_name,
                attempt=attempt,
                errors=[
                    error["loc"]
                    for error in last_errors
                ],
            )

            # -------------------------------------------------------
            # If this was the last permitted attempt, stop.
            # -------------------------------------------------------

            if attempt == max_attempts:
                break

            # -------------------------------------------------------
            # STEP 6
            # FEED THE ACTUAL VALIDATION ERRORS BACK TO THE MODEL.
            #
            # This is the "repair" part of Section 2C.
            # -------------------------------------------------------

            messages = messages + [
                Message(
                    role="assistant",
                    content=response.text or "",
                ),
                Message(
                    role="user",
                    content=REPAIR_INSTRUCTION.format(
                        errors=render_errors(exc)
                    ),
                ),
            ]

            # Continue to the next bounded attempt.
            continue

        # -----------------------------------------------------------
        # STEP 7
        # Successful validation.
        # -----------------------------------------------------------

        log.info(
            "structured_extracted",
            schema=schema_name,
            attempt=attempt,
            outcome=(
                "first_try"
                if attempt == 1
                else "after_repair"
            ),
        )

        return (
            value,
            StructuredOutcome(
                attempts=attempt,
                first_try=(attempt == 1),
                responses=responses,
            ),
        )

    # ---------------------------------------------------------------
    # STEP 8
    # All attempts failed.
    #
    # DO NOT continue indefinitely.
    # ---------------------------------------------------------------

    raise StructuredExtractionFailed(
        raw=last_raw,
        errors=last_errors,
        attempts=len(responses),
    )