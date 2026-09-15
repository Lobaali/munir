"""
Munir bounded function-calling loop.

This is the ONLY agentic loop in Munir.

The model can request tools, but it does not execute them itself.

The application controls:

    1. Which tools exist.
    2. Whether the requested tool exists.
    3. Whether the arguments are valid.
    4. Whether the authenticated session is authorized.
    5. Whether the tool can execute.
    6. How many tool iterations are allowed.

RUBRIC
------
Section 2D:
    Real function definitions + execution loop.

Section 2E:
    Bounded loop + authorization gate before sensitive/side-effecting
    tool execution.
"""

from __future__ import annotations

import json

from munir.domain.session import Session
from munir.llm.interfaces import (
    LLMClient,
    LLMRequest,
)
from munir.observability.cost import CostMeter
from munir.tools.registry import TOOLS


# ---------------------------------------------------------------------
# SAFETY LIMIT
# ---------------------------------------------------------------------
#
# The model is never allowed to request tools indefinitely.
#
# This prevents:
#
#     tool -> tool -> tool -> tool -> ...
#
# from becoming an uncontrolled cost or execution problem.
# ---------------------------------------------------------------------

MAX_ITERATIONS = 6


def run_with_tools(
    client: LLMClient,
    *,
    system: str,
    user: str,
    session: Session,
    model_alias: str,
    meter: CostMeter | None = None,
) -> dict:
    """
    Run one bounded model/tool interaction.

    Returns:

        {
            "text": "...",
            "tool_calls": [...],
            "iterations": 2
        }

    The conversation starts with system + user messages.

    If the model requests a tool:

        1. Munir finds the tool in the registry.
        2. Munir parses the arguments.
        3. Munir validates the arguments with Pydantic.
        4. Munir checks authorization.
        5. Munir executes the Python function.
        6. Munir sends the result back to the model.

    The model can then decide whether another tool is needed or whether
    it can produce the final answer.
    """

    # ---------------------------------------------------------------
    # INITIAL CONVERSATION
    # ---------------------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": user,
        },
    ]

    # Give the model ONLY the tools that Munir explicitly registered.
    tool_schemas = [
        tool.schema()
        for tool in TOOLS.values()
    ]

    # Store an audit-friendly summary of executed tool calls.
    executed_calls: list[dict] = []

    # ---------------------------------------------------------------
    # BOUNDED TOOL LOOP
    # ---------------------------------------------------------------

    for iteration in range(
        MAX_ITERATIONS
    ):

        # -----------------------------------------------------------
        # Ask the model for the next step.
        #
        # tools=tool_schemas is the actual function-calling interface.
        # -----------------------------------------------------------

        request = LLMRequest(
            model_alias=model_alias,
            messages=messages,
            tools=tool_schemas,
            max_tokens=500,
        )

        response = client.complete(request)

        # -----------------------------------------------------------
        # COST METERING
        #
        # If Section 5's CostMeter is supplied, record usage from
        # every model call inside this tool loop.
        # -----------------------------------------------------------

        if meter is not None:
            meter.meter(
                model_id=response.model_id,
                stage="tool_loop",
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cached_tokens=response.usage.cached_input_tokens,
            )

        # -----------------------------------------------------------
        # NO TOOL CALL
        #
        # The model has decided it can answer directly.
        # -----------------------------------------------------------

        if not response.tool_calls:
            return {
                "text": response.text,
                "tool_calls": executed_calls,
                "iterations": iteration + 1,
            }

        # -----------------------------------------------------------
        # TOOL REQUESTS
        #
        # The model may request more than one tool in a single response.
        # -----------------------------------------------------------

        for call in response.tool_calls:

            tool_name = call.name

            # -------------------------------------------------------
            # Parse the JSON arguments supplied by the model.
            # -------------------------------------------------------

            try:
                args_dict = (
                    json.loads(call.arguments)
                    if call.arguments
                    else {}
                )

            except json.JSONDecodeError:
                result = {
                    "error": "tool arguments were not valid JSON"
                }

                tool = TOOLS.get(tool_name)

                executed_calls.append(
                    {
                        "tool": tool_name,
                        "risk": (
                            tool.risk
                            if tool
                            else "unknown"
                        ),
                        "args": {},
                        "result": result,
                    }
                )

                # Give the error back to the model rather than crashing
                # the entire conversation.
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [call],
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(
                            result,
                            ensure_ascii=False,
                        ),
                        "tool_call_id": call.id,
                    }
                )

                continue

            # -------------------------------------------------------
            # Look up the requested tool in the allow-list.
            # -------------------------------------------------------

            tool = TOOLS.get(tool_name)

            if tool is None:
                # The model requested something Munir has not registered.
                result = {
                    "error": f"unknown tool: {tool_name}"
                }

            else:
                # ---------------------------------------------------
                # Execute through the centralized safety function.
                # ---------------------------------------------------
                result = _execute_tool(
                    tool,
                    args_dict,
                    session,
                )

            # -------------------------------------------------------
            # Record an audit-friendly representation.
            # -------------------------------------------------------

            executed_calls.append(
                {
                    "tool": tool_name,
                    "risk": (
                        tool.risk
                        if tool
                        else "unknown"
                    ),
                    "args": args_dict,
                    "result": result,
                }
            )

            # -------------------------------------------------------
            # Send the tool call + result back to the model.
            # -------------------------------------------------------

            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [call],
                }
            )

            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(
                        result,
                        ensure_ascii=False,
                    ),
                    "tool_call_id": call.id,
                }
            )

            # -------------------------------------------------------
            # TERMINAL TOOL
            #
            # Escalation ends the agentic loop immediately.
            # -------------------------------------------------------

            if (
                tool is not None
                and tool.risk == "terminal"
            ):
                return {
                    "text": result.get(
                        "reason",
                        "Escalated to a human.",
                    ),
                    "tool_calls": executed_calls,
                    "iterations": iteration + 1,
                }

    # -----------------------------------------------------------------
    # MAXIMUM ITERATIONS REACHED
    #
    # This is the fail-safe path.
    #
    # We NEVER continue the loop beyond MAX_ITERATIONS.
    # -----------------------------------------------------------------

    return {
        "text": (
            "I wasn't able to complete that within "
            "the allowed number of steps. "
            "Let me connect you with the Registrar's Office."
        ),
        "tool_calls": executed_calls,
        "iterations": MAX_ITERATIONS,
    }


def _execute_tool(
    tool,
    args_dict: dict,
    session: Session,
) -> dict:
    """
    Execute one registered tool safely.

    The order is deliberate:

        1. Validate arguments.
        2. Determine whether the tool touches student data.
        3. Authorize against the authenticated session.
        4. Execute the function.
        5. For side effects, enforce idempotency.
        6. Sanitize unexpected Python exceptions.

    The model's arguments are NOT trusted simply because the model
    produced them.
    """

    # ---------------------------------------------------------------
    # STEP 1 — VALIDATE TOOL ARGUMENTS
    # ---------------------------------------------------------------

    try:
        if tool.args_model is not None:
            validated_args = tool.args_model.model_validate(
                args_dict
            )
        else:
            validated_args = None

    except Exception as exc:
        # Pydantic errors are returned as data.

        # We intentionally do not raise the raw exception outside
        # this function because the model does not need a Python
        # traceback.
        return {
            "error": f"invalid arguments: {exc}"
        }

    # ---------------------------------------------------------------
    # STEP 2 — DETERMINE WHETHER THE TOOL TOUCHES STUDENT DATA
    # ---------------------------------------------------------------
    #
    # Read-only does NOT automatically mean safe.
    #
    # check_enrollment_status does not modify the database, but it
    # still reveals private academic information.
    # ---------------------------------------------------------------

    touches_student_data = (
        validated_args is not None
        and hasattr(
            validated_args,
            "student_id",
        )
    )

    # ---------------------------------------------------------------
    # STEP 3 — AUTHORIZATION GATE
    # ---------------------------------------------------------------
    #
    # Authorization happens in Python.
    #
    # We do NOT tell the model:
    #
    #     "Please only access the current student's record."
    #
    # and trust it.
    #
    # The Session object is the source of truth.
    # ---------------------------------------------------------------

    if (
        tool.risk == "side_effecting"
        or touches_student_data
    ):

        args_for_authz = (
            validated_args.model_dump()
            if validated_args is not None
            else {}
        )

        verdict = session.authorize(
            tool.name,
            args_for_authz,
        )

        if not verdict.allowed:
            return {
                "error": (
                    f"not authorized: "
                    f"{verdict.reason}"
                )
            }

    # ---------------------------------------------------------------
    # STEP 4 — ACTUAL FUNCTION EXECUTION
    # ---------------------------------------------------------------

    def _run():
        """
        Small wrapper around the actual registered Python function.

        The registrar escalation tool has a simple string argument,
        while the other tools receive their Pydantic object.
        """

        if tool.name == "escalate_to_registrar":
            return tool.fn(
                args_dict.get(
                    "reason",
                    "unspecified",
                ),
                session,
            )

        return tool.fn(
            validated_args,
            session,
        )

    # ---------------------------------------------------------------
    # STEP 5 — IDEMPOTENCY FOR SIDE EFFECTS
    # ---------------------------------------------------------------
    #
    # If the exact same side-effecting operation is retried, Munir
    # replays the previous result instead of performing the operation
    # twice.
    #
    # Example:
    #
    #     book appointment
    #     network retry
    #     book appointment again
    #
    # Without idempotency, that could create two appointments.
    # ---------------------------------------------------------------

    try:

        if tool.risk == "side_effecting":

            args_repr = json.dumps(
                args_dict,
                sort_keys=True,
                ensure_ascii=False,
            )

            idempotency_key = (
                f"{tool.name}:{args_repr}"
            )

            result, was_replayed = (
                session.replay_side_effect(
                    idempotency_key,
                    _run,
                )
            )

            if was_replayed:
                result = {
                    **result,
                    "idempotent_replay": True,
                }

            return result

        # Read-only and terminal tools execute normally.
        return _run()

    except Exception:
        # -----------------------------------------------------------
        # STEP 6 — SANITIZED FAILURE
        # -----------------------------------------------------------
        #
        # Never expose Python stack traces to the model.
        # -----------------------------------------------------------

        return {
            "error": (
                "an internal error occurred "
                "while running this action"
            )
        }