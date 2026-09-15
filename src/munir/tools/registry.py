"""
Munir tool registry.

This file defines the functions that the language model is allowed to
request through function calling.

IMPORTANT ARCHITECTURE RULE
---------------------------
The model NEVER directly executes Python functions.

The flow is:

    Model
      ↓
    Tool request
      ↓
    Munir tool registry
      ↓
    Argument validation
      ↓
    Authorization
      ↓
    Python function execution
      ↓
    Tool result
      ↓
    Model

Each tool has a risk classification:

    read_only
        Does not modify application state.

    side_effecting
        Changes state or creates an external request.

    terminal
        Ends the tool loop, such as escalation to a human.

RUBRIC
------
Section 2D:
    Real tool definitions and execution support.

Section 2E:
    Tool risk classification is used by the bounded tool loop and
    authorization layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from pydantic import BaseModel

from munir.domain.session import Session
from munir.domain.ticket import (
    BookingRequest,
    EnrollmentStatusQuery,
    TranscriptRequest,
    strict_schema,
)


# ---------------------------------------------------------------------
# DEMO BACKEND DATA
# ---------------------------------------------------------------------
#
# These are deterministic in-memory stand-ins for actual university
# services.
#
# In a real deployment, the functions below could call:
#
#     - a university database
#     - an academic information system
#     - a calendar service
#     - a transcript service
#
# The tool interface would remain the same.
# ---------------------------------------------------------------------

_FAKE_ENROLLMENT_DB = {
    "STU-100001": {
        "status": "enrolled",
        "credits": 15,
        "standing": "good",
    },
    "STU-100002": {
        "status": "on_probation",
        "credits": 9,
        "standing": "warning",
    },
}

_FAKE_BOOKINGS: list[dict] = []

_FAKE_TRANSCRIPT_REQUESTS: list[dict] = []


# ---------------------------------------------------------------------
# TOOL 1 — CHECK ENROLLMENT
# ---------------------------------------------------------------------

def check_enrollment_status(
    args: EnrollmentStatusQuery,
    session: Session,
) -> dict:
    """
    READ-ONLY tool.

    Returns the current enrollment information for a student.

    IMPORTANT:
    Even though this tool is read-only, it exposes private student data.

    Therefore tool_loop.py still performs authorization based on the
    authenticated session.
    """

    record = _FAKE_ENROLLMENT_DB.get(
        args.student_id
    )

    if record is None:
        return {
            "error": "no enrollment record found for that student id"
        }

    return record


# ---------------------------------------------------------------------
# TOOL 2 — BOOK ADVISOR APPOINTMENT
# ---------------------------------------------------------------------

def book_advisor_appointment(
    args: BookingRequest,
    session: Session,
) -> dict:
    """
    SIDE-EFFECTING tool.

    Creates an academic advisor appointment.

    This function must only be reached AFTER:

        1. Pydantic argument validation.
        2. Session authorization.

    Idempotency is handled by tool_loop.py through the Session object.
    """

    booking = {
        "student_id": args.student_id,
        "advisor_id": args.advisor_id,
        "slot": str(args.slot),
        "reason": args.reason,
        "confirmation_id": (
            f"BOOK-{len(_FAKE_BOOKINGS) + 1:05d}"
        ),
    }

    _FAKE_BOOKINGS.append(booking)

    return booking


# ---------------------------------------------------------------------
# TOOL 3 — REQUEST TRANSCRIPT
# ---------------------------------------------------------------------

def request_transcript(
    args: TranscriptRequest,
    session: Session,
) -> dict:
    """
    SIDE-EFFECTING + privacy-sensitive tool.

    Requests an official transcript.

    A mailed/sealed transcript requires explicit third-party consent.

    This rule is enforced in Python rather than relying on the model
    to remember the rule.
    """

    if (
        args.delivery == "mailed_sealed"
        and not args.third_party_consent
    ):
        return {
            "error": (
                "third_party_consent is required "
                "for mailed/sealed delivery"
            )
        }

    record = {
        "student_id": args.student_id,
        "delivery": args.delivery,
        "request_id": (
            f"TRX-{len(_FAKE_TRANSCRIPT_REQUESTS) + 1:05d}"
        ),
    }

    _FAKE_TRANSCRIPT_REQUESTS.append(record)

    return record


# ---------------------------------------------------------------------
# TOOL 4 — ESCALATE TO REGISTRAR
# ---------------------------------------------------------------------

def escalate_to_registrar(
    reason: str,
    session: Session,
) -> dict:
    """
    TERMINAL tool.

    Hands the conversation to a human registrar.

    Once this tool runs, the tool loop stops.

    This prevents the model from continuing to make tool calls after
    escalation.
    """

    return {
        "escalated": True,
        "reason": reason,
        "contact": "registrar@example.edu",
    }


# ---------------------------------------------------------------------
# TOOL DESCRIPTION OBJECT
# ---------------------------------------------------------------------

@dataclass
class Tool:
    """
    Metadata describing one callable Munir tool.

    The risk classification is stored directly on the tool.

    That means the authorization layer does not need to guess whether
    a function is dangerous.
    """

    name: str

    risk: Literal[
        "read_only",
        "side_effecting",
        "terminal",
    ]

    description: str

    # None means the tool accepts no Pydantic argument object.
    args_model: type[BaseModel] | None

    # The actual Python function that executes the tool.
    fn: Callable

    def schema(self) -> dict:
        """
        Convert this tool into an OpenAI-compatible function definition.

        The model receives this schema and can therefore request the
        tool by name with structured arguments.

        IMPORTANT:
        Function-call parameters use the BARE JSON Schema.

        This differs from response_format for structured generation,
        which uses the wrapped:

            {
                "type": "json_schema",
                "json_schema": {...}
            }

        envelope.
        """

        if self.args_model is not None:
            # strict_schema() returns the response_format envelope.
            #
            # Function calling needs only the inner JSON Schema.
            parameters = strict_schema(
                self.args_model,
                self.name,
            )["json_schema"]["schema"]

        else:
            # escalate_to_registrar accepts only a simple reason string
            # handled by the tool loop.
            parameters = {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string"
                    }
                },
                "required": [
                    "reason"
                ],
                "additionalProperties": False,
            }

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }


# ---------------------------------------------------------------------
# MUNIR TOOL REGISTRY
# ---------------------------------------------------------------------
#
# This is the allow-list.
#
# If a model asks for a function that is NOT in this dictionary,
# tool_loop.py will reject it as an unknown tool.
# ---------------------------------------------------------------------

TOOLS: dict[str, Tool] = {

    "check_enrollment_status": Tool(
        name="check_enrollment_status",

        risk="read_only",

        description=(
            "Look up a student's current enrollment status, "
            "credit hours, and academic standing. "
            "Use ONLY for status lookups. "
            "Do NOT use this tool to answer general questions "
            "about admissions or university policy."
        ),

        args_model=EnrollmentStatusQuery,

        fn=check_enrollment_status,
    ),

    "book_advisor_appointment": Tool(
        name="book_advisor_appointment",

        risk="side_effecting",

        description=(
            "Book an academic advisor appointment for the CURRENT "
            "authenticated student. Never book an appointment for "
            "another student. The requested date must be a future "
            "Sunday-Thursday date. Do NOT use this tool to determine "
            "whether an appointment slot is available."
        ),

        args_model=BookingRequest,

        fn=book_advisor_appointment,
    ),

    "request_transcript": Tool(
        name="request_transcript",

        risk="side_effecting",

        description=(
            "Request an official transcript for the CURRENT "
            "authenticated student. Mailed/sealed delivery to a "
            "third party requires explicit consent. Do NOT use this "
            "tool for unofficial grade lookups."
        ),

        args_model=TranscriptRequest,

        fn=request_transcript,
    ),

    "escalate_to_registrar": Tool(
        name="escalate_to_registrar",

        risk="terminal",

        description=(
            "Hand the conversation to a human registrar. "
            "Use this for grade disputes, appeals, or situations "
            "that cannot safely be completed automatically."
        ),

        args_model=None,

        fn=escalate_to_registrar,
    ),
}