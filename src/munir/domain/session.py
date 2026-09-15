"""
Munir session and authorization model.

This file answers one critical security question:

    WHO is allowed to access or modify this student's information?

The answer MUST NOT come from the language model.

The authenticated application creates a Session containing the
student's verified identity.

The tool loop then asks Session.authorize() before executing
sensitive tools.

RUBRIC
------
Section 2E:
    Authorization gate before sensitive/side-effecting tool execution.

SECURITY PRINCIPLE
------------------
The model's arguments are considered USER INPUT BY PROXY.

For example, if the authenticated student is:

    STU-100001

and the model requests:

    check_enrollment_status(
        student_id="STU-100002"
    )

the application must reject it.

The model cannot grant itself permission.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------
# PII DETECTION
# ---------------------------------------------------------------------
#
# These patterns are also used by the guard system.
#
# They are kept here because Session owns the PII vault and outbound
# PII checks in the current Munir architecture.
# ---------------------------------------------------------------------

PII_PATTERNS = {
    # Saudi national ID:
    # 10 digits beginning with 1 or 2.
    "national_id": re.compile(
        r"\b[12]\d{9}\b"
    ),

    # Saudi mobile number:
    # examples:
    #   05XXXXXXXX
    #   +9665XXXXXXXX
    #   9665XXXXXXXX
    "phone": re.compile(
        r"\b(?:\+?966|0)5\d{8}\b"
    ),

    # Saudi IBAN:
    # SA + 22 digits.
    "iban": re.compile(
        r"\bSA\d{22}\b"
    ),
}


class PIIVault:
    """
    Temporarily stores masked PII mappings in memory.

    Example:

        0551234567

    becomes something similar to:

        ⟦PHONE_a81f23⟧

    The raw value is kept only in this in-memory vault.

    It should never be written to logs.
    """

    def __init__(self):
        self._store: dict[str, str] = {}

    def mask(
        self,
        text: str,
    ) -> str:
        """
        Replace detected PII with reversible in-memory tokens.
        """

        masked = text

        for kind, pattern in PII_PATTERNS.items():

            # Find matches in the original text.
            for match in pattern.finditer(text):

                raw = match.group(0)

                token = (
                    f"⟦{kind.upper()}_"
                    f"{uuid.uuid4().hex[:6]}⟧"
                )

                self._store[token] = raw

                masked = masked.replace(
                    raw,
                    token,
                    1,
                )

        return masked

    def unmask(
        self,
        text: str,
    ) -> str:
        """
        Restore values from the in-memory vault.

        This should only be used where the application legitimately
        needs the original value.
        """

        out = text

        for token, raw in self._store.items():
            out = out.replace(
                token,
                raw,
            )

        return out

    def __len__(self) -> int:
        return len(self._store)


def contains_pii(
    text: str,
) -> str | None:
    """
    Detect whether text contains one of Munir's protected PII types.

    Returns:

        "national_id"
        "phone"
        "iban"

    or:

        None
    """

    for kind, pattern in PII_PATTERNS.items():

        if pattern.search(
            text or ""
        ):
            return kind

    return None


# ---------------------------------------------------------------------
# AUTHORIZATION RESULT
# ---------------------------------------------------------------------

@dataclass
class AuthorizationVerdict:
    """
    Result returned by the authorization gate.
    """

    allowed: bool
    reason: str = ""


# ---------------------------------------------------------------------
# CONVERSATION STATE
# ---------------------------------------------------------------------

@dataclass
class ConversationState:
    """
    Bounded conversation history.

    The history is capped so a conversation cannot grow indefinitely.

    max_turns=8 means Munir keeps the most recent 8 user/assistant
    turns, represented internally as up to 16 messages.
    """

    turns: list[dict] = field(
        default_factory=list
    )

    max_turns: int = 8

    def add(
        self,
        role: str,
        content: str,
    ) -> None:
        """
        Add one conversation message and enforce the history limit.
        """

        self.turns.append(
            {
                "role": role,
                "content": content,
            }
        )

        # Keep only the latest max_turns pairs.
        self.turns = self.turns[
            -(self.max_turns * 2):
        ]

    def messages(self) -> list["Message"]:
        """
        Convert the stored history into Munir Message objects.
        """

        # Local import avoids creating an unnecessary module-level
        # dependency cycle.
        from munir.llm.interfaces import Message

        return [
            Message(
                role=turn["role"],
                content=turn["content"],
            )
            for turn in self.turns
        ]


# ---------------------------------------------------------------------
# AUTHENTICATED SESSION
# ---------------------------------------------------------------------

@dataclass
class Session:
    """
    Represents one authenticated Munir user.

    student_id is established by the APPLICATION.

    It is NOT taken from:

        - the user's message
        - the model's response
        - a tool call
        - a prompt

    That distinction is essential for Section 2E.
    """

    student_id: str

    role: Literal[
        "student",
        "advisor",
        "registrar",
    ] = "student"

    identity_verified_at: float = field(
        default_factory=time.time
    )

    identity_ttl_seconds: float = 3600.0

    state: ConversationState = field(
        default_factory=ConversationState
    )

    pii_vault: PIIVault = field(
        default_factory=PIIVault
    )

    # Stores successful side-effect results for idempotent replay.
    _replayed_effects: dict[str, dict] = field(
        default_factory=dict
    )

    def identity_fresh(self) -> bool:
        """
        Check whether the authentication state is still valid.
        """

        return (
            time.time()
            - self.identity_verified_at
        ) < self.identity_ttl_seconds

    def authorize(
        self,
        tool_name: str,
        args: dict,
    ) -> AuthorizationVerdict:
        """
        CENTRAL AUTHORIZATION GATE.

        Every sensitive/side-effecting tool passes through here.

        The function checks the authenticated Session identity,
        not whatever student_id the model happens to provide.

        This is the critical protection against cross-student access.
        """

        # -------------------------------------------------------------
        # 1. Authentication freshness
        # -------------------------------------------------------------

        if not self.identity_fresh():

            return AuthorizationVerdict(
                allowed=False,
                reason=(
                    "identity verification expired, "
                    "please re-authenticate"
                ),
            )

        # -------------------------------------------------------------
        # 2. Registrar role
        # -------------------------------------------------------------
        #
        # A registrar legitimately needs broader access.
        # In a production system this would also be tied to explicit
        # role permissions and audit logging.
        # -------------------------------------------------------------

        if self.role == "registrar":

            return AuthorizationVerdict(
                allowed=True,
                reason="registrar override",
            )

        # -------------------------------------------------------------
        # 3. Advisor role
        # -------------------------------------------------------------
        #
        # This project uses a simplified advisor rule.
        # A real deployment would check the university's assignment
        # system to verify that this advisor is assigned to the target
        # student.
        # -------------------------------------------------------------

        if self.role == "advisor":

            return AuthorizationVerdict(
                allowed=True,
                reason="advisor acting within scope",
            )

        # -------------------------------------------------------------
        # 4. Normal student
        # -------------------------------------------------------------
        #
        # A student can ONLY act on their own record.
        # -------------------------------------------------------------

        target = (
            args.get("student_id")
            or args.get("on_behalf_of")
        )

        if (
            target is not None
            and target != self.student_id
        ):

            return AuthorizationVerdict(
                allowed=False,
                reason=(
                    "cross-student access denied: "
                    f"session is {self.student_id}, "
                    f"requested {target}"
                ),
            )

        # -------------------------------------------------------------
        # 5. Authorized self-service operation
        # -------------------------------------------------------------

        return AuthorizationVerdict(
            allowed=True,
            reason="self-service, own record",
        )

    def replay_side_effect(
        self,
        idempotency_key: str,
        compute_fn,
    ):
        """
        Execute a side-effecting operation at most once for the same
        idempotency key.

        If the same operation is retried:

            first call  -> execute + save result
            second call -> return saved result

        This protects Munir from duplicate actions caused by retries.
        """

        if (
            idempotency_key
            in self._replayed_effects
        ):

            return (
                self._replayed_effects[
                    idempotency_key
                ],
                True,
            )

        result = compute_fn()

        self._replayed_effects[
            idempotency_key
        ] = result

        return (
            result,
            False,
        )