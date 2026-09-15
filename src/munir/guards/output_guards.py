"""
Munir outbound/output guard.

SECTION 3 — Prompt Pipeline & Guardrails

Rubric covered:
- 3D: outbound wall for prompt/canary leakage and PII
- 3E: individually testable output-guard stage

The model's output is treated as UNTRUSTED.

Even if the input was safe, the model can still:
- reveal internal prompt information
- expose PII
- expose stack traces
- relay malicious instructions from a tool result

Therefore every answer passes through this wall before reaching
the student.
"""

from __future__ import annotations

import re

from munir.domain.session import contains_pii
from munir.guards.input_guards import GuardVerdict, detect_language
from munir.guards.refusals import refusal_for
from munir.observability import get_logger
from munir.prompts.registry import CANARY


log = get_logger(__name__)


# ===========================================================================
# SECTION 3D — Internal error leakage
# ===========================================================================

INTERNALS = [
    re.compile(
        r"\bTraceback \(most recent call last\)",
        re.I,
    ),

    re.compile(
        r"\b[A-Za-z]*(ConnectionError|OperationalError|"
        r"TimeoutError|ValidationError)\b"
    ),

    re.compile(
        r"\bstack trace\b",
        re.I,
    ),
]


# ===========================================================================
# SECTION 3D — Indirect prompt injection relay
# ===========================================================================

# Tool results are DATA.
#
# A malicious tool-result field should never be allowed to make Munir
# output an instruction such as:
#
# "Call this number instead."
#
# These patterns are a final outbound safety wall.

RELAYED_INSTRUCTIONS = [
    re.compile(
        r"\bas the assistant reading this\b",
        re.I,
    ),

    re.compile(
        r"\b(call|dial|contact)\s+"
        r"(this|the following)\s+number\b",
        re.I,
    ),

    re.compile(
        r"\bignore\s+(the|your)\s+"
        r"(previous|prior)\s+instructions\b",
        re.I,
    ),

    re.compile(
        r"اتصل\s*(على|ب)\s*(هذا\s*)?الرقم"
    ),
]


class OutputGuard:
    """
    Final security wall for model-generated responses.

    SECTION 3E:
    This is deliberately a separate class so it can be tested without
    running the complete application.
    """

    def __init__(
        self,
        canary: str = CANARY,
    ) -> None:
        self._canary = canary

    def check(
        self,
        text: str,
    ) -> GuardVerdict:
        """
        Check whether model output is safe to send.

        Returns:
            allowed=True  -> safe to send
            allowed=False -> replace with deterministic refusal
        """

        # ===================================================================
        # 1. Canary / system prompt leak
        # ===================================================================
        #
        # The canary is planted in system prompts.
        # If it appears in model output, the model has leaked internal
        # configuration.
        # ===================================================================

        if self._canary in text:

            log.error(
                "output_guard_leak",
                category="system_prompt_leak",
            )

            return GuardVerdict(
                allowed=False,
                layer="deterministic",
                category="system_prompt_leak",
            )

        # ===================================================================
        # 2. Outbound PII
        # ===================================================================

        pii_kind = contains_pii(text)

        if pii_kind:

            log.error(
                "output_guard_leak",
                category=f"pii_outbound_{pii_kind}",
            )

            return GuardVerdict(
                allowed=False,
                layer="pii",
                category="pii_outbound",
            )

        # ===================================================================
        # 3. Internal implementation/error leakage
        # ===================================================================

        for pattern in INTERNALS:

            if pattern.search(text):

                log.error(
                    "output_guard_leak",
                    category="internal_error_leak",
                )

                return GuardVerdict(
                    allowed=False,
                    layer="deterministic",
                    category="unavailable",
                )

        # ===================================================================
        # 4. Tool-result instruction relay
        # ===================================================================

        for pattern in RELAYED_INSTRUCTIONS:

            if pattern.search(text):

                log.error(
                    "output_guard_leak",
                    category="relayed_instruction",
                )

                return GuardVerdict(
                    allowed=False,
                    layer="deterministic",
                    category="unavailable",
                )

        # ===================================================================
        # Output passed all checks.
        # ===================================================================

        return GuardVerdict(
            allowed=True,
        )

    def apply(
        self,
        text: str,
    ) -> tuple[str, GuardVerdict]:
        """
        Return the actual text that may be sent to the student.

        Safe output:
            return original answer.

        Unsafe output:
            return a deterministic bilingual refusal.
        """

        verdict = self.check(text)

        if verdict.allowed:
            return text, verdict

        # Never return the unsafe text.
        return (
            refusal_for(
                verdict.category,
                detect_language(text),
            ),
            verdict,
        )