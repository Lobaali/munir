"""
Munir input guard pipeline.

SECTION 3 — Prompt Pipeline & Guardrails

Rubric covered:
- 3B: deterministic English + Arabic injection detection
- 3C: Saudi PII masking before model/log
- 3E: named, individually testable guard stage

The input guard runs BEFORE the router and before normal LLM processing.

Pipeline:

    raw student message
            |
            v
    deterministic checks
            |
            v
       PII masking
            |
            v
    optional classifier
            |
            v
       GuardedInput
            |
            v
       router / LLM


Important security rule:
The original message is never sent to the LLM after PII masking.
The masked `text` field is the value that downstream model stages use.
"""

from __future__ import annotations

import hashlib
import re
import time
import unicodedata
from typing import Literal

from pydantic import BaseModel

from munir.domain.session import PIIVault
from munir.guards.refusals import refusal_for
from munir.llm.interfaces import LLMClient
from munir.observability import get_logger
from munir.prompts.registry import load_prompt


log = get_logger(__name__)


# ===========================================================================
# SECTION 3B — Deterministic bilingual injection detection
# ===========================================================================

# English injection patterns.
#
# These patterns target common attempts to:
# - override system instructions
# - reveal prompts
# - change the assistant's role
# - bypass authorization
#
# They are intentionally deterministic so they do not require an LLM call.

ENGLISH_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\bignore\s+(all\s+)?(your\s+|the\s+)?"
        r"(previous|prior|above|earlier)\s+"
        r"(instructions|rules|prompts?)",
        re.I,
    ),

    re.compile(
        r"\bdisregard\s+(all\s+)?(your\s+|the\s+)?"
        r"(previous|prior|above)\b",
        re.I,
    ),

    re.compile(
        r"\b(reveal|show|print|repeat|output|display|tell me)\b"
        r".{0,40}"
        r"\b(system prompt|your instructions|developer message|"
        r"initial prompt|your rules)\b",
        re.I,
    ),

    re.compile(
        r"\byou are (now|no longer)\b.{0,40}"
        r"\b(assistant|admin|developer|unrestricted)\b",
        re.I,
    ),

    re.compile(
        r"\b(new|updated|revised)\s+(system\s+)?instructions?\s*:",
        re.I,
    ),

    re.compile(
        r"</?(system|instructions?)>",
        re.I,
    ),

    re.compile(
        r"\bact as (if you are |a )?(the )?"
        r"(system|developer|administrator)\b",
        re.I,
    ),

    re.compile(
        r"\bas the (system )?administrator\b",
        re.I,
    ),

    re.compile(
        r"\boverride the (authorization|authorisation)\b",
        re.I,
    ),

    re.compile(
        r"\brepeat everything above\b",
        re.I,
    ),

    re.compile(
        r"\btranslate your (system )?instructions\b",
        re.I,
    ),
]


# Arabic injection patterns.
#
# Munir supports Arabic, so an English-only security layer would leave
# a significant part of the attack surface uncovered.

ARABIC_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"تجاهل\s*(كل\s*)?(ال)?"
        r"(تعليمات|أوامر|الأوامر|التعليمات)\s*"
        r"(السابقة|السابق)?"
    ),

    re.compile(
        r"(اطبع|أظهر|إظهار|اعرض|عرض|كشف|أعد|كرر|أخبرني)"
        r"\s*.{0,25}"
        r"(تعليمات النظام|التعليمات السابقة|البرومبت|"
        r"تعليماتك|قواعدك|إعداداتك|الرمز المرجعي)"
    ),

    re.compile(
        r"أنت\s*الآن\s*(مساعد|نظام|مطور|مدير)"
    ),

    re.compile(
        r"تصرف\s*ك(نظام|مطور|مدير|مساعد)"
    ),

    re.compile(
        r"تظاهر\s*بأن"
    ),

    re.compile(
        r"تجاوز\s*(التحقق|الصلاحية)"
    ),

    re.compile(
        r"أعد\s*.{0,20}ما\s*ورد\s*أعلاه"
    ),
]


# Combine the two lists so the deterministic layer can test both languages.
INJECTION_PATTERNS = (
    ENGLISH_INJECTION_PATTERNS
    + ARABIC_INJECTION_PATTERNS
)


# ---------------------------------------------------------------------------
# Legitimate context exception
# ---------------------------------------------------------------------------
# A normal student might ask:
#
# "What are the steps to apply for admission?"
#
# The word "steps" or "requirements" should not automatically be treated
# as an injection.
#
# This carve-out reduces false positives for normal campus-service questions.
# ---------------------------------------------------------------------------

LEGITIMATE_CONTEXT = re.compile(
    r"\b(steps|documents|requirements)\s+(for|to)\s+"
    r"(apply|applying|enroll|enrolling|book|booking|request|requesting)",
    re.I,
)


# ===========================================================================
# SECTION 3B — Input normalization
# ===========================================================================

# Control characters, zero-width characters, and bidirectional characters
# can be abused to hide malicious text from simple string matching.

_CONTROL_RANGES = [
    (0x00, 0x08),
    (0x0B, 0x0C),
    (0x0E, 0x1F),
    (0x7F, 0x9F),
    (0x200B, 0x200F),
    (0x202A, 0x202E),
    (0x2066, 0x2069),
    (0xFEFF, 0xFEFF),
]


CONTROL_CHARS = re.compile(
    "["
    + "".join(f"{chr(a)}-{chr(b)}" for a, b in _CONTROL_RANGES)
    + "]"
)


# Language detection is intentionally lightweight and deterministic.
ARABIC = re.compile(r"[\u0600-\u06FF]")
LATIN = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    """
    Detect whether the message is primarily Arabic or English.

    SECTION 3B:
    The result is used to select the correct bilingual refusal.
    """

    arabic_count = len(ARABIC.findall(text))
    english_count = len(LATIN.findall(text))

    if arabic_count and english_count:
        return (
            "ar"
            if arabic_count / (arabic_count + english_count) >= 0.5
            else "en"
        )

    return "ar" if arabic_count else "en"


def match_variants(text: str) -> list[str]:
    """
    Produce normalized variants for deterministic injection matching.

    Why two variants?

    Example:

        Ignore<ZERO-WIDTH>previous<ZERO-WIDTH>instructions

    A naive detector may miss this.

    We therefore test:
    1. the version with invisible characters removed
    2. the version where invisible characters become spaces
    """

    folded = unicodedata.normalize("NFKC", text)

    removed = CONTROL_CHARS.sub("", folded)
    spaced = CONTROL_CHARS.sub(" ", folded)

    return [removed, spaced]


def _hash(text: str) -> str:
    """
    Return a short SHA-256 fingerprint.

    SECTION 3C:
    Logs should identify a payload without storing the sensitive payload
    itself.
    """

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:16]


# ===========================================================================
# Structured guard result models
# ===========================================================================

class GuardVerdict(BaseModel):
    """
    Structured result of a guard decision.

    This makes guard behavior easy to test and inspect.
    """

    allowed: bool = True

    layer: Literal[
        "deterministic",
        "pii",
        "classifier",
        "none",
    ] = "none"

    category: str = "ok"

    latency_ms: float = 0.0

    payload_sha256: str = ""


class GuardClassification(BaseModel):
    """
    Structured output expected from the optional Layer 3 classifier.

    SECTION 3B:
    The classifier is not allowed to return arbitrary text.
    """

    model_config = {"extra": "forbid"}

    is_attack: bool

    category: Literal[
        "injection",
        "cross_student_social_engineering",
        "prompt_leak",
        "none",
    ]


class GuardedInput(BaseModel):
    """
    Input after the security pipeline.

    `original`
        Original application input, retained only inside the application.

    `text`
        Normalized and PII-masked text.

    Downstream LLM stages MUST use `text`, not `original`.
    """

    original: str

    text: str

    language: str

    verdict: GuardVerdict

    refusal: str | None = None

    @property
    def blocked(self) -> bool:
        return not self.verdict.allowed


# ===========================================================================
# SECTION 3B — Deterministic layer
# ===========================================================================

def deterministic_checks(
    text: str,
    *,
    max_chars: int = 4000,
) -> GuardVerdict | None:
    """
    Run the cheap deterministic security checks.

    Returns:
        GuardVerdict if the request should be blocked.
        None if it should continue to the next layer.
    """

    # -----------------------------------------------------------------------
    # Length protection
    # -----------------------------------------------------------------------
    if len(text) > max_chars:
        return GuardVerdict(
            allowed=False,
            layer="deterministic",
            category="too_long",
            payload_sha256=_hash(text),
        )

    # -----------------------------------------------------------------------
    # Normalize the message before injection matching.
    # -----------------------------------------------------------------------
    variants = match_variants(text)

    # -----------------------------------------------------------------------
    # Legitimate campus-service questions should not be blocked simply
    # because they contain words such as "steps" or "requirements".
    # -----------------------------------------------------------------------
    if any(
        LEGITIMATE_CONTEXT.search(candidate)
        for candidate in variants
    ):
        return None

    # -----------------------------------------------------------------------
    # English + Arabic injection detection.
    # -----------------------------------------------------------------------
    for candidate in variants:
        for pattern in INJECTION_PATTERNS:
            if pattern.search(candidate):
                return GuardVerdict(
                    allowed=False,
                    layer="deterministic",
                    category="injection_pattern",
                    payload_sha256=_hash(text),
                )

    return None


# ===========================================================================
# SECTION 3C — Full input guard
# ===========================================================================

class InputGuard:
    """
    The complete Munir input security wall.

    SECTION 3E:
    This class is a named, independently testable pipeline stage.

    The order is important:

        deterministic checks
                 ↓
             PII mask
                 ↓
          optional classifier

    The model only receives the masked value.
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        max_chars: int = 4000,
        classifier_enabled: bool = True,
        classifier_alias: str = "munir-guard",
        prompt_ref: str = "guard_classifier.v1",
        meter=None,
    ) -> None:

        self._client = client
        self._meter = meter

        self._max_chars = max_chars

        self._classifier_enabled = (
            classifier_enabled
            and client is not None
        )

        self._classifier_alias = classifier_alias

        self._prompt_ref = prompt_ref

        # Useful for evaluation and later performance measurements.
        self.last_layer_timings: dict[str, float] = {}

    def check(
        self,
        text: str,
        student_id: str,
        pii_vault: PIIVault,
    ) -> GuardedInput:
        """
        Run the complete input guard.

        SECTION 3B + 3C:
        No normal model processing occurs until this method finishes.
        """

        start_time = time.perf_counter()

        language = detect_language(text)

        # ===================================================================
        # LAYER 1 — deterministic security checks
        # ===================================================================

        verdict = deterministic_checks(
            text,
            max_chars=self._max_chars,
        )

        self.last_layer_timings = {
            "deterministic": (
                time.perf_counter() - start_time
            ) * 1000
        }

        if verdict is not None:
            verdict.latency_ms = (
                self.last_layer_timings["deterministic"]
            )

            log.warning(
                "guard_blocked",
                layer=verdict.layer,
                category=verdict.category,
                payload_sha256=verdict.payload_sha256,
            )

            return GuardedInput(
                original=text,
                text=text,
                language=language,
                verdict=verdict,
                refusal=refusal_for(
                    verdict.category,
                    language,
                ),
            )

        # ===================================================================
        # LAYER 2 — PII masking
        # ===================================================================
        #
        # IMPORTANT:
        # This happens BEFORE the classifier and before normal model calls.
        #
        # The PII vault keeps the real values in application memory.
        # Downstream processing uses only the masked string.
        # ===================================================================

        pii_start = time.perf_counter()

        masked = pii_vault.mask(text)

        self.last_layer_timings["pii"] = (
            time.perf_counter() - pii_start
        ) * 1000

        # ===================================================================
        # LAYER 3 — optional semantic classifier
        # ===================================================================

        if self._classifier_enabled:

            classifier_start = time.perf_counter()

            category = self._classify(
                masked,
                student_id,
            )

            self.last_layer_timings["classifier"] = (
                time.perf_counter() - classifier_start
            ) * 1000

            if category != "none":

                blocked = GuardVerdict(
                    allowed=False,
                    layer="classifier",
                    category=category,
                    latency_ms=(
                        time.perf_counter()
                        - start_time
                    ) * 1000,
                    payload_sha256=_hash(masked),
                )

                log.warning(
                    "guard_blocked",
                    layer=blocked.layer,
                    category=blocked.category,
                    payload_sha256=blocked.payload_sha256,
                )

                return GuardedInput(
                    original=text,
                    text=masked,
                    language=language,
                    verdict=blocked,
                    refusal=refusal_for(
                        category,
                        language,
                    ),
                )

        # ===================================================================
        # Input passed all guard layers.
        # ===================================================================

        return GuardedInput(
            original=text,
            text=masked,
            language=language,
            verdict=GuardVerdict(
                allowed=True,
                latency_ms=(
                    time.perf_counter()
                    - start_time
                ) * 1000,
            ),
        )

    def _classify(
        self,
        text: str,
        student_id: str,
    ) -> str:
        """
        Run the optional semantic security classifier.

        The classifier itself uses:
        - a versioned prompt artifact
        - structured output
        - the same LLM boundary used elsewhere

        It never receives the raw PII-containing message.
        """

        # Local import avoids unnecessary dependency loading until the
        # classifier is actually needed.
        from munir.pipeline.structured import (
            StructuredExtractionFailed,
            extract_structured,
        )

        # ---------------------------------------------------------------
        # SECTION 3A:
        # Load classifier instructions from a versioned file.
        # ---------------------------------------------------------------
        prompt = load_prompt(self._prompt_ref)

        try:
            instance, outcome = extract_structured(
                self._client,
                GuardClassification,

                # The student ID is application-controlled context.
                system=prompt.render(
                    student_id=student_id,
                ),

                # IMPORTANT:
                # This is already PII-masked.
                user=text,

                schema_name="guard_classification",

                model_alias=self._classifier_alias,

                # SECTION 1 / architecture:
                # Never send an unbounded request.
                max_tokens=20,
            )

        except StructuredExtractionFailed as exc:

            log.warning(
                "guard_classifier_no_valid_response",
                errors=str(exc.errors),
            )

            # The deterministic wall already passed.
            # A classifier outage should not make ordinary campus service
            # unavailable.
            return "none"

        except Exception as exc:
            log.error(
                "guard_classifier_unavailable",
                error=type(exc).__name__,
            )

            return "none"

        # Cost accounting is used later by Section 5.
        # We keep it optional so Section 3 remains usable independently.
        if self._meter is not None:
            for response in outcome.responses:
                self._meter.meter(
                    model_id=response.model_id,
                    stage="input_guard",
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    cached_tokens=response.usage.cached_input_tokens,
                )

        if instance.is_attack:
            return instance.category

        return "none"