"""
Controlled refusal messages for Munir.

SECTION 3 — Prompt Pipeline & Guardrails

Rubric covered:
- 3B: deterministic injection handling
- 3D: safe outbound responses

Why this file exists:
A security guard should not let the model invent its own refusal.
Instead, Munir uses predefined bilingual messages.

The refusal never echoes the original user input.
This prevents an attack or piece of PII from being repeated back
to the student.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# SECTION 3D — Controlled outbound responses
# ---------------------------------------------------------------------------
# Each guard category has an English and Arabic response.
#
# These are deliberately short and do not contain:
# - the offending input
# - system prompts
# - internal configuration
# - PII
# ---------------------------------------------------------------------------

REFUSALS: dict[str, dict[str, str]] = {
    "injection_pattern": {
        "ar": "أستطيع مساعدتك في خدمات الحرم الجامعي. كيف يمكنني خدمتك اليوم؟",
        "en": "I can help with campus services. How can I assist you today?",
    },

    "cross_student_social_engineering": {
        "ar": (
            "لا أستطيع تنفيذ طلبات نيابة عن طالب آخر. "
            "يمكنني مساعدتك في شؤونك الخاصة فقط."
        ),
        "en": (
            "I can't act on another student's behalf. "
            "I can only help with your own record."
        ),
    },

    "system_prompt_leak": {
        "ar": (
            "لا أستطيع مشاركة إعدادات النظام. "
            "كيف يمكنني مساعدتك في خدمات الحرم الجامعي؟"
        ),
        "en": (
            "I can't share system configuration. "
            "How can I help with campus services?"
        ),
    },

    "pii_outbound": {
        "ar": (
            "لا أستطيع عرض بيانات شخصية في هذه المحادثة. "
            "يمكنك مراجعتها في بوابة الطالب."
        ),
        "en": (
            "I can't show personal data in this conversation. "
            "You can view it in the student portal."
        ),
    },

    "off_scope": {
        "ar": (
            "هذا خارج نطاق خدمتي — أنا هنا للمساعدة في خدمات الحرم الجامعي. "
            "يمكنك سؤالي عن الرسوم أو المواعيد أو حالة التسجيل."
        ),
        "en": (
            "That's outside what I can help with — I'm here for campus services. "
            "Try asking about fees, deadlines, or your enrollment status."
        ),
    },

    "too_long": {
        "ar": "الرسالة طويلة جداً. هل يمكنك تلخيص طلبك؟",
        "en": (
            "That message is longer than I can take in. "
            "Could you summarise your request?"
        ),
    },

    "unavailable": {
        "ar": (
            "أعتذر، لا أستطيع الإجابة في هذه اللحظة. "
            "يرجى مراجعة مكتب القبول والتسجيل أو المحاولة لاحقاً."
        ),
        "en": (
            "I can't answer right now. Please try again shortly, "
            "or contact the Registrar's Office."
        ),
    },
}


def refusal_for(category: str, language: str = "en") -> str:
    """
    Return the predefined refusal for a guard category.

    SECTION 3:
    Keeping refusal generation deterministic means a blocked request
    does not need another LLM call.

    Unknown categories safely fall back to the generic off-scope response.
    """

    template = REFUSALS.get(category) or REFUSALS["off_scope"]

    # If an unexpected language is supplied, English is the safe fallback.
    return template.get(language, template["en"])