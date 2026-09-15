"""Structured ticket extraction -- Module 3's headline, in a few lines of
glue over pipeline/structured.py.
"""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from munir.llm.interfaces import LLMClient
from munir.pipeline.structured import StructuredExtractionFailed, StructuredOutcome, extract_structured
from munir.prompts.registry import load_prompt

T = TypeVar("T", bound=BaseModel)


class ExtractionFailed(StructuredExtractionFailed):
    """Raised when both attempts fail. Carries the raw output to human review."""


def extract_ticket(
    client: LLMClient,
    schema_model: type[T],
    schema_name: str,
    student_message: str,
    *,
    prompt_ref: str = "extract_ticket.v1",
    model_alias: str = "munir-extract",
) -> tuple[T, StructuredOutcome]:
    prompt = load_prompt(prompt_ref)
    try:
        return extract_structured(
            client, schema_model,
            system=prompt.render(),
            user=f"<student_message>\n{student_message}\n</student_message>",
            schema_name=schema_name,
            model_alias=model_alias,
            temperature=0.0,
            max_tokens=400,
        )
    except StructuredExtractionFailed as exc:
        raise ExtractionFailed(exc.raw, exc.errors, exc.attempts) from exc


class CorpusReport(BaseModel):
    """The numbers Module 3 asks for, plus the per-language split."""

    total: int = 0
    first_try: int = 0
    after_repair: int = 0
    escalated: int = 0
    by_language: dict[str, dict[str, int]] = {}

    def record(self, language: str, outcome: str) -> None:
        self.total += 1
        setattr(self, outcome, getattr(self, outcome) + 1)
        bucket = self.by_language.setdefault(language, {"total": 0, "first_try": 0, "after_repair": 0, "escalated": 0})
        bucket["total"] += 1
        bucket[outcome] += 1

    @property
    def first_try_rate(self) -> float:
        return self.first_try / self.total if self.total else 0.0

    @property
    def after_repair_rate(self) -> float:
        return (self.first_try + self.after_repair) / self.total if self.total else 0.0

    def language_rate(self, language: str) -> tuple[float, float]:
        bucket = self.by_language.get(language)
        if not bucket or not bucket["total"]:
            return 0.0, 0.0
        return (
            bucket["first_try"] / bucket["total"],
            (bucket["first_try"] + bucket["after_repair"]) / bucket["total"],
        )

    def render(self) -> str:
        lines = [
            f"{self.total} messages | first-try pass: {self.first_try}/{self.total} "
            f"({self.first_try_rate:.0%}) | after repair: "
            f"{self.first_try + self.after_repair}/{self.total} ({self.after_repair_rate:.0%}) | "
            f"escalated: {self.escalated}"
        ]
        parts = []
        for language in sorted(self.by_language):
            bucket = self.by_language[language]
            first, after = self.language_rate(language)
            parts.append(
                f"{language} {bucket['first_try']}/{bucket['total']} ({first:.0%}) "
                f"-> {bucket['first_try'] + bucket['after_repair']}/{bucket['total']} ({after:.0%})"
            )
        lines.append("   by language: " + " | ".join(parts))
        return "\n".join(lines)
