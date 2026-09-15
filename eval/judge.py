"""The calibrated LLM-as-judge for Munir.

Rubric coverage
---------------
- 4.4 A written rubric evaluates one quality dimension at a time.
- 4.5 The same judge is used by the calibration script so we can measure
      agreement with human labels and Cohen's kappa.

Important design decision
-------------------------
The judge does NOT replace deterministic safety assertions.  A judge can
provide a useful quality signal, but safety properties such as "the answer
must not contain an unapproved fee" are cheaper and more reliable in Python.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from munir.domain.directory import rendered_directory
from munir.llm.interfaces import LLMClient
from munir.pipeline.structured import extract_structured
from munir.prompts.registry import load_prompt


class JudgeVerdict(BaseModel):
    """Strict wire contract for the judge's answer."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(description="Only 0.0, 0.5, or 1.0")
    evidence: str = Field(min_length=1, max_length=500)

    @field_validator("score")
    @classmethod
    def score_must_be_allowed(cls, value: float) -> float:
        """Keep the judge on the rubric's three-point scale."""
        if value not in {0.0, 0.5, 1.0}:
            raise ValueError("score must be exactly 0.0, 0.5, or 1.0")
        return value


def judge_case(
    client: LLMClient,
    *,
    answer: str,
    language: str,
    rubric_text: str,
    model_alias: str,
) -> dict:
    """Score one answer against one rubric dimension.

    The trusted directory is the context.  The rubric is supplied separately
    so changing the judge criterion is a versioned prompt/rubric change,
    rather than a code change.
    """
    directory = rendered_directory("ar" if language == "ar" else "en")
    system = load_prompt("judge_groundedness.v1").render()
    user = (
        f"<rubric>\n{rubric_text}\n</rubric>\n\n"
        f"<context>\n{directory}\n</context>\n\n"
        f"<answer>\n{answer}\n</answer>"
    )

    verdict, outcome = extract_structured(
        client,
        JudgeVerdict,
        system=system,
        user=user,
        schema_name="groundedness_judge_verdict",
        model_alias=model_alias,
        temperature=0.0,
        max_tokens=200,
    )

    return {
        "score": verdict.score,
        "evidence": verdict.evidence,
        "model_id": outcome.responses[-1].model_id if outcome.responses else "",
        "attempts": outcome.attempts,
    }
