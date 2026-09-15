"""The router: the workhorse pattern, and the biggest cost lever in an assistant.

Munir's traffic is expected to be FAQ-heavy (fee/deadline/document
questions), with a smaller share of transactional requests (booking,
status, transcript) and a small escalation tail -- see DECISIONS.md
ADR-001. Routing is what makes the economics of that traffic mix work,
and because a routing change moves quality, it's eval-gated exactly like
a prompt change (Module 5).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from munir.llm.interfaces import LLMClient
from munir.observability import get_logger
from munir.pipeline.structured import StructuredExtractionFailed, extract_structured
from munir.prompts.registry import load_prompt

log = get_logger(__name__)

Intent = Literal["faq", "service", "escalate"]

#: Config, not code -- reviewed alongside the price sheet. None means
#: "not a model call at all": humans are not a route.
ROUTING_TABLE: dict[str, str | None] = {
    "faq": "munir-cheap",
    "service": "munir-default",
    "escalate": None,
}


class RouteVerdict(BaseModel):
    model_config = {"extra": "forbid"}

    intent: Intent


class IntentRouter:
    def __init__(
        self,
        classifier: LLMClient,
        *,
        model_alias: str = "munir-router",
        prompt_ref: str = "route_intent.v1",
        meter=None,
    ) -> None:
        self._classifier = classifier
        self._meter = meter
        self._model_alias = model_alias
        self._prompt = load_prompt(prompt_ref)

    @property
    def prompt_version(self) -> str:
        return self._prompt.ref

    def classify(self, text: str) -> Intent:
        try:
            verdict, outcome = extract_structured(
                self._classifier, RouteVerdict,
                system=self._prompt.render(),
                user=f"<student_message>\n{text}\n</student_message>",
                schema_name="route_verdict",
                model_alias=self._model_alias,
                temperature=0.0,
                max_tokens=20,
            )
        except StructuredExtractionFailed as exc:
            # A router that cannot decide sends traffic to the handler
            # that can cope with anything. Failing to the expensive route
            # costs money; failing to the cheap one costs correctness.
            # Choose deliberately.
            log.warning("router_unavailable", error="StructuredExtractionFailed", errors=str(exc.errors), fallback="service")
            return "service"
        except Exception as exc:  # noqa: BLE001
            log.warning("router_unavailable", error=type(exc).__name__, fallback="service")
            return "service"
        if self._meter is not None:
            for response in outcome.responses:
                self._meter.meter(
                    model_id=response.model_id, stage="router",
                    input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens,
                    cached_tokens=response.usage.cached_input_tokens,
                )
        log.info("routed", intent=verdict.intent, prompt_version=self._prompt.ref)
        return verdict.intent
