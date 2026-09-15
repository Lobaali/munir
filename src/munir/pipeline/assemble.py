"""The whole Munir request path as five named, individually testable stages.

    guard_input | route_intent | (faq | service | escalate) | guard_output
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from munir.domain.session import Session
from munir.guards.input_guards import GuardedInput, InputGuard
from munir.guards.output_guards import OutputGuard
from munir.guards.refusals import refusal_for
from munir.observability import get_logger
from munir.pipeline.faq import FAQHandler
from munir.pipeline.router import IntentRouter
from munir.pipeline.service import ServiceWorkflow
from munir.pipeline.types import Reply
from munir.tools.registry import escalate_to_registrar

log = get_logger(__name__)


class EscalationHandler:
    """Humans are not a model call -- so this stage makes none."""

    def handoff(self, guarded: GuardedInput, session: Session) -> Reply:
        record = escalate_to_registrar("student requested or was routed to escalation", session)
        language = guarded.language
        text = (
            "سأحوّلك إلى مكتب القبول والتسجيل الآن، وسيصلك رد في أقرب وقت."
            if language == "ar"
            else "I'm connecting you with the Registrar's Office now; they'll follow up shortly."
        )
        return Reply(
            text=text, intent="escalate", language=language, escalated=True,
            tool_calls=[{"tool": "escalate_to_registrar", "risk": "terminal", "args": {}, "result": record}],
        )


@dataclass
class Dependencies:
    input_guard: InputGuard
    router: IntentRouter
    faq_handler: FAQHandler
    service_workflow: ServiceWorkflow
    output_guard: OutputGuard
    escalation: EscalationHandler


class Munir:
    """A thin façade over the pipeline -- what a notebook, a script, or
    the eval harness calls."""

    def __init__(self, deps: Dependencies) -> None:
        self.deps = deps

    # --- Stage 1: guard_input -----------------------------------------
    def _guard_input(self, text: str, session: Session) -> GuardedInput | Reply:
        guarded = self.deps.input_guard.check(text, session.student_id, session.pii_vault)
        if guarded.blocked:
            return Reply(
                text=guarded.refusal or refusal_for("off_scope", guarded.language),
                intent="refused", language=guarded.language, blocked=True,
                guard_layer=guarded.verdict.layer, guard_category=guarded.verdict.category,
                latency_ms=guarded.verdict.latency_ms,
            )
        return guarded

    # --- Stage 2: route_intent -----------------------------------------
    def _route_intent(self, guarded: GuardedInput) -> str:
        return self.deps.router.classify(guarded.text)

    # --- Stage 3: faq | service | escalate -----------------------------
    def _dispatch(self, intent: str, guarded: GuardedInput, session: Session) -> Reply:
        if intent == "faq":
            return self.deps.faq_handler.answer(guarded, session)
        if intent == "service":
            return self.deps.service_workflow.run(guarded, session)
        return self.deps.escalation.handoff(guarded, session)

    # --- Stage 4: guard_output ------------------------------------------
    def _guard_output(self, reply: Reply) -> Reply:
        if reply.blocked:
            return reply
        text, verdict = self.deps.output_guard.apply(reply.text)
        reply.output_guard_category = verdict.category
        if not verdict.allowed:
            log.warning("output_guard_blocked", category=verdict.category)
            reply.text = text
            reply.blocked = True
        return reply

    # --- Stage 5: remember, and the public entry point -------------------
    def ask(self, text: str, session: Session, *, remember: bool = True) -> Reply:
        from munir.observability import new_trace_id

        new_trace_id()  # every log line and cost record for this request shares this id
        t0 = time.perf_counter()

        guarded_or_reply = self._guard_input(text, session)
        if isinstance(guarded_or_reply, Reply):
            guarded_or_reply.latency_ms = (time.perf_counter() - t0) * 1000
            return guarded_or_reply
        guarded = guarded_or_reply

        intent = self._route_intent(guarded)
        reply = self._dispatch(intent, guarded, session)
        reply.intent = intent
        reply = self._guard_output(reply)
        reply.latency_ms = (time.perf_counter() - t0) * 1000

        if remember and not reply.blocked:
            session.state.add("user", text)
            session.state.add("assistant", reply.text)

        return reply