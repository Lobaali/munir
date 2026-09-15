"""The service workflow: the route that ACTS.

the steps are known
at design time (guard -> route -> tool loop -> reply), so the
decomposition is fixed and only the tool CHOICES are the model's.
"""

from __future__ import annotations

import time
from datetime import date

from munir.domain import directory
from munir.domain.session import Session
from munir.guards.input_guards import GuardedInput
from munir.llm.interfaces import LLMClient
from munir.observability import get_logger
from munir.pipeline.tool_loop import run_with_tools
from munir.pipeline.types import Reply
from munir.prompts.registry import load_prompt

log = get_logger(__name__)

SERVICE_PROMPT_REF = "service_workflow.v1"


class ServiceWorkflow:
    def __init__(
        self,
        client: LLMClient,
        *,
        model_alias: str = "munir-default",
        prompt_ref: str = SERVICE_PROMPT_REF,
        meter=None,
    ) -> None:
        self._client = client
        self._model_alias = model_alias
        self._prompt = load_prompt(prompt_ref)
        self._meter = meter

    @property
    def prompt_version(self) -> str:
        return self._prompt.ref

    def run(self, guarded: GuardedInput, session: Session) -> Reply:
        t0 = time.perf_counter()
        language = guarded.language
        directory_text = directory.rendered_directory("ar" if language == "ar" else "en")
        today = date.today().isoformat()
        system = self._prompt.render(service_directory=directory_text, today=today, student_id=session.student_id)

        outcome = run_with_tools(
            self._client, system=system, user=guarded.text, session=session,
            model_alias=self._model_alias, meter=self._meter,
        )

        escalated = any(call.get("risk") == "terminal" for call in outcome["tool_calls"])
        bound_hit = outcome["iterations"] >= 6 and not outcome["tool_calls"]

        return Reply(
            text=outcome["text"], intent="service", language=language, model_id=self._model_alias,
            prompt_version=self._prompt.ref, tool_calls=outcome["tool_calls"],
            tool_iterations=outcome["iterations"], escalated=escalated,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
