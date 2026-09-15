"""FAQ stage: the main Module 6 optimisation target.

The order matters:
1. Build a stable system prefix.
2. Put volatile request data after that prefix.
3. Check the exact response cache.
4. Call the model only on a cache miss.
5. Meter the real provider usage.
6. Optionally cascade from the cheap model to the flagship model.

Rubric coverage:
- 5.1 per-request usage -> CostRecord
- 5.2 prompt-cache usage observed through cached_input_tokens
- 5.3 stable-prefix / volatile-tail discipline
- 5.4 answer-complete exact response cache key
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime

from munir.caching.response_cache import ResponseCache
from munir.domain import directory
from munir.domain.session import Session
from munir.guards.input_guards import GuardedInput
from munir.llm.interfaces import LLMClient, LLMRequest, Message
from munir.observability import get_logger
from munir.observability.cost import CostMeter
from munir.pipeline.groundedness import unsupported_amounts
from munir.pipeline.types import Reply
from munir.prompts.registry import PromptArtifact, load_prompt

log = get_logger(__name__)

DEFAULT_FAQ_PROMPT = os.environ.get(
    "MUNIR_FAQ_PROMPT",
    "answer_faq.v2",
)


def build_faq_messages(
    prompt: PromptArtifact,
    service_directory: str,
    history: list[Message],
    user_text: str,
    *,
    today: str | None = None,
) -> list[Message]:
    """Build messages with a stable prefix and a volatile tail.

    v2 deliberately has no timestamp in the system prompt. A timestamp at
    the top would change the cacheable prefix every second. The current date
    instead travels with the user turn, which is volatile by nature.
    """

    variables = {"service_directory": service_directory}

    if "now" in prompt.required_vars:
        # Kept only for the historical v1 prompt so we can measure the
        # before/after cache effect. Do not use v1 for production.
        variables["now"] = datetime.now().isoformat(
            timespec="seconds"
        )

    system = prompt.render(**variables)

    turn = f"<student_message>\n{user_text}\n</student_message>"

    if "now" not in prompt.required_vars:
        # Stable system prompt above; volatile date below.
        turn = (
            f"Today is {today or date.today().isoformat()}.\n"
            f"{turn}"
        )

    return [
        Message(role="system", content=system),
        *history,
        Message(role="user", content=turn),
    ]


class FAQHandler:
    def __init__(
        self,
        client: LLMClient,
        *,
        model_alias: str = "munir-default",
        prompt_ref: str = DEFAULT_FAQ_PROMPT,
        cache: ResponseCache | None = None,
        meter: CostMeter | None = None,
        cascade_enabled: bool = False,
        cascade_alias: str = "munir-cheap",
        flagship_alias: str = "munir-flagship",
    ) -> None:
        self._client = client
        self._model_alias = model_alias
        self._prompt = load_prompt(prompt_ref)
        self._cache = cache
        self._meter = meter
        self._cascade_enabled = cascade_enabled
        self._cascade_alias = cascade_alias
        self._flagship_alias = flagship_alias
        self.escalations = 0

    @property
    def prompt_version(self) -> str:
        return self._prompt.ref

    def answer(
        self,
        guarded: GuardedInput,
        session: Session,
    ) -> Reply:
        started = time.perf_counter()
        language = guarded.language

        directory_text = directory.rendered_directory(
            "ar" if language == "ar" else "en"
        )

        history = session.state.messages()

        messages = build_faq_messages(
            self._prompt,
            directory_text,
            history,
            guarded.text,
        )

        model_alias = (
            self._cascade_alias
            if self._cascade_enabled
            else self._model_alias
        )

        parameters = {
            "temperature": 0.4,
            "max_tokens": 700,
        }

        # Exact caching is conservative: once conversation history exists,
        # the previous turns may change the answer. We therefore cache only
        # standalone FAQ questions. This prevents a personalised conversation
        # from accidentally sharing a cached answer.
        cache_key = None
        if self._cache is not None and not history:
            history_for_key = [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in history
            ]

            cache_key = self._cache.exact_key(
                model_alias,
                self._prompt.ref,
                guarded.text,
                language,
                rendered_prompt=messages[0].content,
                history=history_for_key,
                parameters=parameters,
            )

            cached = self._cache.get(cache_key)

            if cached is not None:
                return Reply(
                    text=cached["text"],
                    intent="faq",
                    language=language,
                    model_id=cached.get("model_id", ""),
                    prompt_version=self._prompt.ref,
                    cache_tier="exact",
                    latency_ms=(
                        time.perf_counter() - started
                    ) * 1000,
                )

        response = self._client.complete(
            LLMRequest(
                messages=messages,
                model_alias=model_alias,
                temperature=parameters["temperature"],
                max_tokens=parameters["max_tokens"],
                # The first message is the stable prefix. The mock provider
                # observes repeated system prompts as cached input.
                cache_prefix_messages=1,
            )
        )

        text = response.text or ""

        if response.finish_reason == "length":
            log.warning(
                "answer_truncated",
                model_id=response.model_id,
            )

        if self._meter is not None:
            self._meter.meter_response(
                response,
                stage=(
                    "faq_cheap"
                    if self._cascade_enabled
                    else "faq"
                ),
                intent="faq",
                prompt_version=self._prompt.ref,
            )

        escalated_flag = False

        # Cascade: cheap first, flagship only when a deterministic
        # groundedness signal says the cheap answer is unsafe.
        if (
            self._cascade_enabled
            and unsupported_amounts(
                text,
                directory_text,
            )
        ):
            escalated_flag = True
            self.escalations += 1

            log.warning(
                "cascade_escalated",
                from_alias=model_alias,
                to_alias=self._flagship_alias,
                reason="unsupported_amount",
            )

            response = self._client.complete(
                LLMRequest(
                    messages=messages,
                    model_alias=self._flagship_alias,
                    temperature=parameters["temperature"],
                    max_tokens=parameters["max_tokens"],
                    cache_prefix_messages=1,
                )
            )

            text = response.text or text

            if self._meter is not None:
                self._meter.meter_response(
                    response,
                    stage="faq_escalated",
                    intent="faq",
                    prompt_version=self._prompt.ref,
                )

        if self._cache is not None and cache_key is not None:
            self._cache.put(
                cache_key,
                {
                    "text": text,
                    "model_id": response.model_id,
                },
            )

        return Reply(
            text=text,
            intent="faq",
            language=language,
            route=response.route,
            model_id=response.model_id,
            prompt_version=self._prompt.ref,
            escalated=escalated_flag,
            latency_ms=(
                time.perf_counter() - started
            ) * 1000,
            input_tokens=response.usage.input_tokens,
            cached_tokens=response.usage.cached_input_tokens,
            output_tokens=response.usage.output_tokens,
        )
