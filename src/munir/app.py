"""Munir's composition root.
"""

from __future__ import annotations

import os

from munir.caching.response_cache import ResponseCache
from munir.config import PriceSheet, load_settings
from munir.guards.input_guards import InputGuard
from munir.guards.output_guards import OutputGuard
from munir.llm.fake import MockBrainClient
from munir.llm.resilient import ResilientClient
from munir.observability.cost import CostMeter
from munir.pipeline.assemble import Dependencies, EscalationHandler, Munir
from munir.pipeline.faq import FAQHandler
from munir.pipeline.router import IntentRouter
from munir.pipeline.service import ServiceWorkflow


def build_client(settings: dict, route_name: str):
    """Build a provider implementation from configuration only."""

    route = settings["routes"][route_name]

    if route["backend"] == "real":
        from munir.llm.openai_compat import OpenAICompatClient

        return OpenAICompatClient(
            base_url=route["base_url"],
            api_key_env=route["api_key_env"],
            model_map=route["aliases"],
        )

    model_id = route["aliases"].get(
        "munir-default",
        "campus-flagship",
    )
    return MockBrainClient(model_id=model_id, route=route_name)


def build_assistant(
    route: str | None = None,
    *,
    cache_enabled: bool | None = None,
    cascade_enabled: bool | None = None,
    faq_prompt: str | None = None,
    meter_sink: str | None = None,
) -> Munir:
    """Build the real application, optionally with an experiment profile.

    These keyword overrides are for measured experiments only. Production
    defaults still come from configs/munir.yaml. This lets scripts/replay.py
    compare configurations using exactly the same Munir pipeline.
    """

    settings = load_settings()
    primary_name = route or settings["primary_route"]
    fallback_name = settings["fallback_route"]

    primary = build_client(settings, primary_name)
    fallback = build_client(settings, fallback_name)
    client = ResilientClient(
        chain=[("primary", primary), ("fallback", fallback)]
    )

    prices = PriceSheet.from_settings(settings)
    meter = CostMeter(prices, sink=meter_sink)

    cache_default = settings.get("cache", {}).get(
        "enabled",
        False,
    )
    # Environment overrides are used by scripts/replay.py to reproduce
    # BEFORE and AFTER optimisation configurations without editing code.
    env_cache = os.environ.get("MUNIR_CACHE_ENABLED")
    if cache_enabled is None and env_cache is not None:
        cache_enabled = env_cache.lower() in {"1", "true", "yes", "on"}
    use_cache = cache_default if cache_enabled is None else cache_enabled
    cache = (
        ResponseCache(
            ttl_seconds=int(
                settings.get("cache", {}).get("ttl_seconds", 3600)
            )
        )
        if use_cache
        else None
    )

    cascade_default = settings.get("pipeline", {}).get("faq_cascade_enabled", True)
    env_cascade = os.environ.get("MUNIR_CASCADE_ENABLED")
    if cascade_enabled is None and env_cascade is not None:
        cascade_enabled = env_cascade.lower() in {"1", "true", "yes", "on"}
    use_cascade = (
        cascade_default
        if cascade_enabled is None
        else cascade_enabled
    )

    if faq_prompt is None:
        faq_prompt = os.environ.get("MUNIR_FAQ_PROMPT")

    deps = Dependencies(
        input_guard=InputGuard(
            client=client,
            classifier_alias=settings["guards"]["classifier_alias"],
            meter=meter,
        ),
        router=IntentRouter(
            client,
            model_alias="munir-router",
            meter=meter,
        ),
        faq_handler=FAQHandler(
            client,
            model_alias="munir-default",
            prompt_ref=faq_prompt or "answer_faq.v2",
            cascade_enabled=use_cascade,
            cascade_alias="munir-cheap",
            cache=cache,
            meter=meter,
        ),
        service_workflow=ServiceWorkflow(
            client,
            model_alias="munir-default",
            meter=meter,
        ),
        output_guard=OutputGuard(),
        escalation=EscalationHandler(),
    )

    assistant = Munir(deps)

    # Expose experiment telemetry to scripts/notebooks. This is not used
    # by business logic; it is observability for Module 6.
    assistant.meter = meter
    assistant.cache = cache

    return assistant
