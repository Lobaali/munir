"""
Composition root for the Munir application.

SECTION 6 — COMMERCIAL VS OPEN-WEIGHT

The important Section 6 rule is that a model comparison must change the
configured route, not duplicate the application.

build_assistant(route=...) therefore builds the SAME Munir pipeline for
either the commercial route or the open-weight route.
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


def _env_bool(name: str, default: bool) -> bool:
    """
    Read a boolean environment variable.

    This lets us change experimental settings without editing
    the application code.
    """
    value = os.environ.get(name)

    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_client(settings: dict, route_name: str):
    """
    Build the configured client for one route.

    The route can point to:
    - a commercial provider
    - an open-weight/self-hosted provider
    - the deterministic mock backend used for development/testing

    Provider SDK imports remain inside the adapter, preserving
    the Section 1 model boundary.
    """
    route = settings["routes"][route_name]

    if route["backend"] == "real":
        from munir.llm.openai_compat import OpenAICompatClient

        return OpenAICompatClient(
            base_url=route["base_url"],
            api_key_env=route["api_key_env"],
            model_map=route["aliases"],
        )

    # The mock backend is used for reproducible grading runs.
    # Its model ID still comes from the configuration aliases.
    model_id = route["aliases"].get(
        "munir-default",
        "campus-flagship",
    )

    return MockBrainClient(
        model_id=model_id,
        route=route_name,
    )


def build_assistant(
    route: str | None = None,
    *,
    cascade_enabled: bool | None = None,
    cache_enabled: bool | None = None,
) -> Munir:
    """
    Build one complete Munir assistant.

    route:
        Selects which configured model route should be tested.

    cascade_enabled:
        Explicitly enables/disables the FAQ cheap-first cascade.

        For Section 6 model comparison we use False so that both
        commercial and open-weight routes are compared directly.

    cache_enabled:
        Explicitly enables/disables response caching.

        For Section 6 we use False so that both routes actually
        execute the same model requests.

    Environment overrides are also supported:

        MUNIR_CASCADE_ENABLED=true|false
        MUNIR_CACHE_ENABLED=true|false
    """

    settings = load_settings()

    # The requested route becomes the primary route.
    primary_name = route or settings["primary_route"]

    # Keep the configured fallback chain from Section 1.
    fallback_name = settings["fallback_route"]

    primary = build_client(
        settings,
        primary_name,
    )

    fallback = build_client(
        settings,
        fallback_name,
    )

    client = ResilientClient(
        chain=[
            ("primary", primary),
            ("fallback", fallback),
        ]
    )

    # CostMeter is reused from Section 5.
    prices = PriceSheet.from_settings(settings)
    meter = CostMeter(prices)

    # Read the configured cache setting unless an experiment explicitly
    # overrides it.
    configured_cache = settings.get(
        "cache",
        {},
    ).get(
        "enabled",
        False,
    )

    if cache_enabled is None:
        use_cache = _env_bool(
            "MUNIR_CACHE_ENABLED",
            configured_cache,
        )
    else:
        use_cache = cache_enabled

    cache = ResponseCache() if use_cache else None

    # Read the configured cascade setting unless an experiment explicitly
    # overrides it.
    configured_cascade = settings.get(
        "pipeline",
        {},
    ).get(
        "faq_cascade_enabled",
        False,
    )

    if cascade_enabled is None:
        use_cascade = _env_bool(
            "MUNIR_CASCADE_ENABLED",
            configured_cascade,
        )
    else:
        use_cascade = cascade_enabled

    deps = Dependencies(
        input_guard=InputGuard(
            client=client,
            classifier_alias="munir-guard",
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

    # Expose the real meter and cache so benchmark/evaluation scripts
    # can inspect what actually happened.
    assistant.meter = meter
    assistant.cache = cache

    return assistant