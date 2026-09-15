"""Cost metering: turn every model response into a money record.

Rubric 5.1
-----------
Every provider response already contains input/output/cached token usage.
This module converts that usage into a CostRecord immediately, so cost is
measured per request rather than estimated later from averages.

The price sheet is configuration, while this module owns the calculation.
That separation means a price change does not require changing application
logic.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from munir.config import PriceSheet
from munir.llm.interfaces import LLMResponse
from munir.observability import get_logger

log = get_logger(__name__)


class CostRecord(BaseModel):
    """One measured model call and the money it consumed."""

    route: str = ""
    intent: str = "unknown"
    stage: str = ""
    model_id: str
    prompt_version: str = ""

    # These come from the provider response; they are not guessed by the
    # application after the fact.
    input_tokens: int = Field(ge=0)
    cached_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)

    latency_ms: float = Field(default=0.0, ge=0.0)
    cost_sar: float = Field(ge=0.0)

    # Non-empty means this was served by the response cache rather than a
    # provider call. Such a response has zero model cost.
    cache_tier: str = ""
    trace_id: str = ""


class CostMeter:
    """Collect, persist, and aggregate measured model-call costs."""

    def __init__(
        self,
        prices: PriceSheet,
        sink: str | Path | None = None,
    ) -> None:
        self._prices = prices
        self.records: list[CostRecord] = []
        self._sink = Path(sink) if sink else None

        if self._sink:
            self._sink.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

    @staticmethod
    def _plain_model_id(model_id: str) -> str:
        """ResilientClient may prefix a model id with a route name."""

        return model_id.rsplit(":", 1)[-1]

    def price_of(
        self,
        model_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        """Convert token usage into SAR using the configured price row."""

        price = self._prices.for_model(
            self._plain_model_id(model_id)
        )

        fresh_input = max(
            input_tokens - cached_tokens,
            0,
        )

        cost = (
            fresh_input * price.input_per_mtok
            + cached_tokens * price.cached_input_per_mtok
            + output_tokens * price.output_per_mtok
        ) / 1_000_000

        return round(cost, 6)

    def meter(
        self,
        *,
        model_id: str,
        stage: str = "",
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        route: str = "",
        intent: str = "unknown",
        prompt_version: str = "",
        latency_ms: float = 0.0,
        cache_tier: str = "",
    ) -> CostRecord:
        """Create and store one cost record."""

        from munir.observability import current_trace_id

        cost_sar = (
            0.0
            if cache_tier
            else self.price_of(
                model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
            )
        )

        record = CostRecord(
            route=route,
            intent=intent,
            stage=stage,
            model_id=model_id,
            prompt_version=prompt_version,
            input_tokens=input_tokens,
            cached_tokens=cached_tokens,
            output_tokens=output_tokens,
            latency_ms=round(latency_ms, 1),
            cost_sar=cost_sar,
            cache_tier=cache_tier,
            trace_id=current_trace_id(),
        )

        self.records.append(record)
        log.info("llm_cost", **record.model_dump())

        if self._sink:
            with self._sink.open(
                "a",
                encoding="utf-8",
            ) as fh:
                fh.write(
                    json.dumps(
                        record.model_dump(),
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        return record

    def meter_response(
        self,
        response: LLMResponse,
        **kwargs,
    ) -> CostRecord:
        """Meter directly from the usage returned by the model boundary."""

        return self.meter(
            model_id=response.model_id,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cached_tokens=response.usage.cached_input_tokens,
            latency_ms=response.latency_ms,
            route=response.route,
            **kwargs,
        )

    @property
    def total_sar(self) -> float:
        return round(
            sum(record.cost_sar for record in self.records),
            6,
        )

    @property
    def total_input_tokens(self) -> int:
        return sum(
            record.input_tokens
            for record in self.records
        )

    @property
    def total_cached_tokens(self) -> int:
        return sum(
            record.cached_tokens
            for record in self.records
        )

    @property
    def total_output_tokens(self) -> int:
        return sum(
            record.output_tokens
            for record in self.records
        )

    def by(self, field: str) -> dict[str, float]:
        """Aggregate spend by route, intent, or pipeline stage."""

        out: dict[str, float] = defaultdict(float)

        for record in self.records:
            out[getattr(record, field)] += record.cost_sar

        return {
            key: round(value, 6)
            for key, value in sorted(
                out.items(),
                key=lambda item: -item[1],
            )
        }

    def cached_input_share(self) -> float:
        """Fraction of input tokens reported as provider-cache hits."""

        if self.total_input_tokens == 0:
            return 0.0

        return (
            self.total_cached_tokens
            / self.total_input_tokens
        )

    def reset(self) -> None:
        self.records.clear()
