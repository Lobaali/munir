"""What comes out of the pipeline.

"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Reply(BaseModel):
    text: str = ""
    intent: str = "faq"
    language: str = "en"
    route: str = ""
    model_id: str = ""
    prompt_version: str = ""
    cache_tier: str = ""  # "", "exact"
    blocked: bool = False
    guard_layer: str = "none"
    guard_category: str = "ok"
    output_guard_category: str = "ok"
    tool_calls: list[dict] = Field(default_factory=list)
    tool_iterations: int = 0
    escalated: bool = False
    latency_ms: float = 0.0
    cost_sar: float = 0.0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
