"""Exact response caching for deterministic, non-personalised FAQ answers.

Rubric 5.4: the cache key includes every known answer-changing variable:
model alias, prompt version, rendered trusted context, user text, language,
sampling parameters, and conversation history.

We deliberately do NOT cache personalised service answers here. A response
that depends on identity, authorization, or mutable account state must not
be served to another request just because the wording is similar.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheStats:
    """Small set of counters used in the cost/latency report."""

    lookups: int = 0
    hits: int = 0
    stores: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def render(self) -> str:
        return (
            f"lookups {self.lookups} | hits {self.hits} | "
            f"hit rate {self.hit_rate:.0%} | stores {self.stores}"
        )


class ResponseCache:
    """A bounded-by-TTL in-memory exact cache.

    The cache is intentionally simple for the capstone. The important
    engineering lesson is the key, not the storage technology.
    """

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._data: dict[str, tuple[float, str]] = {}
        self._ttl = ttl_seconds
        self.stats = CacheStats()

    @staticmethod
    def exact_key(
        model_alias: str,
        prompt_version: str,
        query: str,
        language: str,
        *,
        rendered_prompt: str = "",
        history: list[dict[str, str]] | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> str:
        """Create a key containing all known answer-changing inputs.

        Why each field is here:
        - model_alias: different models can answer differently.
        - prompt_version: a prompt change changes behaviour.
        - rendered_prompt: the actual campus facts/rules can change even if
          the prompt version does not.
        - query: the student's question.
        - language: Arabic and English responses can differ.
        - history: previous turns can change the answer.
        - parameters: temperature/max_tokens and future sampling controls.
        """

        payload = {
            "model": model_alias,
            "prompt_version": prompt_version,
            "rendered_prompt": rendered_prompt,
            "query": query.strip(),
            "language": language,
            "history": history or [],
            "parameters": parameters or {},
        }

        blob = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        return "exact:" + hashlib.sha256(
            blob.encode("utf-8")
        ).hexdigest()

    def get(self, key: str) -> Any | None:
        self.stats.lookups += 1

        entry = self._data.get(key)
        if entry is None:
            return None

        expires_at, value = entry
        if time.time() > expires_at:
            del self._data[key]
            return None

        self.stats.hits += 1
        return json.loads(value)

    def put(self, key: str, value: Any) -> None:
        self._data[key] = (
            time.time() + self._ttl,
            json.dumps(value, ensure_ascii=False),
        )
        self.stats.stores += 1

    def clear(self) -> None:
        self._data.clear()
        self.stats = CacheStats()
