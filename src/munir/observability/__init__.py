"""Observability primitives shared by the cost/latency instrumentation.

A trace id groups all model calls produced by one user request. This matters
for Module 6 because one request may call the guard classifier, router, FAQ
model, and fallback model; the meter must let us see that as one trace.
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from contextvars import ContextVar
from pathlib import Path

import structlog

_trace_id: ContextVar[str] = ContextVar(
    "trace_id",
    default="",
)
_configured = False


def new_trace_id() -> str:
    """Start a new trace for one incoming request."""

    trace_id = uuid.uuid4().hex[:12]
    _trace_id.set(trace_id)
    return trace_id


def current_trace_id() -> str:
    return _trace_id.get()


def _add_trace_id(_logger, _name, event_dict):
    if trace_id := _trace_id.get():
        event_dict.setdefault(
            "trace_id",
            trace_id,
        )
    return event_dict


def configure_logging(
    *,
    json_lines: bool | None = None,
    path: str | Path | None = None,
) -> None:
    """Configure human-readable or JSON-line structured logging."""

    global _configured

    json_lines = (
        json_lines
        if json_lines is not None
        else os.environ.get("MUNIR_LOG_JSON") == "1"
    )

    path = path or os.environ.get(
        "MUNIR_LOG_FILE"
    )

    stream = sys.stderr

    if path:
        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        stream = open(
            path,
            "a",
            encoding="utf-8",
        )
        json_lines = True

    renderer = (
        structlog.processors.JSONRenderer(
            ensure_ascii=False
        )
        if json_lines
        else structlog.dev.ConsoleRenderer(
            colors=False
        )
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _add_trace_id,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(
                logging,
                os.environ.get(
                    "MUNIR_LOG_LEVEL",
                    "INFO",
                ).upper(),
                logging.INFO,
            )
        ),
        logger_factory=structlog.PrintLoggerFactory(
            file=stream
        ),
        cache_logger_on_first_use=True,
    )

    _configured = True


def get_logger(name: str = "munir"):
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)
