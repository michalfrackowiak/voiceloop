"""JSON-formatted structured logging configuration."""

from __future__ import annotations

import logging as std_logging
from typing import Any

import structlog


def configure_logging(service_name: str, level: str = "INFO") -> None:
    """Configure structlog to emit JSON log events tagged with the service name.

    Args:
        service_name: Value bound to every event as ``service``.
        level: Minimum log level name, case-insensitive. Unknown values fall
            back to INFO so logging setup cannot break service startup.

    """
    numeric_level = std_logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        numeric_level = std_logging.INFO

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(service=service_name)


def log_context(**values: Any) -> None:
    """Bind context variables to every subsequent log event in this context."""
    structlog.contextvars.bind_contextvars(**values)
