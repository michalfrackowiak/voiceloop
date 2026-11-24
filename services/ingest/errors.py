"""Structured error helpers for the ingest HTTP API."""

from __future__ import annotations

import re
from typing import Any

from voiceloop_common.errors import ValidationError as ContractValidationError

_FIELD_RE = re.compile(r" at ([^:]+): ")
_QUOTED_RE = re.compile(r"['\"][^'\"\n]{0,200}['\"]")


class IngestHTTPError(Exception):
    """An HTTP failure with a stable code and a safe, structured detail object."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Store the status, machine-readable code, message and details."""
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details: dict[str, Any] = details or {}


def error_body(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the shared ``{code, message, details}`` error response shape."""
    return {"code": code, "message": message, "details": details or {}}


def schema_error_details(exc: ContractValidationError) -> dict[str, Any]:
    """Extract safe field-level context from a shared-schema validation error.

    Quoted values are removed from the underlying message because they can
    contain transcript text or other guest data. The schema name and field
    location are retained for callers.
    """
    message = str(exc)
    match = _FIELD_RE.search(message)
    field = match.group(1) if match else "<root>"
    reason = _QUOTED_RE.sub("<redacted>", message.split(" at ", 1)[-1].split(": ", 1)[-1])
    return {
        "schema": "turn_event",
        "field": field,
        "reason": reason[:300],
    }
