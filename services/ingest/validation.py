"""Request-time turn-event validation built on the shared contract schema.

``voiceloop_common.validate`` is the authoritative gate and is invoked at
startup for the ``turn_event`` schema. It re-runs ``check_schema`` on every
call (~0.4 s in this environment), which would add unacceptable latency to a
live call path (``context.md`` §2 invariant 1). Request-time checks therefore
use a cached ``Draft202012Validator`` compiled from the *same* schema document,
with identical format checks, and map failures to the shared structured error
shape. The performance trade-off is recorded in ``docs/handoffs/P1-01.md``.
"""

from __future__ import annotations

import functools
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from services.ingest.errors import IngestHTTPError
from voiceloop_common import load_schema
from voiceloop_common import validate as common_validate

_SCHEMA_NAME = "turn_event"
_WARMUP_EVENT: dict[str, Any] = {
    "event_id": "00000000-0000-4000-8000-000000000000",
    "call_id": "voiceloop-schema-warmup",
    "turn_index": 0,
    "audio": {"uri": "voiceloop://schema-warmup"},
    "stt": {"text": ""},
}


@functools.lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    """Verify the contract once with the shared gate, then compile it once."""
    common_validate(_WARMUP_EVENT, _SCHEMA_NAME)
    return Draft202012Validator(load_schema(_SCHEMA_NAME), format_checker=FormatChecker())


def warmup() -> None:
    """Compile and verify the shared schema once at service startup."""
    _validator()


def validate_turn(event: Any) -> None:
    """Validate a turn payload against the shared schema without per-call setup.

    Args:
        event: Decoded JSON value to check.

    Raises:
        IngestHTTPError: Structured 422 when the payload violates the contract.

    """
    error = next(iter(_validator().iter_errors(event)), None)
    if error is not None:
        raise _structured_error(error)


def _structured_error(error: JsonSchemaValidationError) -> IngestHTTPError:
    """Convert a schema error to a safe structured HTTP error."""
    field = ".".join(str(part) for part in error.absolute_path) or "<root>"
    details: dict[str, Any] = {
        "schema": _SCHEMA_NAME,
        "field": field,
        "validator": error.validator or "invalid",
        "constraint": str(error.validator_value)[:200],
    }
    return IngestHTTPError(
        422,
        "validation_error",
        "Turn event failed contract validation.",
        details,
    )
