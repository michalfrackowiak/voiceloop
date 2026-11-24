"""Validation entry point for the turn event contract."""

from __future__ import annotations

from typing import Any

from . import load_schema, validate

TURN_EVENT_SCHEMA_NAME = "turn_event.schema.json"


def load_turn_event_schema() -> dict[str, Any]:
    """Load the versioned turn event JSON Schema.

    Returns:
        The parsed turn event schema.

    """
    return load_schema(TURN_EVENT_SCHEMA_NAME)


def validate_turn_event(event: dict[str, Any]) -> None:
    """Validate a turn event at ingest, including JSON Schema format checks.

    Args:
        event: Decoded turn event payload.

    Raises:
        jsonschema.ValidationError: If the payload violates the contract.

    """
    validate(event, TURN_EVENT_SCHEMA_NAME)
