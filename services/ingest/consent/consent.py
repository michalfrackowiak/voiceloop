"""Consent checks for dataset eligibility (P1-09.1)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

TurnEvent = Mapping[str, Any]


def consent_block(event: TurnEvent) -> Mapping[str, Any]:
    """Return an event's consent block as a read-only mapping.

    Args:
        event: A schema-valid turn event, treated strictly read-only.

    Returns:
        The nested ``consent`` mapping, or an empty mapping when absent or
        malformed. Missing consent is deliberately indistinguishable from a
        missing flag at the eligibility boundary.

    """
    consent = event.get("consent")
    return consent if isinstance(consent, Mapping) else {}


def is_eligible(event: TurnEvent) -> bool:
    """Return whether a turn may enter a training dataset.

    Both ``consent.recording`` and ``consent.training`` must be the boolean
    ``True``. Consent is optional in the ingest transport schema for incremental
    adoption, but it is mandatory for dataset eligibility (`docs/schemas.md`).
    The event is never mutated; malformed flags fail closed.

    Args:
        event: A schema-valid turn event, read-only.

    Returns:
        ``True`` only when recording and training consent are both granted.

    """
    consent = consent_block(event)
    return consent.get("recording") is True and consent.get("training") is True
