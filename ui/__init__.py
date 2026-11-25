"""P4-03 synthetic custom review UI and its deterministic offline harness.

Everything in this package is a fixture/fake stand-in. It performs no audio
playback, network access, human-usability measurement or promotion decision.
"""

from __future__ import annotations

from .constants import (
    ALLOWED_SPEEDS,
    EXPECTED_BOOKING_SCHEMA_SHA256,
    EXPECTED_FIXTURE_SHA256,
    EXPECTED_GOLDEN_CONTENT_HASH,
    INTERNAL_STATE_VERSION,
    LOW_CONFIDENCE_CEILING,
    ROUTING_LABELS,
    SYNTHETIC_LABEL,
    USABILITY_ACCEPTANCE,
    USABILITY_STATUS,
    WAVEFORM_BARS,
)
from .errors import UiValidationError

__all__ = [
    "ALLOWED_SPEEDS",
    "EXPECTED_BOOKING_SCHEMA_SHA256",
    "EXPECTED_FIXTURE_SHA256",
    "EXPECTED_GOLDEN_CONTENT_HASH",
    "INTERNAL_STATE_VERSION",
    "LOW_CONFIDENCE_CEILING",
    "ROUTING_LABELS",
    "SYNTHETIC_LABEL",
    "USABILITY_ACCEPTANCE",
    "USABILITY_STATUS",
    "WAVEFORM_BARS",
    "UiValidationError",
]
