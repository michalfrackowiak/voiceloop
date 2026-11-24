"""Outcome connectors: booking state changes mapped to VoiceLoop call outcomes."""

from __future__ import annotations

from .connector import HttpOutcomeSink, OutcomeConnector, create_connector
from .interface import (
    BookingEvent,
    BookingOutcomeSource,
    BookingSnapshotSource,
    DeliveryReceipt,
    OutcomeSink,
    SyncReport,
)
from .mapping import (
    OutcomeMappingError,
    OutcomeStatus,
    build_outcome_payload,
    map_booking_state,
    outcome_to_booking_state,
    parse_booking_event,
    validate_outcome_payload,
)
from .mock_booking import FixtureEventStore, InMemoryOutcomeSink, MockBookingSystem
from .poller import PollingBookingSource
from .settings import ConnectorMode, ConnectorSettings
from .webhook import WebhookBookingSource

__all__ = [
    "BookingEvent",
    "BookingOutcomeSource",
    "BookingSnapshotSource",
    "ConnectorMode",
    "ConnectorSettings",
    "DeliveryReceipt",
    "FixtureEventStore",
    "HttpOutcomeSink",
    "InMemoryOutcomeSink",
    "MockBookingSystem",
    "OutcomeConnector",
    "OutcomeMappingError",
    "OutcomeSink",
    "OutcomeStatus",
    "PollingBookingSource",
    "SyncReport",
    "WebhookBookingSource",
    "build_outcome_payload",
    "create_connector",
    "map_booking_state",
    "outcome_to_booking_state",
    "parse_booking_event",
    "validate_outcome_payload",
]
