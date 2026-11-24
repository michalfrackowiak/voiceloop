"""Fail-open orchestration from booking events to ingest outcome PATCHes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

import httpx
import structlog

from .interface import (
    BookingEvent,
    BookingOutcomeSource,
    BookingSnapshotSource,
    DeliveryReceipt,
    OutcomeSink,
    SyncReport,
)
from .mapping import build_outcome_payload, validate_outcome_payload
from .mock_booking import MockBookingSystem
from .poller import PollingBookingSource
from .settings import ConnectorMode, ConnectorSettings
from .webhook import WebhookBookingSource

logger = structlog.get_logger("voiceloop.connectors")


class HttpOutcomeSink:
    """Send validated outcomes to ``PATCH /v1/calls/{call_id}/outcome``.

    The HTTP client is optional injection for tests; only an internally created
    client is closed by :meth:`close`.
    """

    def __init__(
        self,
        ingest_endpoint: str,
        timeout_seconds: float,
        client: Any | None = None,
    ) -> None:
        """Create a sink with a bounded request timeout.

        Args:
            ingest_endpoint: Ingest base URL, e.g. ``http://localhost:8000``.
            timeout_seconds: Per-request timeout passed to httpx.
            client: Optional object with an httpx-compatible ``patch`` method.

        """
        self._base = ingest_endpoint.rstrip("/")
        self._timeout = timeout_seconds
        self._owns_client = client is None
        self._client: Any = client if client is not None else httpx.Client()

    def patch_outcome(self, call_id: str, payload: Mapping[str, Any]) -> DeliveryReceipt:
        """PATCH one outcome body; return a failed receipt instead of raising."""
        try:
            validate_outcome_payload(call_id, payload)
            url = f"{self._base}/v1/calls/{quote(call_id, safe='')}/outcome"
            response = self._client.patch(url, json=dict(payload), timeout=self._timeout)
            status = int(response.status_code)
            if 200 <= status < 300:
                return DeliveryReceipt(ok=True, status_code=status)
            self._log(
                "warning",
                "connector_outcome_rejected",
                call_id=call_id,
                status_code=status,
            )
            return DeliveryReceipt(ok=False, status_code=status, error=f"http_{status}")
        except Exception as exc:
            self._log("warning", "connector_outcome_send_failed", call_id=call_id, error=str(exc))
            return DeliveryReceipt(ok=False, error=str(exc))

    def close(self) -> None:
        """Close an internally owned HTTP client; safe to call repeatedly."""
        if not self._owns_client:
            return
        try:
            self._client.close()
        except Exception as exc:
            self._log("warning", "connector_http_close_failed", error=str(exc))

    @staticmethod
    def _log(level: str, message: str, **fields: Any) -> None:
        """Emit a structured log line, swallowing logging failures."""
        try:
            event = getattr(logger, level, logger.warning)
            event(message, **fields)
        except Exception:
            pass


class OutcomeConnector:
    """Collect, map, validate and deliver booking outcomes outside the call path.

    Every public operation is fail-open: source, mapping, sink and log failures
    are counted or logged, never propagated to a caller.
    """

    def __init__(self, source: BookingOutcomeSource, sink: OutcomeSink) -> None:
        """Create a connector over one source and one sink.

        Args:
            source: Webhook queue or poller implementing ``BookingOutcomeSource``.
            sink: PATCH destination implementing ``OutcomeSink``.

        """
        self._source = source
        self._sink = sink

    @property
    def source_name(self) -> str:
        """Return the source mode label, degrading to ``unknown`` on failure."""
        try:
            return str(self._source.source_name)
        except Exception:
            return "unknown"

    def run_once(self, *, max_events: int | None = None) -> SyncReport:
        """Process one batch of booking events without raising.

        Args:
            max_events: Optional cap for this pass; settings supply the default
                in :func:`create_connector`.

        Returns:
            A :class:`SyncReport` with delivered/skipped/failed counts.

        """
        source_name = self.source_name
        try:
            events: list[BookingEvent] = list(self._source.collect())
        except Exception as exc:
            self._log("error", "connector_source_failed", error=str(exc))
            return SyncReport(source=source_name, failed=1)

        limit = len(events) if max_events is None else max(0, max_events)
        considered, delivered, skipped, failed = 0, 0, len(events) - limit, 0
        for event in events[:limit]:
            considered += 1
            try:
                payload = build_outcome_payload(event)
                validate_outcome_payload(event.call_id, payload)
            except Exception as exc:
                failed += 1
                self._log(
                    "warning",
                    "connector_event_skipped",
                    call_id=event.call_id,
                    event_id=event.event_id,
                    error=str(exc),
                )
                continue
            try:
                receipt = self._sink.patch_outcome(event.call_id, payload)
            except Exception as exc:
                receipt = DeliveryReceipt(ok=False, error=str(exc))
            if not receipt.ok:
                failed += 1
                self._log(
                    "warning",
                    "connector_delivery_failed",
                    call_id=event.call_id,
                    event_id=event.event_id,
                    error=receipt.error or f"http_{receipt.status_code}",
                )
                continue
            delivered += 1
            try:
                self._source.acknowledge(event.event_id)
            except Exception as exc:
                self._log(
                    "warning",
                    "connector_ack_failed",
                    event_id=event.event_id,
                    error=str(exc),
                )
        return SyncReport(
            source=source_name,
            considered=considered,
            delivered=delivered,
            skipped=max(0, skipped),
            failed=failed,
        )

    @staticmethod
    def _log(level: str, message: str, **fields: Any) -> None:
        """Emit a structured log line, swallowing logging failures."""
        try:
            event = getattr(logger, level, logger.warning)
            event(message, **fields)
        except Exception:
            pass


def create_connector(
    settings: ConnectorSettings | None = None,
    *,
    booking: BookingSnapshotSource | None = None,
    sink: OutcomeSink | None = None,
) -> OutcomeConnector:
    """Build a connector with its source selected by typed settings.

    Args:
        settings: Defaults to environment variables prefixed
            ``VOICELOOP_CONNECTOR_``.
        booking: Optional booking adapter; defaults to the fixture-seeded mock
            when polling is selected.
        sink: Optional PATCH sink; defaults to the httpx ingest sink.

    Returns:
        A configured connector.

    """
    config = settings or ConnectorSettings()
    source: BookingOutcomeSource
    if config.source_mode == ConnectorMode.POLLER:
        adapter = booking or MockBookingSystem.seed_from_fixtures(config.fixture_dir)
        source = PollingBookingSource(adapter, max_events=config.max_events_per_run)
    else:
        source = WebhookBookingSource(max_queue=config.webhook_queue_size)
    return OutcomeConnector(
        source,
        sink
        if sink is not None
        else HttpOutcomeSink(config.ingest_endpoint, config.timeout_seconds),
    )
