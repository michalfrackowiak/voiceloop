"""End-to-end connector tests: mock bookings update fixture-call outcomes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sdk.connectors import (
    BookingEvent,
    ConnectorMode,
    ConnectorSettings,
    DeliveryReceipt,
    FixtureEventStore,
    HttpOutcomeSink,
    InMemoryOutcomeSink,
    MockBookingSystem,
    OutcomeConnector,
    PollingBookingSource,
    WebhookBookingSource,
    create_connector,
)


def test_webhook_late_outcome_updates_fixture_events(
    booking: MockBookingSystem, events: FixtureEventStore
) -> None:
    """A late webhook outcome is mapped, validated and merged into fixture turns."""
    source = WebhookBookingSource()
    sink = InMemoryOutcomeSink(events)
    connector = OutcomeConnector(source, sink)

    original = next(
        event for event in booking.list_booking_events() if event.call_id == "call-0001"
    )
    assert original.final_record is not None
    record = dict(original.final_record)
    record["room_type"] = "suite"
    change = booking.update("call-0001", "edited_by_staff", record)

    assert source.receive(change)
    report = connector.run_once()

    assert report.delivered == 1
    assert report.failed == 0
    assert sink.deliveries()[0][0] == "call-0001"
    assert sink.deliveries()[0][1]["status"] == "edited_by_staff"
    updated = events.events_for("call-0001")
    assert len(updated) == 6
    assert all(turn["outcome"]["status"] == "edited_by_staff" for turn in updated)
    assert all(turn["outcome"]["final_record"]["room_type"] == "suite" for turn in updated)


def test_poller_updates_fixture_call_and_deduplicates(
    booking: MockBookingSystem, events: FixtureEventStore
) -> None:
    """Polling delivers every seeded fixture call, then late changes exactly once."""
    source = PollingBookingSource(booking)
    sink = InMemoryOutcomeSink(events)
    connector = OutcomeConnector(source, sink)

    first = connector.run_once()
    assert first.delivered == 6
    for index in range(1, 7):
        call_id = f"call-{index:04d}"
        assert len(events.events_for(call_id)) == 6
        assert events.events_for(call_id)[0]["outcome"]["status"] in {
            "confirmed",
            "edited_by_staff",
        }

    booking.update("call-0003", "abandoned")
    second = connector.run_once()
    assert second.delivered == 1
    updated = events.events_for("call-0003")[0]["outcome"]
    assert updated["status"] == "abandoned"
    assert "final_record" not in updated

    third = connector.run_once()
    assert third.delivered == 0
    assert third.failed == 0


class BrokenSource:
    """Source that fails both collection and acknowledgement."""

    source_name = "broken"

    def collect(self) -> Sequence[BookingEvent]:
        """Simulate a source outage."""
        raise RuntimeError("source down")

    def acknowledge(self, event_id: str) -> None:
        """Simulate a broken acknowledgement."""
        raise RuntimeError("ack down")


class ExplodingSink:
    """Sink that raises instead of returning a receipt."""

    def patch_outcome(self, call_id: str, payload: Mapping[str, Any]) -> DeliveryReceipt:
        """Simulate an unexpected sink failure."""
        raise RuntimeError("sink down")


def test_connector_never_raises_on_source_or_sink_failures() -> None:
    """Fail-open holds for a broken source and a raising sink."""
    connector = OutcomeConnector(BrokenSource(), ExplodingSink())
    report = connector.run_once()
    assert report.failed == 1
    assert report.delivered == 0

    class OneEventSource(BrokenSource):
        """Source with one deliverable event and a failing acknowledgement."""

        def collect(self) -> Sequence[BookingEvent]:
            """Return one valid event."""
            return (
                BookingEvent(
                    event_id="evt-fail",
                    call_id="call-0001",
                    state="confirmed",
                ),
            )

    second = OutcomeConnector(OneEventSource(), ExplodingSink()).run_once()
    assert second.failed == 1
    assert second.delivered == 0


def test_connector_skips_invalid_booking_record_without_raising(
    events: FixtureEventStore,
) -> None:
    """An invalid booking record is counted as a skip, not delivered."""
    source = WebhookBookingSource()
    invalid = BookingEvent(
        event_id="evt-invalid",
        call_id="call-0001",
        state="confirmed",
        final_record={"guest_name": "Ada Fixture", "guests": 2},
    )
    assert source.receive(invalid)
    sink = InMemoryOutcomeSink(events)
    report = OutcomeConnector(source, sink).run_once()
    assert report.delivered == 0
    assert report.failed == 1
    assert sink.deliveries() == []
    assert events.events_for("call-0001")[0].get("outcome") is None


def test_create_connector_selects_source_from_settings(
    booking: MockBookingSystem, events: FixtureEventStore
) -> None:
    """Typed settings choose webhook or poller implementations."""
    webhook = create_connector(
        ConnectorSettings(source_mode=ConnectorMode.WEBHOOK),
        sink=InMemoryOutcomeSink(events),
    )
    poller = create_connector(
        ConnectorSettings(source_mode=ConnectorMode.POLLER),
        booking=booking,
        sink=InMemoryOutcomeSink(events),
    )
    assert webhook.source_name == "webhook"
    assert poller.source_name == "poller"
    assert poller.run_once().delivered == 6


class FakeResponse:
    """Minimal httpx response stand-in."""

    def __init__(self, status_code: int) -> None:
        """Store the HTTP status."""
        self.status_code = status_code


class FakeClient:
    """Records PATCH calls without any network access."""

    def __init__(self, status_code: int = 202) -> None:
        """Create a client that returns a fixed status."""
        self.status_code = status_code
        self.calls: list[tuple[str, dict[str, Any], float]] = []

    def patch(self, url: str, *, json: dict[str, Any], timeout: float) -> FakeResponse:
        """Record one PATCH request."""
        self.calls.append((url, dict(json), timeout))
        return FakeResponse(self.status_code)


def test_http_sink_patches_ingest_outcome_endpoint() -> None:
    """The real sink targets the ingest PATCH path with the outcome body."""
    client = FakeClient()
    sink = HttpOutcomeSink("http://ingest.test/", 0.5, client)
    payload = {"status": "confirmed", "updated_at": "2026-11-25T12:00:00Z"}

    receipt = sink.patch_outcome("call-0001", payload)

    assert receipt.ok
    assert receipt.status_code == 202
    assert client.calls == [("http://ingest.test/v1/calls/call-0001/outcome", payload, 0.5)]


def test_http_sink_reports_non_success_status() -> None:
    """A rejected PATCH is returned as a failed receipt, not an exception."""
    sink = HttpOutcomeSink("http://ingest.test", 0.5, FakeClient(422))
    receipt = sink.patch_outcome("call-0001", {"status": "confirmed"})
    assert not receipt.ok
    assert receipt.status_code == 422
