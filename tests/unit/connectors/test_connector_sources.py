"""Webhook and poller source behaviour tests."""

from __future__ import annotations

from sdk.connectors import (
    BookingEvent,
    MockBookingSystem,
    PollingBookingSource,
    WebhookBookingSource,
)


def test_webhook_receive_and_drain() -> None:
    """Webhook deliveries are queued, drained once, and never re-emitted."""
    source = WebhookBookingSource()
    assert source.source_name == "webhook"
    assert source.receive({"event_id": "evt-1", "call_id": "call-0001", "state": "confirmed"})
    assert source.pending_count() == 1
    batch = source.collect()
    assert [event.call_id for event in batch] == ["call-0001"]
    assert source.collect() == ()
    source.acknowledge("evt-1")
    assert source.pending_count() == 0


def test_webhook_bad_input_and_bounded_queue_are_fail_open() -> None:
    """Bad bodies return False and a full queue drops its oldest entry."""
    source = WebhookBookingSource(max_queue=1)
    assert source.receive({"call_id": "", "state": "confirmed"}) is False
    assert source.receive({"event_id": "evt-1", "call_id": "call-1", "state": "confirmed"})
    assert source.receive({"event_id": "evt-2", "call_id": "call-2", "state": "unknown"})
    assert [event.call_id for event in source.collect()] == ["call-2"]


def test_poller_only_emits_unacknowledged_events(booking: MockBookingSystem) -> None:
    """Acknowledged events are not re-emitted; unacked events are retried."""
    source = PollingBookingSource(booking)
    assert source.source_name == "poller"
    first = source.collect()
    assert len(first) == 6
    for event in first:
        source.acknowledge(event.event_id)
    assert source.collect() == ()
    booking.update("call-0001", "abandoned")
    second = source.collect()
    assert [event.call_id for event in second] == ["call-0001"]
    assert second[0].state == "abandoned"


class BrokenBooking:
    """Booking adapter whose fetch always fails."""

    def list_booking_events(self) -> list[BookingEvent]:
        """Simulate a vendor outage."""
        raise RuntimeError("booking API down")


def test_poller_fetch_failure_is_fail_open() -> None:
    """A broken booking adapter yields no events and does not raise."""
    source = PollingBookingSource(BrokenBooking())
    assert source.collect() == ()


def test_webhook_accepts_booking_event_objects() -> None:
    """The mock can publish a BookingEvent directly into the webhook source."""
    source = WebhookBookingSource()
    event = BookingEvent(
        event_id="evt-direct",
        call_id="call-0006",
        state="transferred",
        updated_at="2026-11-25T12:00:00Z",
    )
    assert source.receive(event)
    assert source.collect() == (event,)
