# Outcome connectors (P1-05)

Maps hotel-booking state changes to late outcomes and sends them to ingest's
`PATCH /v1/calls/{call_id}/outcome`. The connector runs **outside the call path**:
all public entry points catch failures, log them, and return a status or report
instead of raising.

## Layout

| File | Role |
|---|---|
| `interface.py` | `BookingOutcomeSource`, `BookingSnapshotSource`, `OutcomeSink`, `BookingEvent`, receipts/reports |
| `settings.py` | `ConnectorSettings`, `ConnectorMode`; env prefix `VOICELOOP_CONNECTOR_` |
| `mapping.py` | Booking state → `outcome.status`, final-record validation, shared-schema validation |
| `webhook.py` | `WebhookBookingSource`: bounded in-process queue fed by `receive(payload)` |
| `poller.py` | `PollingBookingSource`: polls a snapshot source, deduplicates by source `event_id` |
| `mock_booking.py` | Fixture-seeded `MockBookingSystem`, `FixtureEventStore`, `InMemoryOutcomeSink` |
| `connector.py` | `OutcomeConnector.run_once()`, `HttpOutcomeSink`, `create_connector()` |

## Run and test

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/connectors -q
.\.venv\Scripts\python.exe -m ruff check sdk/connectors tests/unit/connectors
.\.venv\Scripts\python.exe -m mypy sdk/connectors tests/unit/connectors
```

Typical in-process demo:

```python
from sdk.connectors import (
    FixtureEventStore,
    InMemoryOutcomeSink,
    MockBookingSystem,
    OutcomeConnector,
    WebhookBookingSource,
)

booking = MockBookingSystem.seed_from_fixtures()
events = FixtureEventStore.load()
source = WebhookBookingSource()
sink = InMemoryOutcomeSink(events)
connector = OutcomeConnector(source, sink)

change = booking.update(
    "call-0001",
    "edited_by_staff",
    booking.list_booking_events()[0].final_record,  # synthetic fixture slots
)
source.receive(change)  # normally the webhook caller does this
report = connector.run_once()  # delivered == 1
assert events.events_for("call-0001")[0]["outcome"]["status"] == "edited_by_staff"
```

Select the poller with `VOICELOOP_CONNECTOR_SOURCE_MODE=poller`, or pass
`ConnectorSettings(source_mode=ConnectorMode.POLLER)` to `create_connector()`.
The demo can be driven with an in-memory sink; production uses `HttpOutcomeSink`,
which adds no dependencies beyond the already-installed `httpx`.

## Contract notes

- **Join:** `call_id` only (`context.md` §5). No guest identifiers, transcript
  text or turn indices are used to join booking records to calls. Open question 4
  (context.md §16) remains unanswered; a real API adapter can be injected via
  `create_connector(booking=...)` by implementing `BookingSnapshotSource`.
- **Payload:** exactly the `outcome` object from `schemas/turn_event.schema.json`
  1.0.0: required `status`, optional `final_record` and `updated_at`. It is
  validated with `voiceloop_common.validate()` on a minimal carrier event; no
  parallel schema is defined. `final_record` must satisfy
  `schemas/booking.schema.json` and `validate_booking()` cross-field rules.
- **Status map:** `confirmed`/`booked`/`completed` → `confirmed`;
  `edited_by_staff`/`staff_edited`/`modified_by_staff` → `edited_by_staff`;
  `abandoned`/`cancelled`/`dropped` → `abandoned`;
  `transferred`/`escalated` → `transferred`; anything unrecognised → `unknown`.
- **Data:** the mock is seeded solely from synthetic `tests/fixtures/turns.jsonl`
  names and slots. The connector never invents a guest name, email or phone;
  records that fail booking validation are skipped and logged.
- **Fail-open:** malformed webhooks return `False`; a broken poller yields no
  events; invalid/mapping-failed events are skipped; sink failures are reported
  as `DeliveryReceipt(ok=False)`. A poller event is only acknowledged after a
  successful delivery, so transient failures are retried on the next poll.
- **Deployment:** run this as a separate worker/cron job, never inside a live
  call. PII screening remains P1-09's responsibility; the connector only
  transports already-approved synthetic records in this mock phase.
