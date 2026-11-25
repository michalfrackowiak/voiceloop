# Ingest fail-open buffer (P1-02)

Self-contained, stdlib-only module for bounded buffering during Postgres/S3
outages. It is not yet wired into the ingest FastAPI app: P1-01 was editing the
`services/ingest/` root files concurrently, so the lead performs integration.
`docs/handoffs/P1-02.md` is the canonical wiring reference.

## Run / test

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/unit/ingest_failopen
.\.venv\Scripts\python.exe -m ruff check services/ingest/failopen tests/unit/ingest_failopen
.\.venv\Scripts\python.exe -m ruff format --check services/ingest/failopen tests/unit/ingest_failopen
.\.venv\Scripts\python.exe -m mypy services/ingest/failopen tests/unit/ingest_failopen
```

## Public API

Exported by `services.ingest.failopen`:

- `BoundedBuffer`, `BufferedEvent`, `EnqueueResult`, `FlushResult`
- `FailOpenMetrics`, `HealthSnapshot`, `HealthStatus`
- `FailOpenSettings`, `RecordingSink`, `ShedReason`, `StorageSink`

```python
from services.ingest.failopen import (
    BoundedBuffer,
    BufferedEvent,
    EnqueueResult,
    FailOpenMetrics,
    FailOpenSettings,
    FlushResult,
    HealthSnapshot,
    HealthStatus,
    RecordingSink,
    ShedReason,
    StorageSink,
)

sink = RecordingSink(down=True)  # simulates a storage outage
buffer = BoundedBuffer(sink, FailOpenSettings(capacity=64))
receipt = buffer.enqueue(event)  # never raises; signal is "accepted"|"shed"
result = buffer.flush()  # timed_out=True when budget is exhausted
health = buffer.health().to_dict()  # healthy/degraded plus counters
counters = buffer.counters()
buffer.close()  # also supports `with BoundedBuffer(...) as buffer:`
```

`FailOpenSettings` reads `VOICELOOP_INGEST_FAILOPEN_*` and exposes `backend`,
`capacity`, `enqueue_timeout_ms`, `max_payload_bytes`, `writer_workers`,
`flush_timeout_s`, `degraded_ratio` and `disk_dir`. No schema or cross-service
contract changed.
