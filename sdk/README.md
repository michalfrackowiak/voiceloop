# VoiceLoop Python SDK (P1-03)

Fail-open turn logging and native-rate audio helpers for voice agents. The SDK
honours `context.md` §2.1: `log_turn` performs no validation, I/O, sleeps or
logging in the calling thread. A daemon worker validates against
`turn_event.schema.json` 1.0.0, sends through an injectable transport, retries,
and sheds load when ingest is unavailable. Callers never receive transport or
validation exceptions.

## Quick start

```python
from sdk import InMemoryTransport, SdkClient, SdkSettings, log_turn

# Create once at process startup, not per turn.
client = SdkClient(settings=SdkSettings(), transport=InMemoryTransport())

status = log_turn(
    {
        "event_id": "3f2504e0-4f89-41d3-9a0c-0305e82c3301",
        "call_id": "call-0001",
        "turn_index": 0,
        "audio": {"uri": "s3://bucket/call-0001/turn-0000.wav"},
        "stt": {"text": "A guest utterance."},
    },
    client=client,
)
# status is EnqueueStatus.ACCEPTED or ERROR; it is diagnostic only.
```

Use `SdkClient()` without a transport for the httpx transport pointed at
`VOICELOOP_SDK_ENDPOINT`. `log_turn(event)` without a client uses a lazily
created process-wide client. Do not branch live-call logic on the return value.

## Fail-open behaviour

| Concern | Policy |
|---|---|
| Caller latency | Bounded queue append (`dict()` shallow copy only); default capacity 1024. |
| Queue full | Drop the oldest queued event and increment `dropped_oldest`; accept the new event. |
| Validation | `voiceloop_common.validate(event, "turn_event")` runs in the worker; failures increment `validation_failures`/`dropped_invalid` and are logged. |
| Send timeout | `timeout_seconds` passed to the transport for every attempt. |
| Retries | Up to `max_retries` extra attempts with `retry_backoff_seconds` delay. |
| Circuit breaker | Opens after `circuit_breaker_failure_threshold` consecutive failures; sheds later sends (`shed_by_breaker`), then allows one probe after `circuit_breaker_recovery_seconds`. |
| Exhausted retries | Dropped and counted in `dropped_after_retries`. |
| Exceptions | `enqueue`/`log_turn` catch all `Exception`s and return `ERROR`; no exception reaches the call path. KeyboardInterrupt/SystemExit are not swallowed. |

`SdkClient.metrics.snapshot()` returns an immutable `SdkMetricsSnapshot` with
`accepted`, `enqueue_errors`, `dropped_oldest`, `shed_by_breaker`, `sent`,
`send_failures`, `validation_failures`, `dropped_invalid`, `retried`,
`dropped_after_retries` and `internal_errors`. Structured warnings are emitted
from the worker with logger name `voiceloop.sdk`.

`wait_until_idle()`, `pending_event_ids()` and `close()` are test/shutdown
helpers; `SdkClient` is also a context manager. The worker daemon is started at
client construction.

## Settings

All environment variables use the `VOICELOOP_SDK_` prefix (typed via
`voiceloop_common.settings.BaseSettings`).

| Variable | Default | Meaning |
|---|---|---|
| `VOICELOOP_SDK_ENDPOINT` | `http://localhost:8000/v1/turns` | Ingest URL for the httpx transport. |
| `VOICELOOP_SDK_TIMEOUT_SECONDS` | `1.0` | Per-attempt transport timeout. |
| `VOICELOOP_SDK_QUEUE_CAPACITY` | `1024` | Maximum pending caller-side events. |
| `VOICELOOP_SDK_MAX_RETRIES` | `3` | Extra attempts after the first failure. |
| `VOICELOOP_SDK_RETRY_BACKOFF_SECONDS` | `0.25` | Delay before a retry. |
| `VOICELOOP_SDK_CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `5` | Failures before shedding. |
| `VOICELOOP_SDK_CIRCUIT_BREAKER_RECOVERY_SECONDS` | `30.0` | Time before a probe is allowed. |

## Transports

- `InMemoryTransport` — deterministic fake; records successful events and can
  simulate delay (`delay_seconds`) and initial failures (`failures_remaining`).
- `HttpTransport` — optional httpx-based `POST` transport. httpx is imported
  lazily on first send, and an injected `httpx.Client` is supported for tests.
- Any object implementing `TurnTransport.send(event, *, timeout_seconds)` can
  be injected.

## Audio capture

`capture_wav(path, uri=..., include_frames=...)` reads the WAV header with
stdlib `wave` and returns an `AudioCapture` with the **original** sample rate,
channel count and duration. `capture_audio` is an alias for WAV files;
`capture_pcm(frames, uri=..., sample_rate_hz=..., channels=...)` wraps raw PCM
frames. Nothing resamples: native 8 kHz telephony audio stays at 8 kHz and
`include_frames`/`capture_pcm` preserve the bytes unchanged.
`AudioCapture.event_audio()` returns the schema-shaped `audio` block.

## Verification

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/sdk -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy .
```

## Contracts and ownership

Consumes `turn_event.schema.json` 1.0.0 without changing it. This package owns
only `sdk/` root files; `sdk/pipecat/` (P1-04) and `sdk/connectors/` (P1-05) are
deliberately absent. No real guest audio or transcripts are used.
