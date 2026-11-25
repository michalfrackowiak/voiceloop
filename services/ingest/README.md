# Ingest service (P1-01)

Validates turn events against `schemas/turn_event.schema.json` 1.0.0, records
metadata in Postgres (in-memory fake by default), writes audio bytes only to an
S3-compatible object store, and accepts OpenTelemetry span payloads. Handlers
**validate, enqueue and return**: all storage I/O happens on a bounded,
fail-open background queue, so ingest cannot add latency to or break a live
call (`context.md` §2 invariant 1).

## Run

From the repository root (Python 3.11+):

```powershell
.\.venv\Scripts\python.exe -m uvicorn services.ingest.app:app --reload --port 8000
```

No database, object store, driver or network is required. The default settings
select in-memory fakes with the same interfaces as the real backends:

| Setting (prefix `VOICELOOP_INGEST_`) | Default | Purpose |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Structured log level. |
| `REPOSITORY_BACKEND` | `memory` | `postgres` requires a driver adapter (not wired in P1-01). |
| `OBJECT_STORE_BACKEND` | `memory` | `s3` requires a driver adapter (not wired in P1-01). |
| `DATABASE_URL` | unset | Reserved for the Postgres adapter; never contains secrets in the repo. |
| `QUEUE_CAPACITY` | `1024` | Bounded pending writes; a full queue sheds with a `shed` receipt. |
| `WORKER_POLL_INTERVAL_S` | `0.01` | Worker wake-up interval. |
| `WORKER_SHUTDOWN_TIMEOUT_S` | `2.0` | Shutdown/flush bound. |
| `AUTOSTART_WORKER` | `true` | Lazy background worker; tests disable it and call `process_pending()`. |
| `AUDIO_BUCKET` | `voiceloop-local` | Bucket name used in fake object URIs. |
| `AUDIO_URI_SCHEME` | `s3` | URI scheme for object references. |
| `PRESIGN_TTL_S` | `900` | Presigned PUT lifetime. |
| `PRESIGN_SECRET` | ephemeral random | Signing secret; generated per process, never committed. |
| `MAX_AUDIO_BYTES` | `26214400` | Local upload body limit. |

`configure_logging("ingest")` emits JSON logs with `service=ingest`. Logs carry
ids and timings only, never audio or transcript text.

## Endpoints

| Method | Path | Result |
|---|---|---|
| `POST` | `/v1/turns` | `202` receipt `{event_id, call_id, kind, status, audio_state, error}`. |
| `GET` | `/v1/turns/{event_id}` | Stored turn, metadata and current receipt; `404 not_found`. |
| `GET` | `/v1/calls/{call_id}` | Call row, its turns and merged outcome; `404 not_found`. |
| `PATCH` | `/v1/calls/{call_id}/outcome` | `202` outcome receipt; merged by `call_id`. |
| `POST` | `/v1/audio/uploads` | `201` presigned PUT URL, `upload_id`, object key and URI. |
| `PUT` | `/v1/audio/objects/{key}` | `201` local fake upload; query parameter `upload_id` required. |
| `POST` | `/v1/otel/spans` | `202` receipt for one span, or `{count, receipts}` for several. |
| `GET` | `/healthz` | Liveness plus `queued`, `processed`, `failed`, `shed`, `duplicates`. |

All errors use the shared shape from `docs/contracts.md`:
`{"code": "validation_error", "message": "...", "details": {...}}`.
Schema failures include `details.schema`, `details.field` and a redacted reason;
quoted values are removed so transcript text or PII cannot leak into errors.

### Idempotency and fail-open semantics

- `event_id` is claimed atomically before enqueue. The first response receipt is
  stored immutably; `GET` reports the live, updated receipt. A duplicate returns
  the **original receipt verbatim** (even after the first write is persisted),
  sets `X-VoiceLoop-Idempotent: hit`, and never creates a second turn.
- Receipt status moves `queued → persisted` after the worker writes; a storage
  failure moves it to `failed` with an `error` code. The caller always sees
  `202` for a valid event, regardless of storage state.
- A full queue drops the *new* item and returns `202` with `status: "shed"`,
  `error: "queue_full"`. This is the defined shedding signal; P1-02 may add a
  larger buffer and backpressure metrics.
- Model-version rows, call consent flags and per-turn metadata are written by
  the worker. The first non-null consent value per call is preserved; per-turn
  flags are stored as received. Audio bytes are never written to the metadata
  repository.
- Late outcomes merge into one `outcomes` row per `call_id`: later `updated_at`
  wins for status, `final_record` objects merge key by key, and `unknown` never
  overwrites a known status. An outcome for an unknown call creates a call row.
  Outcomes are stored **call-level**; `GET /v1/calls/{call_id}` returns a
  top-level `outcome` object and leaves each `turns[].outcome` as originally
  submitted. Consumers must join by `call_id` and must not expect per-turn
  propagation from this endpoint.

## Audio upload path

`POST /v1/audio/uploads` with `{call_id, turn_index, filename, content_type?}`
returns a presigned PUT URL plus the canonical `s3://<bucket>/calls/...` URI.
In production the SDK PUTs bytes directly to the object store using that URL.
For the in-memory fake, `PUT /v1/audio/objects/{key}?upload_id=...` accepts the
bytes; the id is checked against the issued upload and size is bounded. Then
`POST /v1/turns` references the returned URI in `audio.uri` (fixture events use
repo-relative URIs, which are recorded as external references).

## OpenTelemetry attribute mapping

`POST /v1/otel/spans` accepts a single span, an array of spans, or an OTLP
envelope (`resourceSpans[*].scopeSpans[*].spans[*]`). Attributes may be an
object, `[{"key": ..., "value": ...}]`, or OTLP typed values. Canonical names
and accepted short aliases:

| Attribute | Turn event field |
|---|---|
| `voiceloop.event_id` / `event_id` | `event_id` (uuid5 of trace/span id when absent) |
| `voiceloop.call_id` / `call_id` | `call_id` (required) |
| `voiceloop.turn_index` / `turn_index` | `turn_index` (required) |
| `voiceloop.ts` / `ts` | `ts` |
| `voiceloop.consent.recording` / `.training` | `consent.recording` / `consent.training` (both required if either is sent) |
| `voiceloop.audio.uri` / `audio_uri` | `audio.uri` (required) |
| `voiceloop.audio.sample_rate_hz` / `.channels` / `.duration_ms` | `audio.*` |
| `voiceloop.stt.text` / `hypothesis` | `stt.text` |
| `voiceloop.stt.model_id` / `.model_version` | `stt.model_id` / `stt.model_version` |
| `voiceloop.stt.words` (JSON array/string) | `stt.words` |
| `voiceloop.stt.decoder_stats` or `voiceloop.stt.avg_logprob` / `.compression_ratio` / `.no_speech_prob` | `stt.decoder_stats` |
| `voiceloop.parser.{model_id,model_version,input_text,output,schema_valid,logprob}` | `parser.*` |
| `voiceloop.dialogue_context.{expected_slot,reference_date,timezone,prior_turns_ref}` | `dialogue_context.*` |
| `voiceloop.outcome.{status,final_record,updated_at}` | `outcome.*` (merged by call id) |

Mapped events go through the same contract validator and idempotency ledger as
`POST /v1/turns`. The shared `voiceloop_common.validate(event, "turn_event")`
gate verifies the schema once at startup; request-time checks use a cached
validator compiled from that same schema because the shared helper re-checks
the meta-schema on every call (~0.4 s here), which would break the fail-open
latency invariant.

## Postgres migrations

`migrations/001_initial.sql` creates `model_versions`, `calls`, `turns`,
`outcomes` and the `ingest_receipts` idempotency ledger. Audio is referenced by
`turns.audio_uri`; there is no bytes column. Apply in filename order with
`psql -f` or your migration tool. The fake repository implements the same
`IngestRepository` interface; a real adapter must do the same.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/ingest -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy .
```

Tests ingest all 36 synthetic fixture events read-only from
`tests/fixtures/events/`, verify invalid-event error shapes, idempotency,
outcome merging, presigned uploads, OTel mapping, SQL parity, and a 150-request
storage-outage run that stays under a bounded latency and never raises.

## Known gaps

- Real Postgres and S3 adapters are interfaces only in P1-01; no driver is
  required or installed. Selecting those backends without injection raises a
  startup `ConfigError`.
- Buffered retry/flush of failed writes (beyond the bounded in-process queue)
  is P1-02. Consent enforcement and PII filtering are P1-09; ingest only
  preserves consent flags.
