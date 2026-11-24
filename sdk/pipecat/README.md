# Pipecat integration (P1-04)

Fail-open observer-style integration that captures one guest turn's user audio,
STT hypothesis (with word confidences), parser input/output, model IDs and
dialogue context, then emits a `turn_event.schema.json` 1.0.0 dict. Pipecat,
OpenTelemetry and the network are all optional; nothing here installs or
imports them at module load.

## Run the example

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m sdk.pipecat.example
```

`python -m sdk.pipecat.example` runs a scripted bot through three synthetic
turns (16 kHz, native 8 kHz and silence) into an in-memory local ingest sink,
validates each event with `validate_turn_event`, and prints JSON. The only
runtime requirements are the project environment and stdlib; no Pipecat, GPU,
service or credential is needed. Data is entirely invented.

## Public API

```python
from sdk.pipecat import (
    PipecatObserver,
    InMemoryTurnSink,
    SdkLogTurnSink,
    CapturedAudio,
    SttResult,
    ParserResult,
    DialogueContext,
    ConsentFlags,
    WordConfidence,
    TraceContext,
    DictSpan,
    build_turn_event,
    validate_turn_event,
    bind_conversation_ids,
    current_span,
    DecoderStats,
    default_sink,
)
```

STT frames may also supply a `decoder_stats` attribute; the observer captures it
into `DecoderStats` when present, so lead wiring can pass real decoder
statistics through unchanged.

- `PipecatObserver(call_id, conversation_id=..., sink=..., span_provider=...,
  validator=..., clock=..., event_id_factory=...)` — one instance per call.
  Feed it with `on_user_audio(frame)`, `on_stt(frame)`, `on_parser(frame)`,
  `set_dialogue_context(...)`, `set_consent(...)`, then `finish_turn()`.
  `attach(conversation)` registers those handlers on any `ConversationLike`
  adapter (`on_user_audio`, `on_stt`, `on_parser`, `on_turn_end`,
  `conversation_id`). `begin_turn()` resets per-turn state; `turn_index`
  increments after each successful send.
- `TurnSink` is a one-method protocol: `send(event: Mapping[str, Any]) -> None`.
  `InMemoryTurnSink(validator=validate_turn_event)` is the deterministic local
  ingest fake; `drain()` returns and clears stored events. `SdkLogTurnSink` is
  the default adapter: it imports and calls `sdk.log_turn(event)` **inside**
  `send`, so an incomplete SDK cannot break import. If `log_turn` is missing or
  raises, events are kept in `sink.events` and `used_fallback` becomes true.
  That in-memory fallback is the default for tests. A returned
  `EnqueueStatus.ERROR` is not a fallback trigger — the SDK already absorbs
  delivery failures in its background worker.
- `CapturedAudio.from_frame`, `SttResult.from_frame`, `ParserResult.from_frame`
  and `WordConfidence.from_frame` do defensive duck-typed extraction. Raw PCM
  bytes are stored unchanged; the declared sample rate is passed through, so
  8 kHz audio stays at 8 kHz and `duration_ms` is derived from the native rate
  unless supplied. Missing STT becomes an empty hypothesis (valid for silence).
- `build_turn_event(...)` is the pure builder; `validate_turn_event(event)`
  calls `voiceloop_common.validate(event, "turn_event")` (which resolves
  `schemas/turn_event.schema.json`).
- `PipecatObserver` does **not** validate synchronously: that would add work to
  the call path and violate `context.md` §2.1. Validation happens in the SDK
  background worker or in a local sink/test. Events are dropped with a counter
  rather than raising if an injected validator rejects them.

## Trace attributes

`TraceContext` is a dict carrier, and `SpanLike` is satisfied by any object
with `set_attribute(key, value)` — including a real OpenTelemetry span when
that package happens to be installed. `current_span()` lazily delegates to
`opentelemetry.trace.get_current_span()` and returns `None` when unavailable.
`bind_conversation_ids(...)` writes these names into both the dict context and
the active span; the observer calls it on every finished turn:

| Attribute | Value |
| --- | --- |
| `voiceloop.call_id` | Turn event `call_id`. |
| `voiceloop.conversation_id` | Observer/conversation ID, defaults to `call_id`. |
| `voiceloop.turn_index` | Zero-based turn number. |

Schema `additionalProperties: false` forbids trace fields in the event body,
so IDs are carried in the trace context only, not duplicated in the payload.

## Fail-open behaviour

Every public frame/lifecycle method catches `Exception`, increments
`observer.errors`, records `observer.last_error`, and returns normally.
Turns without audio are dropped (`dropped_events`); sink, span, validation and
frame conversion failures never propagate into the conversation loop.
Counters: `events_built`, `events_sent`, `errors`, `dropped_events`.

## Tests and contracts

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/pipecat -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy .
```

Consumes `turn_event.schema.json` 1.0.0 without changing it. No consent flags
are inferred: omit `consent` when unknown, or set both explicitly. No real
guest audio or transcripts are used. This package owns only `sdk/pipecat/**`;
wiring a deployed ingest URL/credentials is a lead follow-up.
