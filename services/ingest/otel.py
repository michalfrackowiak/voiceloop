"""Map span-like OpenTelemetry payloads onto validated turn events.

Accepted inputs are a single span, a list of spans, or an OTLP-shaped envelope
with ``resourceSpans[*].scopeSpans[*].spans[*]``. Attributes may be a flat
object, a list of ``{"key": ..., "value": ...}`` pairs, or OTLP typed values
(``stringValue``/``intValue``/``boolValue``/``doubleValue``/``arrayValue``/
``kvlistValue``).

Canonical attribute mapping (short aliases without the ``voiceloop.`` prefix
are also accepted):

| Attribute | Turn event field |
|---|---|
| ``voiceloop.event_id`` | ``event_id`` (derived from trace/span id when absent) |
| ``voiceloop.call_id`` | ``call_id`` |
| ``voiceloop.turn_index`` | ``turn_index`` |
| ``voiceloop.ts`` | ``ts`` |
| ``voiceloop.consent.recording`` / ``.training`` | ``consent.*`` |
| ``voiceloop.audio.uri`` (alias ``audio_uri``) | ``audio.uri`` |
| ``voiceloop.audio.sample_rate_hz`` / ``.channels`` / ``.duration_ms`` | ``audio.*`` |
| ``voiceloop.stt.text`` (alias ``hypothesis``) | ``stt.text`` |
| ``voiceloop.stt.model_id`` / ``.model_version`` | ``stt.*`` |
| ``voiceloop.stt.words`` / ``.decoder_stats.*`` | ``stt.words`` / ``stt.decoder_stats`` |
| ``voiceloop.parser.*`` | ``parser.*`` |
| ``voiceloop.dialogue_context.*`` | ``dialogue_context.*`` |
| ``voiceloop.outcome.*`` | ``outcome.*`` (merged by call id) |
"""

from __future__ import annotations

import json
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from services.ingest.errors import IngestHTTPError

ATTRIBUTE_MAPPING: tuple[tuple[str, str], ...] = (
    ("voiceloop.event_id", "event_id"),
    ("voiceloop.call_id", "call_id"),
    ("voiceloop.turn_index", "turn_index"),
    ("voiceloop.ts", "ts"),
    ("voiceloop.consent.recording", "consent.recording"),
    ("voiceloop.consent.training", "consent.training"),
    ("voiceloop.audio.uri", "audio.uri"),
    ("voiceloop.audio.sample_rate_hz", "audio.sample_rate_hz"),
    ("voiceloop.audio.channels", "audio.channels"),
    ("voiceloop.audio.duration_ms", "audio.duration_ms"),
    ("voiceloop.stt.text", "stt.text"),
    ("voiceloop.stt.model_id", "stt.model_id"),
    ("voiceloop.stt.model_version", "stt.model_version"),
    ("voiceloop.parser.model_id", "parser.model_id"),
    ("voiceloop.parser.model_version", "parser.model_version"),
    ("voiceloop.dialogue_context.expected_slot", "dialogue_context.expected_slot"),
    ("voiceloop.outcome.status", "outcome.status"),
)

_MISSING = object()


def _bad_field(field: str, reason: str) -> IngestHTTPError:
    """Build a structured validation error for one span attribute."""
    return IngestHTTPError(
        422,
        "validation_error",
        "OpenTelemetry span could not be mapped to a turn event.",
        {"field": field, "reason": reason},
    )


def _typed_value(value: Any) -> Any:
    """Unwrap an OTLP typed attribute value into a plain JSON value."""
    if not isinstance(value, dict):
        return value
    for key in ("stringValue", "intValue", "boolValue", "doubleValue"):
        if key in value:
            return value[key]
    if "arrayValue" in value:
        items = (
            value["arrayValue"].get("values", []) if isinstance(value["arrayValue"], dict) else []
        )
        return [_typed_value(item.get("value", item)) for item in items if isinstance(item, dict)]
    if "kvlistValue" in value:
        entries = (
            value["kvlistValue"].get("values", []) if isinstance(value["kvlistValue"], dict) else []
        )
        return {
            str(entry.get("key")): _typed_value(entry.get("value"))
            for entry in entries
            if isinstance(entry, dict)
        }
    if "value" in value and "key" in value:
        return _typed_value(value["value"])
    return value


def attribute_map(value: Any) -> dict[str, Any]:
    """Normalise an attribute collection to a flat ``str -> JSON value`` map."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): _typed_value(item) for key, item in value.items()}
    if isinstance(value, list):
        result: dict[str, Any] = {}
        for entry in value:
            if isinstance(entry, dict) and "key" in entry:
                result[str(entry["key"])] = _typed_value(entry.get("value"))
            else:
                raise _bad_field("attributes", "expected key/value attribute objects")
        return result
    raise _bad_field("attributes", "expected an object or a list of key/value pairs")


def unwrap_spans(payload: Any) -> list[dict[str, Any]]:
    """Extract span dictionaries from a span, list of spans, or OTLP envelope."""
    if isinstance(payload, list):
        spans: list[dict[str, Any]] = []
        for item in payload:
            spans.extend(unwrap_spans(item))
        return spans
    if not isinstance(payload, dict):
        raise _bad_field("span", "expected a span object or an OTLP envelope")

    for nested in ("resourceSpans", "scopeSpans", "spans"):
        if nested in payload:
            children = payload[nested]
            if not isinstance(children, list):
                raise _bad_field(nested, "expected an array")
            spans = []
            for child in children:
                spans.extend(unwrap_spans(child))
            return spans
    return [payload]


def _as_int(value: Any, field: str) -> int:
    """Coerce an attribute to an integer, raising a structured error otherwise."""
    if isinstance(value, bool):
        raise _bad_field(field, "expected an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.lstrip("-").isdigit():
        return int(value)
    raise _bad_field(field, "expected an integer")


def _as_bool(value: Any, field: str) -> bool:
    """Coerce an attribute to a boolean, raising a structured error otherwise."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise _bad_field(field, "expected a boolean")


def _json_object(value: Any, field: str) -> dict[str, Any]:
    """Parse a JSON string or object attribute into a mapping."""
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, dict):
        raise _bad_field(field, "expected a JSON object")
    return parsed


def span_to_event(span: dict[str, Any]) -> dict[str, Any]:
    """Map one span's attributes to a turn-event payload for schema validation."""
    attrs = attribute_map(span.get("attributes"))

    def pick(*names: str) -> Any:
        """Return the first present attribute value, or a missing sentinel."""
        for name in names:
            if name in attrs:
                return attrs[name]
        return _MISSING

    trace_id = str(span.get("traceId") or span.get("trace_id") or "")
    span_id = str(span.get("spanId") or span.get("span_id") or "")
    event_id = pick("voiceloop.event_id", "event_id")
    if event_id is _MISSING:
        if span_id or trace_id:
            event_id = str(uuid5(NAMESPACE_URL, f"urn:voiceloop:otel:{trace_id}:{span_id}"))
        else:
            event_id = str(uuid4())

    call_id = pick("voiceloop.call_id", "call_id")
    if not isinstance(call_id, str) or not call_id:
        raise _bad_field("voiceloop.call_id", "missing or empty call id")
    turn_index = pick("voiceloop.turn_index", "turn_index")
    if turn_index is _MISSING:
        raise _bad_field("voiceloop.turn_index", "missing turn index")
    audio_uri = pick("voiceloop.audio.uri", "audio.uri", "audio_uri")
    if not isinstance(audio_uri, str) or not audio_uri:
        raise _bad_field("voiceloop.audio.uri", "missing or empty audio URI")
    text = pick("voiceloop.stt.text", "stt.text", "hypothesis")
    if text is _MISSING:
        text = ""

    audio: dict[str, Any] = {"uri": audio_uri}
    for suffix in ("sample_rate_hz", "channels", "duration_ms"):
        value = pick(f"voiceloop.audio.{suffix}", f"audio.{suffix}")
        if value is not _MISSING:
            audio[suffix] = _as_int(value, f"voiceloop.audio.{suffix}")
    stt: dict[str, Any] = {"text": text}
    for suffix in ("model_id", "model_version"):
        value = pick(f"voiceloop.stt.{suffix}", f"stt.{suffix}")
        if value is not _MISSING:
            stt[suffix] = value

    words = pick("voiceloop.stt.words", "stt.words")
    if words is not _MISSING:
        parsed = json.loads(words) if isinstance(words, str) else words
        if not isinstance(parsed, list):
            raise _bad_field("voiceloop.stt.words", "expected a JSON array")
        stt["words"] = parsed
    stats = pick("voiceloop.stt.decoder_stats", "stt.decoder_stats")
    decoder_stats = (
        _json_object(stats, "voiceloop.stt.decoder_stats") if stats is not _MISSING else {}
    )
    for suffix in ("avg_logprob", "compression_ratio", "no_speech_prob"):
        value = pick(f"voiceloop.stt.{suffix}", f"stt.{suffix}")
        if value is not _MISSING:
            decoder_stats[suffix] = value
    if decoder_stats:
        stt["decoder_stats"] = decoder_stats

    event: dict[str, Any] = {
        "event_id": str(event_id),
        "call_id": call_id,
        "turn_index": _as_int(turn_index, "voiceloop.turn_index"),
        "audio": audio,
        "stt": stt,
    }
    ts = pick("voiceloop.ts", "ts")
    if ts is not _MISSING:
        event["ts"] = ts

    recording = pick("voiceloop.consent.recording", "consent.recording")
    training = pick("voiceloop.consent.training", "consent.training")
    if recording is not _MISSING or training is not _MISSING:
        if recording is _MISSING or training is _MISSING:
            raise _bad_field(
                "voiceloop.consent", "recording and training must be supplied together"
            )
        event["consent"] = {
            "recording": _as_bool(recording, "voiceloop.consent.recording"),
            "training": _as_bool(training, "voiceloop.consent.training"),
        }

    parser: dict[str, Any] = {}
    for suffix in ("model_id", "model_version", "input_text", "output", "schema_valid", "logprob"):
        value = pick(f"voiceloop.parser.{suffix}", f"parser.{suffix}")
        if value is not _MISSING:
            parser[suffix] = (
                _json_object(value, f"voiceloop.parser.{suffix}") if suffix == "output" else value
            )
    if parser:
        event["parser"] = parser

    dialogue: dict[str, Any] = {}
    for suffix in ("expected_slot", "reference_date", "timezone", "prior_turns_ref"):
        value = pick(f"voiceloop.dialogue_context.{suffix}", f"dialogue_context.{suffix}")
        if value is not _MISSING:
            dialogue[suffix] = value
    if dialogue:
        event["dialogue_context"] = dialogue

    outcome: dict[str, Any] = {}
    for suffix in ("status", "final_record", "updated_at"):
        value = pick(f"voiceloop.outcome.{suffix}", f"outcome.{suffix}")
        if value is not _MISSING:
            outcome[suffix] = (
                _json_object(value, f"voiceloop.outcome.{suffix}")
                if suffix == "final_record"
                else value
            )
    if outcome:
        if "status" not in outcome:
            raise _bad_field(
                "voiceloop.outcome.status", "required when outcome attributes are present"
            )
        event["outcome"] = outcome
    return event
