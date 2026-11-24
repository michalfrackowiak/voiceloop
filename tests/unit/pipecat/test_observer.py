"""Unit tests for the fail-open Pipecat observer."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import sdk
from schemas import validate as validate_schema
from sdk.pipecat import (
    CALL_ID_ATTRIBUTE,
    CONVERSATION_ID_ATTRIBUTE,
    TURN_INDEX_ATTRIBUTE,
    CapturedAudio,
    DictSpan,
    InMemoryTurnSink,
    PipecatObserver,
    SdkLogTurnSink,
    validate_turn_event,
)

from .fakes import (
    FakeAudioFrame,
    FakeConversation,
    FakeParserFrame,
    FakeSttFrame,
    FakeTurn,
    FakeWord,
)

FIXED_NOW = datetime(2026, 11, 25, 12, 0, 0, tzinfo=UTC)


def _two_turn_conversation() -> FakeConversation:
    """Build a fake call with a 16 kHz and a native 8 kHz turn."""
    return FakeConversation(
        "conv-123",
        [
            FakeTurn(
                audio=FakeAudioFrame(audio=b"\x01\x00" * 80, sample_rate_hz=16000),
                stt=FakeSttFrame(
                    text="Book a suite.",
                    words=(FakeWord(w="Book", start_ms=0, end_ms=100, conf=0.9),),
                    model_id="stt-fake",
                    model_version="1",
                ),
                parser=FakeParserFrame(
                    input_text="Book a suite.",
                    output={"room_type": "suite"},
                    model_id="parser-fake",
                    model_version="2",
                    schema_valid=True,
                    logprob=-0.2,
                ),
            ),
            FakeTurn(
                audio=FakeAudioFrame(audio=b"\xff\x00" * 40, sample_rate_hz=8000),
                stt=FakeSttFrame(text="Four guests."),
            ),
        ],
    )


def test_fake_conversation_produces_schema_valid_events() -> None:
    """The example driver yields events accepted by both validator APIs."""
    sink = InMemoryTurnSink(validator=validate_turn_event)
    observer = PipecatObserver(call_id="call-1", sink=sink)
    conversation = _two_turn_conversation().attach(observer)

    conversation.run()

    assert len(sink.events) == 2
    for event in sink.events:
        validate_schema(event, "turn_event.schema.json")
        validate_turn_event(event)
    first, second = sink.events
    assert first["call_id"] == "call-1"
    assert first["turn_index"] == 0
    assert first["stt"]["text"] == "Book a suite."
    assert first["stt"]["words"][0]["conf"] == 0.9
    assert first["stt"]["model_id"] == "stt-fake"
    assert first["parser"]["output"] == {"room_type": "suite"}
    assert first["parser"]["model_id"] == "parser-fake"
    assert "consent" not in first
    assert second["turn_index"] == 1
    assert observer.events_built == observer.events_sent == 2
    assert observer.errors == 0


def test_native_8khz_audio_bytes_are_unchanged() -> None:
    """PCM bytes and the declared 8 kHz rate pass through without resampling."""
    frames = b"\x00\x80" * 80
    sink = InMemoryTurnSink()
    observer = PipecatObserver(call_id="call-8k", sink=sink)
    observer.on_user_audio(FakeAudioFrame(audio=frames, sample_rate_hz=8000))

    event = observer.finish_turn()

    assert event is not None
    assert event["audio"]["sample_rate_hz"] == 8000
    assert event["audio"]["duration_ms"] == 20
    captured = CapturedAudio.from_frame(FakeAudioFrame(audio=frames, sample_rate_hz=8000))
    assert captured.frames == frames
    assert captured.sample_rate_hz == 8000
    validate_schema(event, "turn_event.schema.json")


def test_conversation_ids_reach_dict_context_and_span() -> None:
    """Trace shim writes documented attribute names to both carriers."""
    span = DictSpan()
    sink = InMemoryTurnSink()
    observer = PipecatObserver(
        call_id="call-2",
        conversation_id="conv-original",
        sink=sink,
        span_provider=lambda: span,
    )
    conversation = FakeConversation("conv-from-driver")
    conversation.attach(observer)
    conversation.turns = [
        FakeTurn(audio=FakeAudioFrame(audio=b"\x00\x00" * 16, sample_rate_hz=16000))
    ]
    conversation.run()

    expected = {
        CALL_ID_ATTRIBUTE: "call-2",
        CONVERSATION_ID_ATTRIBUTE: "conv-from-driver",
        TURN_INDEX_ATTRIBUTE: 0,
    }
    assert dict(observer.trace_context.attributes) == expected
    assert dict(span.attributes) == expected


def test_conversation_id_defaults_to_call_id() -> None:
    """A missing conversation ID falls back to the call ID."""
    observer = PipecatObserver(
        call_id="call-3", sink=InMemoryTurnSink(), span_provider=lambda: None
    )
    observer.on_user_audio(FakeAudioFrame(audio=b"\x00\x00", sample_rate_hz=8000))
    observer.finish_turn()
    assert observer.trace_context.get(CONVERSATION_ID_ATTRIBUTE) == "call-3"


def test_fail_open_on_bad_frame_missing_audio_and_failing_sink() -> None:
    """No frame, sink or validation failure reaches the conversation loop."""

    class BrokenSink:
        """Sink that always raises."""

        def send(self, event: Mapping[str, Any]) -> None:
            """Raise to simulate a broken local ingest."""
            raise RuntimeError("sink down")

    observer = PipecatObserver(call_id="call-4", sink=BrokenSink())
    observer.on_user_audio(FakeAudioFrame(audio=b"\x00\x00" * 16, sample_rate_hz=0))
    assert observer.errors == 1
    observer.on_stt(FakeSttFrame(text="Still alive."))
    assert observer.finish_turn() is None
    assert observer.dropped_events == 1
    observer.on_user_audio(FakeAudioFrame(audio=b"\x00\x00" * 16, sample_rate_hz=8000))
    assert observer.finish_turn() is None
    assert observer.dropped_events == 2
    assert observer.errors >= 2
    assert observer.events_built == 1
    assert observer.last_error is not None


def test_rejected_event_is_dropped_and_next_turn_succeeds() -> None:
    """An injected validator can drop a turn without breaking the observer."""
    sink = InMemoryTurnSink()

    def reject(event: Mapping[str, Any]) -> None:
        """Reject the first turn only."""
        if event["turn_index"] == 0:
            raise ValueError("bad event")

    observer = PipecatObserver(call_id="call-5", sink=sink, validator=reject)
    for _ in range(2):
        observer.on_user_audio(FakeAudioFrame(audio=b"\x00\x00" * 16, sample_rate_hz=8000))
        observer.finish_turn()
    assert observer.dropped_events == 1
    assert [event["turn_index"] for event in sink.events] == [1]


def test_deterministic_ids_and_timestamp() -> None:
    """Injected clock and ID factory make the builder deterministic."""
    sink = InMemoryTurnSink()
    observer = PipecatObserver(
        call_id="call-6",
        sink=sink,
        clock=lambda: FIXED_NOW,
        event_id_factory=lambda: "11111111-1111-4111-8111-111111111111",
    )
    observer.on_user_audio(FakeAudioFrame(audio=b"\x00\x00", sample_rate_hz=8000))
    event = observer.finish_turn()
    assert event is not None
    assert event["event_id"] == "11111111-1111-4111-8111-111111111111"
    assert event["ts"] == "2026-11-25T12:00:00Z"
    validate_schema(event, "turn_event.schema.json")


def test_default_sdk_sink_calls_log_turn_lazily(monkeypatch: Any) -> None:
    """The default adapter imports and calls the SDK only when sending."""
    calls: list[dict[str, Any]] = []

    def fake_log_turn(event: dict[str, Any], *, client: Any = None) -> None:
        """Record the event passed by the sink."""
        calls.append(event)

    monkeypatch.setattr(sdk, "log_turn", fake_log_turn)
    sink = SdkLogTurnSink()
    sink.send({"event_id": "x"})
    assert calls == [{"event_id": "x"}]
    assert not sink.used_fallback


def test_default_sdk_sink_falls_back_when_sdk_is_unavailable(monkeypatch: Any) -> None:
    """A missing SDK routes events into the documented in-memory fallback."""
    monkeypatch.setitem(sys.modules, "sdk", None)
    sink = SdkLogTurnSink()
    sink.send({"event_id": "y"})
    assert sink.used_fallback
    assert sink.events == [{"event_id": "y"}]
    assert sink.last_error is not None
