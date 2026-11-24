r"""Runnable stdlib-only example of a scripted Pipecat-style bot.

No Pipecat, network, GPU or OpenTelemetry package is needed. The bot drives the
observer with fake frames and writes schema-valid turn events into a local
in-memory "ingest" sink. Run from the repository root with::

    .\\.venv\\Scripts\\python.exe -m sdk.pipecat.example
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .observer import (
    ConsentFlags,
    DialogueContext,
    InMemoryTurnSink,
    PipecatObserver,
    WordConfidence,
    validate_turn_event,
)
from .traces import DictSpan

EXAMPLE_CALL_ID = "call-local-example"
EXAMPLE_CONVERSATION_ID = "conv-local-example"


@dataclass(frozen=True)
class _AudioFrame:
    """Fake user-audio frame with native-rate PCM bytes."""

    audio: bytes
    sample_rate_hz: int
    channels: int = 1


@dataclass(frozen=True)
class _SttFrame:
    """Fake STT frame with word confidences."""

    text: str
    model_id: str
    model_version: str
    words: tuple[WordConfidence, ...] = ()


@dataclass(frozen=True)
class _ParserFrame:
    """Fake parser frame with slots."""

    input_text: str
    output: dict[str, Any]
    model_id: str
    model_version: str
    schema_valid: bool
    logprob: float


def run_example(
    *,
    call_id: str = EXAMPLE_CALL_ID,
    conversation_id: str = EXAMPLE_CONVERSATION_ID,
    span: DictSpan | None = None,
) -> list[dict[str, Any]]:
    """Drive three synthetic turns and return validated local events.

    Turn 1 uses a 16 kHz hypothesis with word confidences; turn 2 uses native
    8 kHz telephony audio; turn 3 is silence. All data is invented and contains
    no real guest audio or transcript.

    Args:
        call_id: Call identifier for the synthetic call.
        conversation_id: Conversation identifier attached to the trace shim.
        span: Optional span fake to inspect bound trace attributes.

    Returns:
        Three turn-event dicts that validate against
        ``turn_event.schema.json`` 1.0.0.

    """
    sink = InMemoryTurnSink(validator=validate_turn_event)
    trace_span = span if span is not None else DictSpan()
    observer = PipecatObserver(
        call_id=call_id,
        conversation_id=conversation_id,
        sink=sink,
        span_provider=lambda: trace_span,
    )
    observer.set_consent(ConsentFlags(recording=True, training=True))
    observer.set_dialogue_context(
        DialogueContext(
            expected_slot="check_in",
            reference_date="2026-11-25",
            timezone="UTC",
            prior_turns_ref=None,
        )
    )

    observer.on_user_audio(_AudioFrame(audio=b"\x01\x00" * 160, sample_rate_hz=16000))
    observer.on_stt(
        _SttFrame(
            text="Book a deluxe room for two nights.",
            model_id="stt-fixture",
            model_version="1.0",
            words=(
                WordConfidence(w="Book", start_ms=0, end_ms=120, conf=0.98),
                WordConfidence(w="deluxe", start_ms=180, end_ms=420, conf=0.91),
            ),
        )
    )
    observer.on_parser(
        _ParserFrame(
            input_text="Book a deluxe room for two nights.",
            output={"room_type": "deluxe", "guests": 2},
            model_id="parser-fixture",
            model_version="1.0",
            schema_valid=False,
            logprob=-0.4,
        )
    )
    observer.finish_turn()

    observer.begin_turn()
    observer.on_user_audio(_AudioFrame(audio=b"\x02\x00" * 80, sample_rate_hz=8000))
    observer.on_stt(
        _SttFrame(
            text="Four guests please.",
            model_id="stt-fixture",
            model_version="1.0",
            words=(WordConfidence(w="Four", conf=0.62),),
        )
    )
    observer.on_parser(
        _ParserFrame(
            input_text="Four guests please.",
            output={"guests": 4},
            model_id="parser-fixture",
            model_version="1.0",
            schema_valid=False,
            logprob=-0.8,
        )
    )
    observer.finish_turn()

    observer.begin_turn()
    observer.on_user_audio(_AudioFrame(audio=b"\x00\x00" * 40, sample_rate_hz=8000))
    observer.on_stt(_SttFrame(text="", model_id="stt-fixture", model_version="1.0"))
    observer.finish_turn()

    return sink.drain()


def main() -> None:
    """Print the example events and their trace attributes as JSON."""
    events = run_example()
    print(json.dumps({"event_count": len(events), "events": events}, indent=2))


if __name__ == "__main__":
    main()
