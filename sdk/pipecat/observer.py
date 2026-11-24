"""Fail-open observer that turns Pipecat-like frames into VoiceLoop turn events.

Pipecat itself is optional. The protocols below describe only the attributes the
observer reads, so a real Pipecat adapter, a test fake or the scripted example
can all drive the same methods. Nothing in the observer validates, logs over
the network or raises into the conversation loop: validation is available as
:func:`validate_turn_event` and runs off the call path (in the SDK worker or a
local sink), while every frame handler swallows its own failures.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from voiceloop_common.clock import utcnow
from voiceloop_common.ids import new_id

from .traces import SpanLike, TraceContext, current_span


class FrameLike(Protocol):
    """Marker protocol for any Pipecat-like frame."""


class UserAudioFrameLike(FrameLike, Protocol):
    """Frame carrying user PCM bytes and their declared native sample rate."""

    @property
    def audio(self) -> bytes:
        """Raw PCM bytes, unchanged."""

    @property
    def sample_rate_hz(self) -> int:
        """Sample rate declared by the capture source (8 kHz stays 8 kHz)."""


class WordLike(FrameLike, Protocol):
    """One STT word with a confidence."""

    @property
    def w(self) -> str:
        """Word text."""

    @property
    def start_ms(self) -> int:
        """Word start offset in milliseconds."""

    @property
    def end_ms(self) -> int:
        """Word end offset in milliseconds."""

    @property
    def conf(self) -> float:
        """Word confidence in ``[0, 1]``."""


class SttFrameLike(FrameLike, Protocol):
    """STT hypothesis frame; optional members are read with ``getattr``."""

    @property
    def text(self) -> str:
        """Hypothesis text (may be empty for silence)."""


class ParserFrameLike(FrameLike, Protocol):
    """Parser input/output frame; optional members are read with ``getattr``."""

    @property
    def input_text(self) -> str:
        """Text the parser received."""

    @property
    def output(self) -> Mapping[str, Any]:
        """Structured slots produced by the parser."""


class ConversationLike(Protocol):
    """Minimal event-registration surface an adapter must provide."""

    @property
    def conversation_id(self) -> str:
        """Conversation identifier propagated to traces."""

    def on_user_audio(self, handler: Callable[[UserAudioFrameLike], None]) -> None:
        """Register a user-audio handler."""

    def on_stt(self, handler: Callable[[SttFrameLike], None]) -> None:
        """Register an STT-result handler."""

    def on_parser(self, handler: Callable[[ParserFrameLike], None]) -> None:
        """Register a parser-result handler."""

    def on_turn_end(self, handler: Callable[[], Any]) -> None:
        """Register a handler invoked when a turn completes."""


class TurnSink(Protocol):
    """Narrow, injectable destination for a completed turn event."""

    def send(self, event: Mapping[str, Any]) -> None:
        """Accept one event; implementations must not require a response."""


@dataclass
class InMemoryTurnSink:
    """Deterministic local sink; ``send`` stores deep copies of events."""

    events: list[dict[str, Any]] = field(default_factory=list)
    validator: Callable[[Mapping[str, Any]], None] | None = None

    def send(self, event: Mapping[str, Any]) -> None:
        """Validate optionally, then append a snapshot of ``event``."""
        snapshot: dict[str, Any] = copy.deepcopy(dict(event))
        if self.validator is not None:
            self.validator(snapshot)
        self.events.append(snapshot)

    def drain(self) -> list[dict[str, Any]]:
        """Return and clear all stored events."""
        drained = self.events
        self.events = []
        return drained


class SdkLogTurnSink:
    """Default sink: call ``sdk.log_turn`` lazily, with an in-memory fallback.

    ``sdk`` is imported inside :meth:`send` so an incomplete or unavailable SDK
    cannot break importing this module. If the import fails or ``log_turn``
    raises, subsequent events are kept in ``events`` and ``used_fallback`` is
    set. A returned ``EnqueueStatus.ERROR`` is *not* a fallback trigger: the SDK
    already absorbs delivery failures in its background worker.
    """

    def __init__(self) -> None:
        """Create the sink with an empty fallback buffer."""
        self._fallback = InMemoryTurnSink()
        self.used_fallback = False
        self.last_error: str | None = None

    def send(self, event: Mapping[str, Any]) -> None:
        """Queue via ``sdk.log_turn`` or record the event in memory."""
        try:
            from sdk import log_turn
        except Exception as exc:
            self._use_fallback(event, exc)
            return
        try:
            log_turn(dict(event))
        except Exception as exc:
            self._use_fallback(event, exc)

    @property
    def events(self) -> list[dict[str, Any]]:
        """Return events stored by the in-memory fallback."""
        return self._fallback.events

    def _use_fallback(self, event: Mapping[str, Any], exc: Exception) -> None:
        """Record a delivery failure and keep the event locally."""
        self.used_fallback = True
        self.last_error = f"{type(exc).__name__}: {exc}"
        self._fallback.send(event)


def default_sink() -> TurnSink:
    """Return the default SDK-backed sink with in-memory fallback."""
    return SdkLogTurnSink()


def _first_attr(value: Any, *names: str) -> Any:
    """Return the first readable attribute from ``names``, else ``None``."""
    for name in names:
        try:
            found = getattr(value, name)
        except Exception:
            continue
        if found is not None:
            return found
    return None


def _nonempty_str(value: Any) -> str | None:
    """Return a non-empty string, or ``None`` for anything else."""
    return value if isinstance(value, str) and value else None


def _as_text(value: Any) -> str:
    """Coerce a hypothesis/input value to text, defaulting to empty."""
    return value if isinstance(value, str) else ""


def _as_int(value: Any, *, minimum: int = 0) -> int:
    """Coerce to an integer clamped at ``minimum``; never raise."""
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return minimum
    return max(minimum, number)


def _as_float(value: Any, *, minimum: float | None = None, maximum: float | None = None) -> float:
    """Coerce to a float clamped into the requested range; never raise."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        number = 0.0
    if not math.isfinite(number):
        return 0.0
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number


@dataclass(frozen=True)
class WordConfidence:
    """Schema-shaped word confidence with safe conversion from frames."""

    w: str
    start_ms: int = 0
    end_ms: int = 0
    conf: float = 0.0

    @classmethod
    def from_frame(cls, value: Any) -> WordConfidence | None:
        """Build from a duck-typed word; skip entries without usable text."""
        text = _nonempty_str(_first_attr(value, "w", "word", "text"))
        if text is None:
            return None
        return cls(
            w=text,
            start_ms=_as_int(_first_attr(value, "start_ms", "start")),
            end_ms=_as_int(_first_attr(value, "end_ms", "end")),
            conf=_as_float(_first_attr(value, "conf", "confidence"), minimum=0.0, maximum=1.0),
        )

    def event_dict(self) -> dict[str, Any]:
        """Return the schema's ``words`` item."""
        return {
            "w": self.w,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "conf": self.conf,
        }


@dataclass(frozen=True)
class DecoderStats:
    """Whisper-family decoder signals required as a group by the schema."""

    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float

    @classmethod
    def from_frame(cls, value: Any) -> DecoderStats | None:
        """Build when all three values are readable, otherwise return ``None``."""
        if value is None:
            return None
        return cls(
            avg_logprob=_as_float(_first_attr(value, "avg_logprob")),
            compression_ratio=_as_float(_first_attr(value, "compression_ratio"), minimum=0.0),
            no_speech_prob=_as_float(
                _first_attr(value, "no_speech_prob"), minimum=0.0, maximum=1.0
            ),
        )

    def event_dict(self) -> dict[str, Any]:
        """Return the schema's ``decoder_stats`` block."""
        return {
            "avg_logprob": self.avg_logprob,
            "compression_ratio": self.compression_ratio,
            "no_speech_prob": self.no_speech_prob,
        }


@dataclass(frozen=True)
class SttResult:
    """Hypothesis, word confidences and model identity for one turn."""

    text: str = ""
    words: tuple[WordConfidence, ...] = ()
    model_id: str | None = None
    model_version: str | None = None
    decoder_stats: DecoderStats | None = None

    @classmethod
    def from_frame(cls, frame: SttFrameLike) -> SttResult:
        """Extract an STT result from a duck-typed frame."""
        raw_words = _first_attr(frame, "words", "word_confidences")
        words: tuple[WordConfidence, ...] = ()
        if isinstance(raw_words, Sequence) and not isinstance(raw_words, (str, bytes)):
            parsed = (WordConfidence.from_frame(item) for item in raw_words)
            words = tuple(word for word in parsed if word is not None)
        return cls(
            text=_as_text(_first_attr(frame, "text", "hypothesis", "transcript")),
            words=words,
            model_id=_nonempty_str(_first_attr(frame, "model_id")),
            model_version=_nonempty_str(_first_attr(frame, "model_version")),
            decoder_stats=DecoderStats.from_frame(_first_attr(frame, "decoder_stats")),
        )

    def event_dict(self) -> dict[str, Any]:
        """Return the schema's ``stt`` block."""
        block: dict[str, Any] = {"text": self.text}
        if self.model_id is not None:
            block["model_id"] = self.model_id
        if self.model_version is not None:
            block["model_version"] = self.model_version
        if self.words:
            block["words"] = [word.event_dict() for word in self.words]
        if self.decoder_stats is not None:
            block["decoder_stats"] = self.decoder_stats.event_dict()
        return block


@dataclass(frozen=True)
class ParserResult:
    """Parser input, output, validity, confidence and model identity."""

    input_text: str | None = None
    output: dict[str, Any] | None = None
    model_id: str | None = None
    model_version: str | None = None
    schema_valid: bool | None = None
    logprob: float | None = None

    @classmethod
    def from_frame(cls, frame: ParserFrameLike) -> ParserResult:
        """Extract a parser result from a duck-typed frame."""
        raw_output = _first_attr(frame, "output", "slots")
        output: dict[str, Any] | None = None
        if isinstance(raw_output, Mapping):
            output = dict(raw_output)
        raw_valid = _first_attr(frame, "schema_valid", "valid")
        raw_logprob = _first_attr(frame, "logprob")
        return cls(
            input_text=_nonempty_str(_first_attr(frame, "input_text", "text")),
            output=output,
            model_id=_nonempty_str(_first_attr(frame, "model_id")),
            model_version=_nonempty_str(_first_attr(frame, "model_version")),
            schema_valid=raw_valid if isinstance(raw_valid, bool) else None,
            logprob=_as_float(raw_logprob) if raw_logprob is not None else None,
        )

    def event_dict(self) -> dict[str, Any]:
        """Return the schema's ``parser`` block (only populated fields)."""
        block: dict[str, Any] = {}
        for key in ("model_id", "model_version", "input_text", "output", "schema_valid", "logprob"):
            value = getattr(self, key)
            if value is not None:
                block[key] = value
        return block


@dataclass(frozen=True)
class DialogueContext:
    """Context the parser saw; empty members are omitted from the event."""

    expected_slot: str | None = None
    reference_date: str | None = None
    timezone: str | None = None
    prior_turns_ref: str | None = None

    def event_dict(self) -> dict[str, Any]:
        """Return the schema's ``dialogue_context`` block."""
        block: dict[str, Any] = {}
        for key in ("expected_slot", "reference_date", "timezone", "prior_turns_ref"):
            value = getattr(self, key)
            if value is not None:
                block[key] = value
        return block


@dataclass(frozen=True)
class ConsentFlags:
    """Per-call consent flags; never inferred and never defaulted to true."""

    recording: bool
    training: bool

    def event_dict(self) -> dict[str, bool]:
        """Return the schema's ``consent`` block."""
        return {"recording": self.recording, "training": self.training}


@dataclass(frozen=True)
class CapturedAudio:
    """Native-rate PCM bytes plus the metadata needed by the ``audio`` block."""

    frames: bytes
    sample_rate_hz: int
    channels: int = 1
    sample_width_bytes: int = 2
    uri: str = ""
    duration_ms: int | None = None

    @classmethod
    def from_frame(cls, frame: UserAudioFrameLike) -> CapturedAudio:
        """Wrap user audio bytes without resampling or copying values."""
        raw = _first_attr(frame, "audio", "frames", "pcm", "data")
        if isinstance(raw, (bytes, bytearray, memoryview)):
            frames = bytes(raw)
        else:
            raise ValueError("audio frame has no bytes-like PCM payload")
        sample_rate_hz = _as_int(_first_attr(frame, "sample_rate_hz", "sample_rate"))
        channels = _as_int(_first_attr(frame, "channels"), minimum=1)
        width = _as_int(_first_attr(frame, "sample_width_bytes"), minimum=1)
        if sample_rate_hz < 1:
            raise ValueError("audio frame has no positive sample_rate_hz")
        uri = _nonempty_str(_first_attr(frame, "uri")) or ""
        duration = _first_attr(frame, "duration_ms")
        duration_ms = _as_int(duration) if duration is not None else None
        return cls(
            frames=frames,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            sample_width_bytes=width,
            uri=uri,
            duration_ms=duration_ms,
        )

    def event_dict(self, *, uri: str) -> dict[str, Any]:
        """Return the schema's ``audio`` block with a non-empty URI."""
        if self.duration_ms is not None:
            duration_ms = self.duration_ms
        else:
            bytes_per_second = self.sample_rate_hz * self.channels * self.sample_width_bytes
            duration_ms = round(len(self.frames) * 1000 / bytes_per_second)
        return {
            "uri": uri or self.uri,
            "sample_rate_hz": self.sample_rate_hz,
            "channels": self.channels,
            "duration_ms": duration_ms,
        }


def build_turn_event(
    *,
    call_id: str,
    turn_index: int,
    audio: CapturedAudio,
    stt: SttResult | None = None,
    parser: ParserResult | None = None,
    dialogue_context: DialogueContext | None = None,
    consent: ConsentFlags | None = None,
    ts: str | None = None,
    event_id: str | None = None,
    clock: Callable[[], datetime] = utcnow,
) -> dict[str, Any]:
    """Build a schema-shaped turn event from captured turn data.

    Args:
        call_id: Non-empty call identifier.
        turn_index: Zero-based index within the call.
        audio: Captured native-rate audio bytes and metadata.
        stt: Hypothesis; an empty hypothesis is used when omitted (silence).
        parser: Optional parser result.
        dialogue_context: Optional parser context.
        consent: Optional explicit consent flags; omitted when unknown.
        ts: Optional RFC 3339 timestamp; uses the injected UTC clock otherwise.
        event_id: Optional UUID; generated otherwise.
        clock: Injectable UTC clock for deterministic tests.

    Returns:
        A dict that validates against ``turn_event.schema.json`` 1.0.0.

    Raises:
        ValueError: If ``call_id`` is empty or ``turn_index`` is negative.

    """
    if not call_id:
        raise ValueError("call_id must be non-empty")
    if turn_index < 0:
        raise ValueError("turn_index must be >= 0")
    stamp = ts if ts is not None else clock().isoformat().replace("+00:00", "Z")
    uri = audio.uri or f"memory://voiceloop/{call_id}/{turn_index}.pcm"
    event: dict[str, Any] = {
        "event_id": event_id or new_id(),
        "call_id": call_id,
        "turn_index": turn_index,
        "ts": stamp,
        "audio": audio.event_dict(uri=uri),
        "stt": (stt if stt is not None else SttResult(text="")).event_dict(),
    }
    if parser is not None:
        event["parser"] = parser.event_dict()
    if dialogue_context is not None:
        event["dialogue_context"] = dialogue_context.event_dict()
    if consent is not None:
        event["consent"] = consent.event_dict()
    return event


def validate_turn_event(event: Mapping[str, Any]) -> None:
    """Validate an event against ``turn_event.schema.json`` 1.0.0.

    The runtime helper from ``voiceloop_common`` is imported lazily so this
    module and the stdlib-only example do not require it at import time.

    Raises:
        voiceloop_common.errors.ValidationError: If the event is invalid.

    """
    from voiceloop_common.schemas import validate

    validate(dict(event), "turn_event")


@dataclass
class _TurnState:
    """Mutable per-turn capture state kept inside the observer."""

    audio: CapturedAudio | None = None
    stt: SttResult | None = None
    parser: ParserResult | None = None
    context: DialogueContext | None = None
    consent: ConsentFlags | None = None


class PipecatObserver:
    """Capture frames for one turn, then emit a fail-open turn event.

    Create one observer per conversation. Frame handlers are safe to call from
    a conversation loop: every public method catches exceptions, counts them in
    :attr:`errors` and records the latest message in :attr:`last_error`.
    """

    def __init__(
        self,
        *,
        call_id: str,
        conversation_id: str | None = None,
        sink: TurnSink | None = None,
        span_provider: Callable[[], SpanLike | None] | None = None,
        validator: Callable[[Mapping[str, Any]], None] | None = None,
        clock: Callable[[], datetime] = utcnow,
        event_id_factory: Callable[[], str] = new_id,
    ) -> None:
        """Configure the observer's identifiers, sink and test seams.

        Args:
            call_id: Non-empty call identifier used in every emitted event.
            conversation_id: Trace conversation ID; defaults to ``call_id``.
            sink: Destination for completed events; defaults to the lazy SDK sink.
            span_provider: Optional OTel-like current-span provider.
            validator: Optional off-call-path validator; ``None`` defers to the sink.
            clock: Injectable UTC clock.
            event_id_factory: Injectable UUID factory for deterministic tests.

        Raises:
            ValueError: If ``call_id`` is empty.

        """
        if not call_id:
            raise ValueError("call_id must be non-empty")
        self.call_id = call_id
        self.conversation_id = conversation_id or call_id
        self._sink = sink if sink is not None else SdkLogTurnSink()
        self._span_provider = span_provider if span_provider is not None else current_span
        self._validator = validator
        self._clock = clock
        self._event_id_factory = event_id_factory
        self._state = _TurnState()
        self._turn_index = 0
        self.trace_context = TraceContext()
        self.errors = 0
        self.dropped_events = 0
        self.events_built = 0
        self.events_sent = 0
        self.last_error: str | None = None

    def attach(self, conversation: ConversationLike) -> None:
        """Register handlers on a conversation-like adapter; never raises."""
        try:
            conversation.on_user_audio(self.on_user_audio)
            conversation.on_stt(self.on_stt)
            conversation.on_parser(self.on_parser)
            conversation.on_turn_end(self.finish_turn)
            identifier = _nonempty_str(conversation.conversation_id)
            if identifier is not None:
                self.conversation_id = identifier
        except Exception as exc:
            self._record_error("attach", exc)

    def begin_turn(
        self,
        *,
        turn_index: int | None = None,
        dialogue_context: DialogueContext | None = None,
        consent: ConsentFlags | None = None,
    ) -> None:
        """Clear captured state for a new turn; never raises."""
        try:
            if turn_index is not None:
                self._turn_index = _as_int(turn_index)
            self._state = _TurnState(context=dialogue_context, consent=consent)
        except Exception as exc:
            self._record_error("begin_turn", exc)

    def on_user_audio(self, frame: UserAudioFrameLike) -> None:
        """Capture native-rate user audio bytes; never raises."""
        try:
            self._state.audio = CapturedAudio.from_frame(frame)
        except Exception as exc:
            self._record_error("user_audio", exc)

    def on_stt(self, frame: SttFrameLike) -> None:
        """Capture the STT hypothesis and word confidences; never raises."""
        try:
            self._state.stt = SttResult.from_frame(frame)
        except Exception as exc:
            self._record_error("stt", exc)

    def on_parser(self, frame: ParserFrameLike) -> None:
        """Capture parser input, output and model identity; never raises."""
        try:
            self._state.parser = ParserResult.from_frame(frame)
        except Exception as exc:
            self._record_error("parser", exc)

    def set_dialogue_context(self, context: DialogueContext) -> None:
        """Set the context the parser saw for this turn; never raises."""
        try:
            self._state.context = context
        except Exception as exc:
            self._record_error("dialogue_context", exc)

    def set_consent(self, consent: ConsentFlags) -> None:
        """Set explicit consent flags for this call; never raises."""
        try:
            self._state.consent = consent
        except Exception as exc:
            self._record_error("consent", exc)

    def finish_turn(
        self,
        *,
        call_id: str | None = None,
        conversation_id: str | None = None,
    ) -> Mapping[str, Any] | None:
        """Build, annotate and send the turn event; never raises.

        Args:
            call_id: Optional per-turn override of the configured call ID.
            conversation_id: Optional per-turn trace conversation ID.

        Returns:
            The emitted event, or ``None`` when the turn was dropped.

        """
        try:
            return self._finish_turn(call_id, conversation_id)
        except Exception as exc:
            self.dropped_events += 1
            self._record_error("finish_turn", exc)
            self._state = _TurnState()
            return None

    def _finish_turn(
        self,
        call_id_override: str | None,
        conversation_id_override: str | None,
    ) -> Mapping[str, Any] | None:
        """Perform the build/annotate/send sequence for one turn."""
        audio = self._state.audio
        if audio is None:
            self.dropped_events += 1
            self.last_error = "finish_turn: no user audio captured"
            self._turn_index += 1
            self._state = _TurnState()
            return None
        index = self._turn_index
        event = build_turn_event(
            call_id=call_id_override or self.call_id,
            turn_index=index,
            audio=audio,
            stt=self._state.stt,
            parser=self._state.parser,
            dialogue_context=self._state.context,
            consent=self._state.consent,
            clock=self._clock,
            event_id=self._event_id_factory(),
        )
        if self._validator is not None:
            try:
                self._validator(event)
            except Exception as exc:
                self.dropped_events += 1
                self._record_error("validation", exc)
                self._turn_index += 1
                self._state = _TurnState()
                return None
        self.events_built += 1
        self._bind_trace(event["call_id"], conversation_id_override, index)
        try:
            self._sink.send(event)
        except Exception as exc:
            self.dropped_events += 1
            self._record_error("sink", exc)
            self._turn_index += 1
            self._state = _TurnState()
            return None
        self.events_sent += 1
        self._turn_index += 1
        self._state = _TurnState()
        return event

    def _bind_trace(self, call_id: str, conversation_id: str | None, turn_index: int) -> None:
        """Attach call/conversation IDs to the dict context and current span."""
        span: SpanLike | None = None
        try:
            span = self._span_provider()
        except Exception as exc:
            self._record_error("span_provider", exc)
        try:
            self.trace_context.bind(
                call_id,
                conversation_id or self.conversation_id,
                span=span,
                turn_index=turn_index,
            )
        except Exception as exc:
            self._record_error("trace_context", exc)

    def _record_error(self, stage: str, exc: Exception) -> None:
        """Count and remember a swallowed failure."""
        self.errors += 1
        self.last_error = f"{stage}: {type(exc).__name__}: {exc}"
