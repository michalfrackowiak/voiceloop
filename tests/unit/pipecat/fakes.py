"""Deterministic fake conversation driver for Pipecat observer tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sdk.pipecat.observer import PipecatObserver


@dataclass(frozen=True)
class FakeWord:
    """Word confidence shaped like a Pipecat STT word."""

    w: str
    start_ms: int = 0
    end_ms: int = 0
    conf: float = 0.0


@dataclass(frozen=True)
class FakeAudioFrame:
    """User audio frame with a declared native sample rate."""

    audio: bytes
    sample_rate_hz: int
    channels: int = 1
    uri: str = ""


@dataclass(frozen=True)
class FakeSttFrame:
    """STT hypothesis frame."""

    text: str
    words: tuple[FakeWord, ...] = ()
    model_id: str | None = None
    model_version: str | None = None
    decoder_stats: Any = None


@dataclass(frozen=True)
class FakeParserFrame:
    """Parser output frame."""

    input_text: str
    output: dict[str, Any] = field(default_factory=dict)
    model_id: str | None = None
    model_version: str | None = None
    schema_valid: bool | None = None
    logprob: float | None = None


@dataclass(frozen=True)
class FakeTurn:
    """One scripted conversation turn."""

    audio: FakeAudioFrame
    stt: FakeSttFrame | None = None
    parser: FakeParserFrame | None = None


class FakeConversation:
    """Minimal ``ConversationLike`` implementation driven by a script."""

    def __init__(self, conversation_id: str, turns: list[FakeTurn] | None = None) -> None:
        """Create a fake with an ID and optional scripted turns."""
        self.conversation_id = conversation_id
        self.turns = turns or []
        self._audio: Callable[[FakeAudioFrame], None] | None = None
        self._stt: Callable[[FakeSttFrame], None] | None = None
        self._parser: Callable[[FakeParserFrame], None] | None = None
        self._turn_end: Callable[[], Any] | None = None

    def on_user_audio(self, handler: Callable[[Any], None]) -> None:
        """Register the audio handler."""
        self._audio = handler

    def on_stt(self, handler: Callable[[Any], None]) -> None:
        """Register the STT handler."""
        self._stt = handler

    def on_parser(self, handler: Callable[[Any], None]) -> None:
        """Register the parser handler."""
        self._parser = handler

    def on_turn_end(self, handler: Callable[[], Any]) -> None:
        """Register the turn-end handler."""
        self._turn_end = handler

    def attach(self, observer: PipecatObserver) -> FakeConversation:
        """Attach an observer and return self for chaining."""
        observer.attach(self)
        return self

    def run(self) -> None:
        """Emit every scripted turn through the registered handlers."""
        for turn in self.turns:
            if self._audio is not None:
                self._audio(turn.audio)
            if self._stt is not None and turn.stt is not None:
                self._stt(turn.stt)
            if self._parser is not None and turn.parser is not None:
                self._parser(turn.parser)
            if self._turn_end is not None:
                self._turn_end()
