"""Deterministic per-word confidences wired to the P2-01 triage signal.

SYNTHETIC / FIXTURE STAND-IN - NOT A REAL NEMO FINE-TUNE. Confidence values are
derived from a SHA-256 of the fake transcript, seed and model id; they are
**not** NeMo posterior probabilities. The generated ``stt.words`` entries use
the exact ``turn_event`` 1.0.0 shape (``w``/``start_ms``/``end_ms``/``conf``),
and the default sink calls P2-01's ``ScoringEngine`` through a narrow Protocol.
P2-01 itself is consumed unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from services.trainers.core import TrainerSettings
from services.triage.scoring import InMemoryExplanationStore, ScoringEngine
from voiceloop_common import ValidationError, validate

from .manifest import NemoManifestRow, sha256_text
from .settings import PROVENANCE_LABEL, NemoSettings


@dataclass(frozen=True, slots=True)
class WordConfidence:
    """One schema-shaped word-confidence entry for ``stt.words``."""

    w: str
    start_ms: int
    end_ms: int
    conf: float

    def to_dict(self) -> dict[str, Any]:
        """Return exactly the four properties allowed by ``turn_event`` 1.0.0."""
        return {
            "w": self.w,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "conf": self.conf,
        }


@dataclass(frozen=True, slots=True)
class WordConfidenceOutput:
    """Fake per-word confidences plus an explicit stand-in provenance label."""

    words: tuple[WordConfidence, ...]
    provenance_label: str = PROVENANCE_LABEL

    def as_schema_words(self) -> list[dict[str, Any]]:
        """Return ``stt.words`` payloads conforming to ``turn_event`` 1.0.0."""
        return [word.to_dict() for word in self.words]

    def to_dict(self) -> dict[str, Any]:
        """Return a labelled stand-in payload (not a schema field of turn events)."""
        return {"provenance_label": self.provenance_label, "words": self.as_schema_words()}


def fake_word_confidences(
    text: str,
    *,
    duration_ms: int,
    seed: int,
    model: str,
    transcript_hash: str | None = None,
) -> WordConfidenceOutput:
    """Derive deterministic fake timings and confidences from a transcript.

    Args:
        text: Fake (fixture) transcript from the P2-05 NeMo row.
        duration_ms: Overall audio duration; word spans partition it evenly.
        seed: Core trainer seed, injected by settings.
        model: Base model id from settings, never hard-coded here.
        transcript_hash: Optional stable row fingerprint; derived from ``text``
            when omitted.

    Returns:
        Labelled word confidences with monotonically non-decreasing starts.

    Raises:
        ValidationError: If the transcript has no words or the duration is invalid.

    """
    tokens = text.split()
    if not tokens:
        raise ValidationError("fake confidences require a non-empty transcript")
    if isinstance(duration_ms, bool) or duration_ms <= 0:
        raise ValidationError("fake confidences require a positive duration_ms")
    source_hash = transcript_hash or sha256_text(text)
    prefix = "|".join((str(seed), model, source_hash))
    count = len(tokens)
    boundaries = [(index * duration_ms) // count for index in range(count + 1)]
    words: list[WordConfidence] = []
    for index, token in enumerate(tokens):
        digest = sha256_text(f"{prefix}|{index}|{token}")
        conf = round(int(digest[:8], 16) / 0xFFFFFFFF, 6)
        start = boundaries[index]
        end = max(start + 1, boundaries[index + 1])
        words.append(WordConfidence(w=token, start_ms=start, end_ms=end, conf=conf))
    return WordConfidenceOutput(words=tuple(words))


def build_turn_event(
    row: NemoManifestRow,
    *,
    words: WordConfidenceOutput,
    event_id: str,
    call_id: str,
    turn_index: int,
    model_id: str,
    model_version: str = "fake-1.0",
) -> dict[str, Any]:
    """Build a schema-valid turn event carrying fake per-word confidences.

    Args:
        row: Read-only NeMo manifest row supplying the fake transcript, audio
            reference and duration.
        words: Fake confidences generated from that transcript.
        event_id: Caller-supplied UUID (never fabricated from real data here).
        call_id: Caller-supplied synthetic call identifier.
        turn_index: Zero-based turn index within that call.
        model_id: STT model id from settings.
        model_version: Fake model version label.

    Returns:
        A dict validated against ``schemas/turn_event.schema.json`` 1.0.0.

    Raises:
        ValidationError: If identifiers are malformed or the result violates
            the event contract. ``additionalProperties: false`` means the
            stand-in label is carried by :class:`ConfidenceEmission`, not the event.

    """
    if not isinstance(call_id, str) or not call_id:
        raise ValidationError("build_turn_event requires a non-empty call_id")
    if isinstance(turn_index, bool) or not isinstance(turn_index, int) or turn_index < 0:
        raise ValidationError("build_turn_event requires a non-negative turn_index")
    event: dict[str, Any] = {
        "event_id": event_id,
        "call_id": call_id,
        "turn_index": turn_index,
        "audio": {
            "uri": row.audio_filepath,
            "duration_ms": max(1, int(round(row.duration * 1000))),
        },
        "stt": {
            "model_id": model_id,
            "model_version": model_version,
            "text": row.text,
            "words": words.as_schema_words(),
        },
    }
    validate(event, "turn_event")
    return event


@dataclass(frozen=True, slots=True)
class ScoreRecord:
    """Narrow view of a P2-01 score explanation returned by any signal sink."""

    event_id: str
    score: float
    signal_names: tuple[str, ...]
    provenance_label: str = PROVENANCE_LABEL

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable, labelled score record."""
        return {
            "event_id": self.event_id,
            "score": self.score,
            "signal_names": list(self.signal_names),
            "provenance_label": self.provenance_label,
        }


class TriageSignalSink(Protocol):
    """Injectable consumer of a turn event carrying word confidences."""

    def score(self, event: Mapping[str, Any]) -> ScoreRecord:
        """Score one event and return its fired signal names."""
        ...


class P2TriageSignalSink:
    """Default sink: P2-01's ``ScoringEngine``, used through the Protocol unchanged."""

    def __init__(self, engine: ScoringEngine | None = None) -> None:
        """Create or accept a P2-01 scoring engine with an in-memory store.

        Args:
            engine: Optional preconfigured engine; a default in-memory engine
                is created when omitted.

        """
        self._engine = engine or ScoringEngine(store=InMemoryExplanationStore())

    @property
    def engine(self) -> ScoringEngine:
        """Return the wrapped P2-01 engine for inspection in tests."""
        return self._engine

    def score(self, event: Mapping[str, Any]) -> ScoreRecord:
        """Score a schema-valid event with P2-01 and project its explanation."""
        explanation = self._engine.score(event, prior_turns=())
        return ScoreRecord(
            event_id=explanation.event_id,
            score=explanation.score,
            signal_names=tuple(signal.name for signal in explanation.signals),
        )


class InMemorySignalSink:
    """Deterministic fake sink for tests that should not depend on P2-01."""

    def __init__(self) -> None:
        """Create an empty event log."""
        self._events: list[dict[str, Any]] = []

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        """Return a snapshot of events received so far."""
        return tuple(self._events)

    def score(self, event: Mapping[str, Any]) -> ScoreRecord:
        """Record the event and derive a simple fake confidence score."""
        snapshot = dict(event)
        self._events.append(snapshot)
        words = snapshot.get("stt", {})
        word_list = words.get("words", []) if isinstance(words, Mapping) else []
        confidences = [
            float(word["conf"])
            for word in word_list
            if isinstance(word, Mapping) and isinstance(word.get("conf"), (int, float))
        ]
        minimum = min(confidences) if confidences else 0.0
        return ScoreRecord(
            event_id=str(snapshot.get("event_id", "")),
            score=round(1.0 - minimum, 6),
            signal_names=("low_word_confidence",) if minimum < 0.6 else (),
        )


@dataclass(frozen=True, slots=True)
class ConfidenceEmission:
    """Labelled word confidences, schema-shaped event and triage score."""

    words: WordConfidenceOutput
    event: dict[str, Any]
    score: ScoreRecord
    provenance_label: str = PROVENANCE_LABEL

    def to_dict(self) -> dict[str, Any]:
        """Return a labelled JSON-serialisable record of the whole emission."""
        return {
            "provenance_label": self.provenance_label,
            "words": self.words.to_dict(),
            "event": self.event,
            "score": self.score.to_dict(),
        }


class ConfidenceEmitter:
    """Generate fake words, build the event and hand it to a triage sink."""

    def __init__(
        self,
        *,
        settings: NemoSettings,
        core_settings: TrainerSettings,
        sink: TriageSignalSink | None = None,
    ) -> None:
        """Wire settings and an injectable score sink.

        Args:
            settings: NeMo settings supplying confidence model id.
            core_settings: Core settings supplying seed/base model.
            sink: Triage sink; defaults to the unmodified P2-01 engine.

        """
        self._settings = settings
        self._core_settings = core_settings
        self._sink: TriageSignalSink = sink or P2TriageSignalSink()

    @property
    def sink(self) -> TriageSignalSink:
        """Return the triage sink used by this emitter."""
        return self._sink

    def emit(
        self,
        row: NemoManifestRow,
        *,
        event_id: str,
        call_id: str,
        turn_index: int,
    ) -> ConfidenceEmission:
        """Produce one labelled emission for a NeMo manifest row.

        Args:
            row: Read-only synthetic manifest row.
            event_id: Caller-supplied UUID for the generated event.
            call_id: Caller-supplied synthetic call ID.
            turn_index: Zero-based turn index.

        Returns:
            Word confidences, a schema-valid event and the sink's score record.

        """
        model = self._core_settings.model or self._settings.confidence_model_id
        words = fake_word_confidences(
            row.text,
            duration_ms=max(1, int(round(row.duration * 1000))),
            seed=self._core_settings.seed,
            model=model,
            transcript_hash=row.fingerprint(),
        )
        event = build_turn_event(
            row,
            words=words,
            event_id=event_id,
            call_id=call_id,
            turn_index=turn_index,
            model_id=self._settings.confidence_model_id,
        )
        return ConfidenceEmission(words=words, event=event, score=self._sink.score(event))


def score_event_with_triage(
    event: Mapping[str, Any], *, sink: TriageSignalSink | None = None
) -> ScoreRecord:
    """Score an existing confidence-bearing event through any injected sink."""
    return (sink or P2TriageSignalSink()).score(event)


__all__: Sequence[str] = (
    "ConfidenceEmission",
    "ConfidenceEmitter",
    "InMemorySignalSink",
    "P2TriageSignalSink",
    "ScoreRecord",
    "TriageSignalSink",
    "WordConfidence",
    "WordConfidenceOutput",
    "build_turn_event",
    "fake_word_confidences",
    "score_event_with_triage",
)
