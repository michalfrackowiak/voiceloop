"""Baseline scoring, the single teacher signal, and injectable queue selectors.

P2-01 is consumed read-only as the baseline scorer with its own teacher seam
explicitly disabled, so this package never changes P2-01 thresholds or adds a
second teacher contribution. The only addition is one extra field on each
:class:`QueueCandidate`; an injected :class:`QueueSelector` decides whether and
how to use it. No alias, gate or promotion state is touched.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from services.triage.scoring import TriageScoringSettings, load_prior_turns, score_turn
from voiceloop_common import ValidationError

from .batch import REPO_ROOT, BatchResult, TurnRecord
from .disagreement import (
    Disagreement,
    ProductionTranscriptSource,
    score_disagreement,
)
from .settings import SIGNAL_NAME, TeacherSettings

_FAKE_BASELINE_SOURCE = "fake://baseline/p2_01_disabled_teacher"


@dataclass(frozen=True)
class BaselineScore:
    """The unchanged P2-01-style score for one turn."""

    event_id: str
    score: float
    signals: tuple[str, ...]
    source: str = _FAKE_BASELINE_SOURCE

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable baseline view."""
        return {
            "event_id": self.event_id,
            "score": self.score,
            "signals": list(self.signals),
            "source": self.source,
        }


@runtime_checkable
class BaselineScorer(Protocol):
    """Injectable source of the baseline triage score."""

    def score(self, record: TurnRecord) -> BaselineScore:
        """Score one turn without using teacher or reviewed-label data."""
        ...


class P2_01BaselineScorer:
    """Read-only adapter over P2-01; its optional teacher seam stays disabled."""

    def __init__(self, *, root: Path = REPO_ROOT) -> None:
        """Set the base directory used to resolve dialogue-context references.

        Args:
            root: Repository root or equivalent base path for prior-turn files.

        """
        self._root = root

    def score(self, record: TurnRecord) -> BaselineScore:
        """Score a turn with P2-01 defaults and no teacher contribution."""
        prior_turns = load_prior_turns(record.event, root=self._root)
        explanation = score_turn(
            record.event,
            settings=TriageScoringSettings(teacher_enabled=False),
            prior_turns=prior_turns,
            persist=False,
        )
        return BaselineScore(
            event_id=record.event_id,
            score=explanation.score,
            signals=tuple(signal.name for signal in explanation.signals),
        )


class InMemoryBaselineScorer:
    """Deterministic in-memory fake baseline keyed by event ID."""

    def __init__(self, scores: Mapping[str, float | BaselineScore]) -> None:
        """Index precomputed baseline scores.

        Args:
            scores: Event ID to scalar score or ready-made :class:`BaselineScore`.

        """
        self._scores: dict[str, BaselineScore] = {}
        for event_id, value in scores.items():
            self._scores[event_id] = (
                value
                if isinstance(value, BaselineScore)
                else BaselineScore(event_id=event_id, score=float(value), signals=(), source="fake")
            )

    def score(self, record: TurnRecord) -> BaselineScore:
        """Return the indexed baseline or raise :class:`ValidationError`."""
        value = self._scores.get(record.event_id)
        if value is None:
            raise ValidationError(f"no baseline score for {record.event_id}")
        return value


@dataclass(frozen=True)
class TeacherSignal:
    """The one additional signal offered to an injected queue selector."""

    event_id: str
    name: str
    score: float
    weight: float
    weighted: float
    reason: str
    production_text_hash: str
    teacher_text_hash: str

    @classmethod
    def from_disagreement(cls, disagreement: Disagreement, weight: float) -> TeacherSignal:
        """Wrap a disagreement and apply the configured, untuned weight.

        Args:
            disagreement: Deterministic token-level difference.
            weight: Multiplier applied to the raw disagreement score.

        """
        return cls(
            event_id=disagreement.event_id,
            name=disagreement.name,
            score=disagreement.score,
            weight=weight,
            weighted=round(disagreement.score * weight, 6),
            reason=disagreement.reason,
            production_text_hash=disagreement.production_text_hash,
            teacher_text_hash=disagreement.teacher_text_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable signal view."""
        return {
            "event_id": self.event_id,
            "name": self.name,
            "score": self.score,
            "weight": self.weight,
            "weighted": self.weighted,
            "reason": self.reason,
            "production_text_hash": self.production_text_hash,
            "teacher_text_hash": self.teacher_text_hash,
        }


@dataclass(frozen=True)
class QueueCandidate:
    """One turn with its baseline score and exactly one optional extra signal."""

    event_id: str
    call_id: str
    turn_index: int
    baseline: BaselineScore
    teacher_signal: TeacherSignal

    @property
    def weighted_score(self) -> float:
        """Return baseline score plus the single weighted teacher signal."""
        return self.baseline.score + self.teacher_signal.weighted

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable candidate view."""
        return {
            "event_id": self.event_id,
            "call_id": self.call_id,
            "turn_index": self.turn_index,
            "baseline": self.baseline.to_dict(),
            "teacher_signal": self.teacher_signal.to_dict(),
            "weighted_score": self.weighted_score,
        }


@runtime_checkable
class QueueSelector(Protocol):
    """Injectable policy choosing which candidate event IDs enter a queue."""

    @property
    def name(self) -> str:
        """Stable selector name recorded in the precision report."""
        ...

    def select(self, candidates: Sequence[QueueCandidate], limit: int) -> tuple[str, ...]:
        """Return up to ``limit`` event IDs in queue order."""
        ...


class BaselineQueueSelector:
    """Fake queue policy that ignores the teacher signal entirely."""

    @property
    def name(self) -> str:
        """Return the baseline selector name."""
        return "baseline_no_teacher"

    def select(self, candidates: Sequence[QueueCandidate], limit: int) -> tuple[str, ...]:
        """Rank by baseline score, breaking ties on event ID."""
        ordered = sorted(candidates, key=lambda item: (-item.baseline.score, item.event_id))
        return tuple(item.event_id for item in ordered[:limit])


class TeacherQueueSelector:
    """Fake queue policy consuming the one extra teacher signal."""

    @property
    def name(self) -> str:
        """Return the teacher-enhanced selector name."""
        return "teacher_enhanced"

    def select(self, candidates: Sequence[QueueCandidate], limit: int) -> tuple[str, ...]:
        """Rank by baseline plus weighted disagreement, breaking ties on event ID."""
        ordered = sorted(candidates, key=lambda item: (-item.weighted_score, item.event_id))
        return tuple(item.event_id for item in ordered[:limit])


class TeacherOnlySelector:
    """Fake selector that consumes the teacher signal alone (unit-test seam)."""

    @property
    def name(self) -> str:
        """Return the teacher-only selector name."""
        return "teacher_only"

    def select(self, candidates: Sequence[QueueCandidate], limit: int) -> tuple[str, ...]:
        """Rank by teacher disagreement alone, breaking ties on event ID."""
        ordered = sorted(candidates, key=lambda item: (-item.teacher_signal.score, item.event_id))
        return tuple(item.event_id for item in ordered[:limit])


def build_candidates(
    turns: Sequence[TurnRecord],
    batch: BatchResult,
    production: ProductionTranscriptSource,
    baseline: BaselineScorer,
    settings: TeacherSettings | None = None,
) -> tuple[QueueCandidate, ...]:
    """Join baseline scores and teacher disagreements into queue candidates.

    Args:
        turns: The same read-only turns used by the batch runner, in order.
        batch: Deterministic batch teacher output.
        production: Production transcript seam.
        baseline: Baseline score seam (P2-01 adapter by default).
        settings: Typed settings; defaults to environment-backed values.

    Returns:
        One candidate per turn, in input order.

    Raises:
        ValidationError: If batch records are missing, duplicated or misaligned.

    """
    config = settings or TeacherSettings()
    by_event = {record.event_id: record for record in batch.records}
    if len(by_event) != len(batch.records):
        raise ValidationError("teacher batch contains duplicate event IDs")
    if set(by_event) != {record.event_id for record in turns}:
        raise ValidationError("teacher batch and turn source are not 1:1")
    candidates: list[QueueCandidate] = []
    for turn in turns:
        record = by_event[turn.event_id]
        production_transcript = production.transcript_for(turn)
        if production_transcript.content_hash != record.production_text_hash:
            raise ValidationError(f"production transcript hash mismatch for {turn.event_id}")
        disagreement = score_disagreement(
            event_id=turn.event_id,
            production_text=production_transcript.text,
            teacher_text=record.teacher_text,
            production_hash=record.production_text_hash,
            teacher_hash=record.teacher_text_hash,
            name=SIGNAL_NAME,
        )
        candidates.append(
            QueueCandidate(
                event_id=turn.event_id,
                call_id=turn.call_id,
                turn_index=turn.turn_index,
                baseline=baseline.score(turn),
                teacher_signal=TeacherSignal.from_disagreement(disagreement, config.signal_weight),
            )
        )
    return tuple(candidates)
