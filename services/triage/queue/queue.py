"""Review-queue construction: turn/score seams, caps, interleaving and blind items.

No scoring package, database, network or audio decoding is imported here. Turn
events arrive through :class:`TurnSource` and P2-01 explanations arrive through
:class:`ScoreSource` (or :class:`ExplanationScoreSource`, which duck-types any
store exposing ``all()``). The read-only fixture sources exist only for
rehearsals and the CLI.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

from voiceloop_common import utcnow

from .clustering import ClusterItem, cluster_turns
from .sampling import (
    FAILURE_SLICE,
    RANDOM_SLICE,
    SliceTag,
    rng_for_seed,
    select_blind_items,
    select_random_slice,
)
from .settings import TriageQueueSettings

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIXTURES_PATH = "tests/fixtures/turns.jsonl"
SCHEMA_VERSION = "1.0.0"
_CARD_PATTERN = re.compile(r"(?:\d[ -]?){13,19}")


class QueueError(ValueError):
    """Raised when a turn, score or queue input is invalid."""


def _as_mapping(value: object) -> Mapping[str, Any]:
    """Return ``value`` as a mapping, or an empty mapping when it is not one."""
    return cast(Mapping[str, Any], value) if isinstance(value, Mapping) else {}


def _contains_card_number(text: str) -> bool:
    """Return whether ``text`` contains a 13–19 digit card-like sequence."""
    for match in _CARD_PATTERN.finditer(text):
        if 13 <= sum(character.isdigit() for character in match.group()) <= 19:
            return True
    return False


@dataclass(frozen=True)
class LowConfidenceSpan:
    """A word-level range to highlight for the human reviewer."""

    start_ms: int
    end_ms: int
    text: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable span matching the review-adapter contract."""
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.text,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ScoreSnapshot:
    """The only failure-score data the queue needs from P2-01."""

    event_id: str
    score: float
    signals: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ScoreSnapshot:
        """Build a snapshot from a score/explanation JSON record.

        Args:
            payload: A dict with ``event_id`` and ``score``; either ``signals``
                (P2-01 ``to_dict()`` shape) or ``signal_names`` is accepted.

        Returns:
            The immutable snapshot.

        Raises:
            QueueError: If identifiers or score are missing or malformed.

        """
        event_id = payload.get("event_id")
        score = payload.get("score")
        if not isinstance(event_id, str) or not event_id:
            raise QueueError("score record needs a non-empty event_id")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise QueueError(f"score record for {event_id} needs a numeric score")
        raw_signals = payload.get("signals", payload.get("signal_names", ()))
        names: list[str] = []
        if isinstance(raw_signals, (list, tuple)):
            for signal in raw_signals:
                if isinstance(signal, Mapping) and isinstance(signal.get("name"), str):
                    names.append(str(signal["name"]))
                elif isinstance(signal, str):
                    names.append(signal)
        return cls(event_id=event_id, score=float(score), signals=tuple(names))


@runtime_checkable
class ScoreSource(Protocol):
    """Injectable seam supplying a P2-01 failure score for an event ID."""

    def snapshot_for(self, event_id: str) -> ScoreSnapshot:
        """Return the score snapshot for ``event_id``."""
        ...


class InMemoryScoreSource:
    """Deterministic fake score source used by tests and offline rehearsals."""

    def __init__(self, scores: Mapping[str, float | ScoreSnapshot] | None = None) -> None:
        """Index ``score`` values by event ID.

        Args:
            scores: Event ID to score or :class:`ScoreSnapshot`.

        """
        self._scores: dict[str, ScoreSnapshot] = {}
        for event_id, value in (scores or {}).items():
            snapshot = (
                value
                if isinstance(value, ScoreSnapshot)
                else ScoreSnapshot(event_id=event_id, score=float(value))
            )
            self._scores[event_id] = snapshot

    def snapshot_for(self, event_id: str) -> ScoreSnapshot:
        """Return a stored snapshot or raise :class:`QueueError`."""
        try:
            return self._scores[event_id]
        except KeyError as exc:
            raise QueueError(f"no failure score for event {event_id}") from exc


class ExplanationScoreSource:
    """Adapt any P2-01-style explanation store to :class:`ScoreSource`.

    The adapter uses duck typing (``explanation.event_id``, ``.score`` and
    ``.signal_names``) so this package does not import P2-01 while that lane is
    in flight. ``InMemoryExplanationStore`` from ``services.triage.scoring``
    satisfies the seam unchanged.
    """

    def __init__(self, store: object) -> None:
        """Index all explanations returned by ``store.all()``.

        Args:
            store: An object exposing ``all() -> iterable of explanations``.

        Raises:
            QueueError: If the store seam or an explanation is malformed.

        """
        all_explanations = getattr(store, "all", None)
        if not callable(all_explanations):
            raise QueueError("explanation store must expose all()")
        self._scores: dict[str, ScoreSnapshot] = {}
        for explanation in all_explanations():
            event_id = getattr(explanation, "event_id", None)
            score = getattr(explanation, "score", None)
            if not isinstance(event_id, str) or not event_id:
                raise QueueError("explanation needs a non-empty event_id")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise QueueError(f"explanation for {event_id} needs a numeric score")
            raw_names = getattr(explanation, "signal_names", ())
            names = (
                tuple(str(name) for name in raw_names)
                if isinstance(raw_names, (tuple, list))
                else ()
            )
            self._scores[event_id] = ScoreSnapshot(event_id, float(score), names)

    def snapshot_for(self, event_id: str) -> ScoreSnapshot:
        """Return a stored snapshot or raise :class:`QueueError`."""
        try:
            return self._scores[event_id]
        except KeyError as exc:
            raise QueueError(f"no explanation for event {event_id}") from exc


class ScoreFileSource:
    """Load score snapshots from a P2-01 CLI report or a simple JSONL file."""

    def __init__(self, path: str | Path, *, root: Path | None = None) -> None:
        """Set the score file path.

        Args:
            path: Repo-relative or absolute JSON/JSONL path.
            root: Base directory for relative paths; defaults to repo root.

        """
        candidate = Path(path)
        self._path = candidate if candidate.is_absolute() else (root or REPO_ROOT) / candidate
        self._index: dict[str, ScoreSnapshot] = {}
        self._loaded = False

    def _records(self) -> tuple[ScoreSnapshot, ...]:
        """Parse JSON, an ``explanations`` report or JSON Lines into snapshots."""
        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            raise QueueError(f"cannot read scores: {self._path}") from exc
        try:
            decoded: Any = json.loads(text)
        except json.JSONDecodeError:
            decoded = [json.loads(line) for line in text.splitlines() if line.strip()]
        if isinstance(decoded, list):
            records: Sequence[Any] = decoded
        elif isinstance(decoded, Mapping):
            records = ()
            for key in ("explanations", "scores", "items"):
                if isinstance(decoded.get(key), list):
                    records = cast(Sequence[Any], decoded[key])
                    break
            if not records:
                records = [
                    {"event_id": event_id, "score": value}
                    for event_id, value in decoded.items()
                    if not isinstance(value, Mapping)
                ]
        else:
            raise QueueError(f"unsupported score file shape: {self._path}")
        snapshots: list[ScoreSnapshot] = []
        for record in records:
            if not isinstance(record, Mapping):
                raise QueueError(f"score record is not an object in {self._path}")
            snapshots.append(ScoreSnapshot.from_mapping(record))
        return tuple(snapshots)

    def snapshot_for(self, event_id: str) -> ScoreSnapshot:
        """Return the snapshot for ``event_id``; read and index on first use."""
        if not self._loaded:
            self._index = {snapshot.event_id: snapshot for snapshot in self._records()}
            self._loaded = True
        try:
            return self._index[event_id]
        except KeyError as exc:
            raise QueueError(f"no score for event {event_id} in {self._path}") from exc


@dataclass(frozen=True)
class TurnRecord:
    """An immutable turn-event view used by queue construction."""

    event_id: str
    call_id: str
    turn_index: int
    event: Mapping[str, Any]

    @classmethod
    def from_event(cls, event: Mapping[str, Any]) -> TurnRecord:
        """Validate the identifiers required for lineage and idempotency.

        Args:
            event: A turn-event dict conforming to ``turn_event`` 1.0.0.

        Returns:
            The immutable record.

        Raises:
            QueueError: If an identifier is missing or malformed.

        """
        event_id = event.get("event_id")
        call_id = event.get("call_id")
        turn_index = event.get("turn_index")
        if not isinstance(event_id, str) or not event_id:
            raise QueueError("turn event needs a non-empty event_id")
        if not isinstance(call_id, str) or not call_id:
            raise QueueError("turn event needs a non-empty call_id")
        if isinstance(turn_index, bool) or not isinstance(turn_index, int) or turn_index < 0:
            raise QueueError("turn event needs a non-negative integer turn_index")
        return cls(event_id=event_id, call_id=call_id, turn_index=turn_index, event=event)

    @property
    def hypothesis(self) -> str:
        """Return the STT hypothesis text, or an empty string when absent."""
        text = _as_mapping(self.event.get("stt")).get("text", "")
        return text if isinstance(text, str) else ""

    @property
    def audio_uri(self) -> str:
        """Return the original-audio URI, or an empty string when absent."""
        uri = _as_mapping(self.event.get("audio")).get("uri", "")
        return uri if isinstance(uri, str) else ""

    @property
    def training_consent(self) -> bool:
        """Return whether training consent was granted (review is still allowed)."""
        return _as_mapping(self.event.get("consent")).get("training") is True

    def low_confidence_spans(self, ceiling: float) -> tuple[LowConfidenceSpan, ...]:
        """Derive review highlights from word confidences below ``ceiling``.

        Args:
            ceiling: Confidence strictly below which a word is highlighted.

        Returns:
            Spans in word order; malformed word records are skipped.

        """
        words = _as_mapping(self.event.get("stt")).get("words", ())
        spans: list[LowConfidenceSpan] = []
        if not isinstance(words, (list, tuple)):
            return ()
        for word in words:
            if not isinstance(word, Mapping):
                continue
            confidence = word.get("conf")
            start = word.get("start_ms")
            end = word.get("end_ms")
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                continue
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            if float(confidence) >= ceiling:
                continue
            spans.append(
                LowConfidenceSpan(
                    start_ms=max(0, start),
                    end_ms=max(start, end),
                    text=str(word.get("w", "")),
                    confidence=float(confidence),
                )
            )
        return tuple(spans)


@runtime_checkable
class TurnSource(Protocol):
    """Injectable seam supplying eligible turn events."""

    def iter_turns(self) -> Iterator[TurnRecord]:
        """Iterate over turn records in a stable order."""
        ...


class InMemoryTurnSource:
    """Fake turn source backed by event dictionaries supplied by the caller."""

    def __init__(self, events: Sequence[TurnRecord | Mapping[str, Any]]) -> None:
        """Validate and index events or already-built records on construction.

        Args:
            events: Turn-event dictionaries or :class:`TurnRecord` views.

        """
        self._records = tuple(
            event if isinstance(event, TurnRecord) else TurnRecord.from_event(event)
            for event in events
        )

    def iter_turns(self) -> Iterator[TurnRecord]:
        """Iterate over the indexed records."""
        return iter(self._records)


class FixtureTurnSource:
    """Read-only source for the synthetic ``turns.jsonl`` fixture manifest.

    Labels locate event files and are checked for call/turn agreement; no
    ``failure_type``, ``expected_transcript`` or ``expected_slots`` value is
    used for sampling, scoring or queueing.
    """

    def __init__(
        self, turns_path: str | Path = DEFAULT_FIXTURES_PATH, *, root: Path | None = None
    ) -> None:
        """Set the manifest path.

        Args:
            turns_path: Repo-relative or absolute labels manifest path.
            root: Base directory for relative paths; defaults to repo root.

        """
        candidate = Path(turns_path)
        self._root = root or REPO_ROOT
        self._path = candidate if candidate.is_absolute() else self._root / candidate

    def iter_turns(self) -> Iterator[TurnRecord]:
        """Load and yield the fixture events referenced by the manifest.

        Yields:
            One :class:`TurnRecord` per manifest row.

        Raises:
            QueueError: If the manifest or an event is unreadable or inconsistent.

        """
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise QueueError(f"cannot read fixtures: {self._path}") from exc
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                label = json.loads(line)
            except json.JSONDecodeError as exc:
                raise QueueError(f"invalid fixture JSON at line {number}") from exc
            if not isinstance(label, Mapping):
                raise QueueError(f"fixture line {number} is not an object")
            event_path = label.get("event_path")
            if not isinstance(event_path, str) or not event_path:
                raise QueueError(f"fixture line {number} needs an event_path")
            candidate = Path(event_path)
            resolved = candidate if candidate.is_absolute() else self._root / candidate
            try:
                event = json.loads(resolved.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise QueueError(f"cannot read fixture event {resolved}") from exc
            if not isinstance(event, Mapping):
                raise QueueError(f"fixture event is not an object: {resolved}")
            record = TurnRecord.from_event(event)
            if label.get("call_id") not in (None, record.call_id):
                raise QueueError(f"call_id mismatch at fixture line {number}")
            if label.get("turn_index") not in (None, record.turn_index):
                raise QueueError(f"turn_index mismatch at fixture line {number}")
            yield record


@dataclass(frozen=True)
class QueueItem:
    """One review-ready item with lineage, slice tag and cluster provenance."""

    event_id: str
    call_id: str
    turn_index: int
    audio_uri: str
    hypothesis: str
    score: float
    slice: SliceTag
    cluster_id: str
    cluster_size: int
    no_prefill: bool
    training_consent: bool
    signals: tuple[str, ...]
    low_confidence_spans: tuple[LowConfidenceSpan, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the queue JSON view, including slice and cluster metadata."""
        return {
            "event_id": self.event_id,
            "call_id": self.call_id,
            "turn_index": self.turn_index,
            "audio_uri": self.audio_uri,
            "hypothesis": self.hypothesis,
            "score": self.score,
            "slice": self.slice,
            "cluster_id": self.cluster_id,
            "cluster_size": self.cluster_size,
            "no_prefill": self.no_prefill,
            "training_consent": self.training_consent,
            "signals": list(self.signals),
            "low_confidence_spans": [span.to_dict() for span in self.low_confidence_spans],
        }

    def to_review_item(self) -> dict[str, Any]:
        """Return a ``ReviewItemCreate`` payload for the P2-03 adapter contract.

        ``no_prefill`` is carried as ``blind``; the OpenAPI payload intentionally
        has no extra fields, so queue-only lineage stays in :meth:`to_dict`.
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "call_id": self.call_id,
            "turn_index": self.turn_index,
            "audio_uri": self.audio_uri,
            "hypothesis": self.hypothesis,
            "low_confidence_spans": [span.to_dict() for span in self.low_confidence_spans],
            "blind": self.no_prefill,
        }


@dataclass(frozen=True)
class QueueStats:
    """Auditable counts from one queue build."""

    eligible_turns: int
    excluded_turns: int
    failure_candidates: int
    clusters: int
    random_slice_size: int
    random_slice_reserved: int
    deferred_by_cap: int
    blind_items: int

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-serialisable stats view."""
        return {
            "eligible_turns": self.eligible_turns,
            "excluded_turns": self.excluded_turns,
            "failure_candidates": self.failure_candidates,
            "clusters": self.clusters,
            "random_slice_size": self.random_slice_size,
            "random_slice_reserved": self.random_slice_reserved,
            "deferred_by_cap": self.deferred_by_cap,
            "blind_items": self.blind_items,
        }


@dataclass(frozen=True)
class QueueReport:
    """A capped, ordered day of review items plus its audit counts."""

    day: str
    seed: int
    items: tuple[QueueItem, ...]
    stats: QueueStats

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable report."""
        return {
            "day": self.day,
            "seed": self.seed,
            "stats": self.stats.to_dict(),
            "items": [item.to_dict() for item in self.items],
        }


def _is_eligible(record: TurnRecord) -> bool:
    """Require recording consent and screen obvious card numbers.

    Training consent is deliberately not required for review: a human may
    listen under recording consent, while dataset use remains gated downstream
    (``context.md`` §2.8). This card-number check is defence in depth, not a
    replacement for the governance/PII filters.
    """
    consent = record.event.get("consent")
    if not isinstance(consent, Mapping) or consent.get("recording") is not True:
        return False
    parser_text = str(_as_mapping(record.event.get("parser")).get("input_text", ""))
    return not any(_contains_card_number(text) for text in (record.hypothesis, parser_text))


def _interleave(
    failures: Sequence[str], random_ids: Sequence[str], interleave_every: int
) -> list[str]:
    """Interleave random-slice items among score-ordered failure items.

    The configured ``interleave_every`` is an upper bound on the gap; for small
    queues the gap shrinks so all reserved random items appear before the tail.

    Args:
        failures: Representative event IDs, highest score first.
        random_ids: Random-slice event IDs in stable order.
        interleave_every: Maximum number of failure items between random items.

    Returns:
        A new ordered ID list.

    """
    if not random_ids:
        return list(failures)
    if not failures:
        return list(random_ids)
    interval = min(interleave_every, max(1, len(failures) // (len(random_ids) + 1)))
    ordered: list[str] = []
    next_random = 0
    for position, event_id in enumerate(failures, start=1):
        ordered.append(event_id)
        if next_random < len(random_ids) and position % interval == 0:
            ordered.append(random_ids[next_random])
            next_random += 1
    ordered.extend(random_ids[next_random:])
    return ordered


def build_queue(
    turns: TurnSource,
    scores: ScoreSource,
    settings: TriageQueueSettings | None = None,
    *,
    day: str | None = None,
) -> QueueReport:
    """Build one capped, deterministic review queue from injected seams.

    Random-slice turns are chosen from all eligible turns *before* the failure
    filter, never clustered and never dropped by the cap except when the cap is
    smaller than the slice itself. Failure turns are clustered; one highest-
    score representative per cluster enters the queue.

    Args:
        turns: Turn-event source.
        scores: Failure-score source (P2-01 via an adapter, or a fake).
        settings: Typed queue settings; defaults to environment-backed values.
        day: ISO date label for the queue; defaults to the current UTC date.

    Returns:
        The ordered :class:`QueueReport`.

    Raises:
        QueueError: If a turn or score is invalid.

    """
    config = settings if settings is not None else TriageQueueSettings()
    records = list(turns.iter_turns())
    eligible = [record for record in records if _is_eligible(record)]
    excluded = len(records) - len(eligible)

    random_ids = select_random_slice(
        [record.event_id for record in eligible],
        config,
        rng=rng_for_seed(config.random_seed, "slice"),
    )
    random_set = frozenset(random_ids)
    snapshots = {record.event_id: scores.snapshot_for(record.event_id) for record in eligible}

    failure_records = [
        record
        for record in eligible
        if record.event_id not in random_set
        and snapshots[record.event_id].score > config.failure_threshold
    ]
    clusters = cluster_turns(
        [
            ClusterItem(
                key=record.event_id,
                text=record.hypothesis,
                score=snapshots[record.event_id].score,
            )
            for record in failure_records
        ],
        config.cluster_similarity_threshold,
    )
    cluster_by_representative = {cluster.representative: cluster for cluster in clusters}
    representatives = sorted(
        cluster_by_representative,
        key=lambda event_id: (-snapshots[event_id].score, event_id),
    )

    reserved_slice = random_ids[: min(len(random_ids), config.daily_cap)]
    failure_slots = max(0, config.daily_cap - len(reserved_slice))
    selected_failures = representatives[:failure_slots]
    ordered_ids = _interleave(selected_failures, reserved_slice, config.interleave_every)
    deferred = len(random_ids) - len(reserved_slice) + len(representatives) - len(selected_failures)
    blind_ids = select_blind_items(
        ordered_ids,
        config.blind_share,
        rng=rng_for_seed(config.random_seed, "blind"),
    )

    by_id = {record.event_id: record for record in eligible}
    items: list[QueueItem] = []
    for event_id in ordered_ids:
        record = by_id[event_id]
        snapshot = snapshots[event_id]
        is_random = event_id in random_set
        cluster = None if is_random else cluster_by_representative.get(event_id)
        items.append(
            QueueItem(
                event_id=event_id,
                call_id=record.call_id,
                turn_index=record.turn_index,
                audio_uri=record.audio_uri,
                hypothesis=record.hypothesis,
                score=snapshot.score,
                slice=RANDOM_SLICE if is_random else FAILURE_SLICE,
                cluster_id=cluster.cluster_id if cluster else f"random-{event_id}",
                cluster_size=cluster.size if cluster else 1,
                no_prefill=event_id in blind_ids,
                training_consent=record.training_consent,
                signals=snapshot.signals,
                low_confidence_spans=record.low_confidence_spans(config.low_confidence_ceiling),
            )
        )

    return QueueReport(
        day=day or utcnow().date().isoformat(),
        seed=config.random_seed,
        items=tuple(items),
        stats=QueueStats(
            eligible_turns=len(eligible),
            excluded_turns=excluded,
            failure_candidates=len(failure_records),
            clusters=len(clusters),
            random_slice_size=len(random_ids),
            random_slice_reserved=len(reserved_slice),
            deferred_by_cap=deferred,
            blind_items=len(blind_ids),
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the queue CLI over read-only fixtures and an external score report.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code (0 on success).

    """
    parser = argparse.ArgumentParser(description="Build a capped triage review queue.")
    parser.add_argument(
        "--fixtures",
        default=DEFAULT_FIXTURES_PATH,
        help="Read-only labels manifest (JSON Lines).",
    )
    parser.add_argument(
        "--scores",
        required=True,
        help="P2-01 score report (JSON) or an event_id/score JSONL file.",
    )
    parser.add_argument(
        "--output", default=None, help="Optional JSON output path; stdout by default."
    )
    parser.add_argument("--day", default=None, help="Optional ISO date label for the queue.")
    args = parser.parse_args(argv)

    report = build_queue(
        FixtureTurnSource(args.fixtures),
        ScoreFileSource(args.scores),
        day=args.day,
    )
    rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
