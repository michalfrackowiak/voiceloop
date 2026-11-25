"""Injectable seams and read-only, fixture-backed fakes for review measurement.

> **SYNTHETIC/FIXTURE STAND-IN ONLY.** No human reviewed anything here. The
> default source replays committed synthetic fixture labels through the P2-03
> fake review tool with an injected clock. Every timestamp comes from that clock
> plus the committed scripted duration; no wall clock, network, review tool or
> audio decoding is used.

The protocols in this module are the substitution points for later lanes:
``ReviewResultSource`` supplies P2-03-style stored results, ``Clock`` supplies
timestamps, ``CandidatePairSource`` supplies explicit blind/prefilled pairs, and
``ReportSink`` receives the finished report.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from services.review_adapter import (
    InMemoryReviewStore,
    InMemoryReviewTool,
    ReviewCoordinator,
    ScriptedReview,
    SequenceQueueSource,
    SimpleQueueItem,
)
from services.review_adapter.interface import ROUTING_LABELS
from voiceloop_common import ValidationError

from .settings import ReviewMetricsSettings

REPO_ROOT = Path(__file__).resolve().parents[3]
"""Repository root; all default inputs are repo-relative and read-only."""

BLIND_REVIEWER_ID = "reviewer-fixture-blind"
"""Pseudonymous scripted reviewer for the no-prefill variant."""

PREFILLED_REVIEWER_ID = "reviewer-fixture-prefilled"
"""Pseudonymous scripted reviewer for the parser-prefilled variant."""

FAKE_REVIEW_WARNING = "fake_reviewer_stand_in"
"""Same warning string the committed P2-03 fixture helper attaches to fake reviews."""


def resolve_path(path: str | Path) -> Path:
    """Resolve a repo-relative path against the repository root.

    Args:
        path: Absolute path or path relative to the repository root.

    Returns:
        The resolved path.

    """
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of one file, read only."""
    return hashlib.sha256(resolve_path(path).read_bytes()).hexdigest()


def sha256_directory(path: str | Path) -> dict[str, str]:
    """Return SHA-256 digests of every file in a directory, keyed by name."""
    base = resolve_path(path)
    return {
        entry.name: hashlib.sha256(entry.read_bytes()).hexdigest()
        for entry in sorted(base.iterdir())
        if entry.is_file()
    }


def parse_utc(value: str | datetime) -> datetime:
    """Parse an RFC 3339 string or datetime into a timezone-aware UTC datetime.

    Args:
        value: Timestamp supplied by settings or an injected fake.

    Returns:
        A timezone-aware UTC datetime.

    Raises:
        ValidationError: If ``value`` is not a valid timestamp.

    """
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValidationError(f"invalid fake timestamp: {value!r}") from exc
    else:
        raise ValidationError("fake timestamp must be an RFC 3339 string or datetime")
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def rfc3339(value: datetime) -> str:
    """Return an RFC 3339 UTC string for a fake/injected timestamp."""
    return parse_utc(value).isoformat().replace("+00:00", "Z")


class ReviewResultLike(Protocol):
    """Read-only view of the P2-03 ``StoredReviewResult`` fields metrics consume."""

    @property
    def event_id(self) -> str:
        """Return the pseudonymous source event ID."""
        ...

    @property
    def call_id(self) -> str:
        """Return the pseudonymous source call ID."""
        ...

    @property
    def turn_index(self) -> int:
        """Return the turn index within the call."""
        ...

    @property
    def item_id(self) -> str:
        """Return the adapter-local item ID."""
        ...

    @property
    def hypothesis(self) -> str:
        """Return the STT hypothesis corrected during stage 1."""
        ...

    @property
    def corrected_transcript(self) -> str:
        """Return the scripted corrected transcript from stage 1."""
        ...

    @property
    def slots(self) -> Mapping[str, Any]:
        """Return the scripted stage 2 slot object."""
        ...

    @property
    def routing(self) -> str:
        """Return the routing label captured by stage 2."""
        ...

    @property
    def reviewer_id(self) -> str:
        """Return the pseudonymous scripted reviewer ID."""
        ...

    @property
    def started_at(self) -> datetime:
        """Return the injected fake review start timestamp."""
        ...

    @property
    def completed_at(self) -> datetime:
        """Return the injected fake review completion timestamp."""
        ...

    @property
    def blind(self) -> bool:
        """Return whether this variant skipped parser prefill."""
        ...

    @property
    def tool_name(self) -> str:
        """Return the fake tool identifier recorded by P2-03."""
        ...

    @property
    def training_consent(self) -> bool:
        """Return the fixture training-consent flag carried by P2-03."""
        ...

    @property
    def warnings(self) -> tuple[str, ...]:
        """Return the fake-tool warning strings; no validator text is exposed."""
        ...


class ReviewResultSource(Protocol):
    """Injectable source of P2-03-style fake review results."""

    def iter_results(self) -> Iterator[ReviewResultLike]:
        """Yield completed fake results in a stable order."""
        ...


class Clock(Protocol):
    """Injectable timestamp factory; never the system clock."""

    def __call__(self) -> datetime:
        """Return the next fake UTC timestamp."""
        ...


@dataclass(frozen=True)
class CandidateAnswer:
    """One scripted synthetic answer for a fixture or golden-set candidate.

    The text is kept in memory for equality checks only. Report writers must
    emit agreement counts and rates, never these strings.
    """

    reviewer_id: str
    corrected_transcript: str
    slots: Mapping[str, Any]


@dataclass(frozen=True)
class CandidatePair:
    """An explicit blind/prefilled pair for one committed fixture item."""

    event_id: str
    call_id: str
    turn_index: int
    blind: CandidateAnswer
    prefilled: CandidateAnswer


class CandidatePairSource(Protocol):
    """Injectable source of explicit paired blind vs prefilled items."""

    def iter_pairs(self) -> Iterator[CandidatePair]:
        """Yield explicitly paired candidates in a stable order."""
        ...


class ReportSink(Protocol):
    """Injectable destination for the finished metrics report."""

    def write(self, report: Mapping[str, Any]) -> str:
        """Persist ``report`` and return a destination label."""
        ...


class FixedClock:
    """Fake clock returning one injected timestamp on every call."""

    def __init__(self, value: str | datetime) -> None:
        """Freeze ``value`` as the only timestamp this clock can return.

        Args:
            value: RFC 3339 string or datetime to return.

        """
        self._value = parse_utc(value)

    def __call__(self) -> datetime:
        """Return the frozen UTC timestamp."""
        return self._value


class SequenceClock:
    """Fake clock advancing by a fixed step, with no wall-clock access."""

    def __init__(self, start: str | datetime, step_seconds: float) -> None:
        """Create a deterministic sequence of fake timestamps.

        Args:
            start: First UTC timestamp returned.
            step_seconds: Seconds added on each subsequent call.

        """
        self._start = parse_utc(start)
        self._step = timedelta(seconds=step_seconds)
        self._index = 0

    def __call__(self) -> datetime:
        """Return the next timestamp in the deterministic sequence."""
        value = self._start + self._index * self._step
        self._index += 1
        return value


class SequenceReviewResultSource:
    """Freeze any iterable of fake results into a stable result source."""

    def __init__(self, results: Iterable[ReviewResultLike]) -> None:
        """Freeze results in their supplied order.

        Args:
            results: Completed fake review results.

        """
        self._results = tuple(results)

    def iter_results(self) -> Iterator[ReviewResultLike]:
        """Iterate over the frozen results."""
        return iter(self._results)


class StoreReviewResultSource:
    """Adapt any P2-03-style store exposing ``all()`` into a result source."""

    def __init__(self, store: object) -> None:
        """Bind a result store without importing a concrete backend.

        Args:
            store: Object with a callable ``all()`` returning fake results.

        Raises:
            ValidationError: If ``store`` has no callable ``all()``.

        """
        getter = getattr(store, "all", None)
        if not callable(getter):
            raise ValidationError("review result store must expose all()")
        self._all: Callable[[], Any] = getter

    def iter_results(self) -> Iterator[ReviewResultLike]:
        """Yield every result in store order."""
        return iter(tuple(self._all()))


class ResultPairSource:
    """Build explicit blind/prefilled pairs from fake results for one event.

    Events without exactly one result of each variant are not paired; the
    anchoring section counts them as unpaired rather than inventing a partner.
    """

    def __init__(self, source: ReviewResultSource) -> None:
        """Bind a result source.

        Args:
            source: Source of committed fake review results.

        """
        self._source = source

    def iter_pairs(self) -> Iterator[CandidatePair]:
        """Yield one pair per event with exactly one blind and one prefilled result."""
        grouped: dict[str, dict[bool, list[ReviewResultLike]]] = {}
        for result in self._source.iter_results():
            variants = grouped.setdefault(result.event_id, {True: [], False: []})
            variants[bool(result.blind)].append(result)
        for event_id, variants in grouped.items():
            blind, prefilled = variants[True], variants[False]
            if len(blind) != 1 or len(prefilled) != 1:
                continue
            first, second = blind[0], prefilled[0]
            yield CandidatePair(
                event_id=event_id,
                call_id=first.call_id,
                turn_index=first.turn_index,
                blind=CandidateAnswer(
                    reviewer_id=first.reviewer_id,
                    corrected_transcript=first.corrected_transcript,
                    slots=first.slots,
                ),
                prefilled=CandidateAnswer(
                    reviewer_id=second.reviewer_id,
                    corrected_transcript=second.corrected_transcript,
                    slots=second.slots,
                ),
            )


class JsonFileReportSink:
    """Write one sorted UTF-8 JSON report; the only filesystem write in this package."""

    def __init__(self, path: str | Path) -> None:
        """Bind the output path.

        Args:
            path: Destination path, relative to the repository root unless absolute.

        """
        self._path = resolve_path(path)

    def write(self, report: Mapping[str, Any]) -> str:
        """Write ``report`` as JSON and return the destination path."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return str(self._path)


class InMemoryReportSink:
    """Report sink fakes for tests; writes no file and no network call."""

    def __init__(self) -> None:
        """Create an empty sink."""
        self.report: dict[str, Any] | None = None

    def write(self, report: Mapping[str, Any]) -> str:
        """Keep a copy of ``report`` in memory and return a synthetic label."""
        self.report = dict(report)
        return "in-memory://review-metrics"


def _routing_label(failure_type: object) -> str:
    """Map a fixture failure type to the P2-03 routing label without new labels.

    ``clean`` maps to ``both_fine`` exactly as the committed P2-03 fixture
    helper does; no routing value is inferred from a score or fabricated.
    """
    label = "both_fine" if failure_type == "clean" else str(failure_type)
    if label not in ROUTING_LABELS:
        raise ValidationError(f"unsupported fixture failure_type: {failure_type!r}")
    return label


class FixtureReviewResultSource:
    """Run committed fixture labels through the P2-03 fake tool twice per item.

    Each synthetic turn yields a blind and a prefilled fake review with distinct
    pseudonymous reviewer IDs. Corrected transcripts, slots and routing labels
    come only from ``tests/fixtures/turns.jsonl`` and its committed turn events.
    Timings come only from the injected clock plus the committed P2-03 scripted
    duration. No human review, real parser, audio decode or network call occurs.
    """

    def __init__(self, settings: ReviewMetricsSettings, clock: Clock) -> None:
        """Precompute all fake results once so repeated reads are deterministic.

        Args:
            settings: Typed fixture-only settings.
            clock: Injected fake clock supplying every review start timestamp.

        Raises:
            ValidationError: If a fixture row or event is missing required data.

        """
        self._settings = settings
        self._clock = clock
        self._results = tuple(self._generate())

    def iter_results(self) -> Iterator[ReviewResultLike]:
        """Iterate over the precomputed fake results."""
        return iter(self._results)

    def _generate(self) -> Iterator[ReviewResultLike]:
        """Drive the P2-03 coordinator over every fixture row and both variants."""
        fixture_path = resolve_path(self._settings.fixtures_path)
        if not fixture_path.is_file():
            raise ValidationError(f"missing fixture labels: {fixture_path}")
        for line in fixture_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("synthetic") is not True:
                raise ValidationError("review metrics require synthetic fixture rows only")
            event = json.loads(resolve_path(str(row["event_path"])).read_text(encoding="utf-8"))
            event_id = str(event["event_id"])
            routing = _routing_label(row.get("failure_type"))
            for blind, reviewer_id in (
                (True, BLIND_REVIEWER_ID),
                (False, PREFILLED_REVIEWER_ID),
            ):
                item = SimpleQueueItem(
                    event_id=event_id,
                    call_id=str(row["call_id"]),
                    turn_index=int(row["turn_index"]),
                    audio_uri=str(event["audio"]["uri"]),
                    hypothesis=str(event["stt"]["text"]),
                    no_prefill=blind,
                    training_consent=bool(row.get("consent", {}).get("training", False)),
                )
                script = ScriptedReview(
                    corrected_transcript=str(row["expected_transcript"]),
                    slots=dict(row["expected_slots"]),
                    routing=routing,
                    reviewer_id=reviewer_id,
                    warnings=(FAKE_REVIEW_WARNING,),
                    duration_seconds=self._settings.scripted_duration_seconds,
                )
                tool = InMemoryReviewTool({event_id: script}, clock=self._clock)
                store = InMemoryReviewStore()
                coordinator = ReviewCoordinator(tool, store)
                coordinator.run_all(SequenceQueueSource((item,)))
                stored = store.all()
                if len(stored) != 1:
                    raise ValidationError("P2-03 fake review did not store exactly one result")
                yield stored[0]
