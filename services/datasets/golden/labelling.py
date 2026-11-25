"""Human labelling workflow behind a tool-agnostic interface.

The Label Studio/Argilla choice (context.md section 11) is hidden behind
:class:`LabelTool`. Only an in-memory fake exists here; it performs no network
I/O. Exports carry ``call_id``, ``turn_index``, ``audio_uri`` and ``hypothesis``;
imports carry a corrected transcript, slots and a reviewer ID. A second review
must come from a distinct reviewer, and agreement is recorded per item.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from voiceloop_common import ConfigError, ValidationError, validate

from .audit import canonical_json, utc_timestamp
from .sampler import SourceTurn
from .settings import GoldenSettings, LabelToolName

ROUTING_LABELS = ("stt_wrong", "parser_wrong", "both_fine", "unusable_audio")


@dataclass(frozen=True)
class ExportRecord:
    """The minimal item sent to a labelling tool."""

    source_id: str
    call_id: str
    turn_index: int
    audio_uri: str
    hypothesis: str


@dataclass(frozen=True)
class LabelResult:
    """One reviewer's imported label for one exported item."""

    source_id: str
    corrected_transcript: str
    slots: dict[str, Any]
    reviewer_id: str
    routing: str | None = None
    reviewed_at: str = ""

    def __post_init__(self) -> None:
        """Validate identity, shape and (when present) booking slots."""
        if not self.source_id or not self.reviewer_id.strip():
            raise ValidationError("LabelResult requires source_id and a non-blank reviewer_id")
        if not isinstance(self.corrected_transcript, str) or not isinstance(self.slots, dict):
            raise ValidationError("LabelResult requires text transcript and object slots")
        if self.routing is not None and self.routing not in ROUTING_LABELS:
            raise ValidationError(f"unknown routing label: {self.routing!r}")
        if self.slots:
            validate(self.slots, "booking.schema.json")


@dataclass(frozen=True)
class SecondReview:
    """A distinct reviewer's verdict on one primary label."""

    source_id: str
    first_reviewer_id: str
    second_reviewer_id: str
    transcript_agreed: bool
    slots_agreed: bool
    reviewed_at: str

    @property
    def agreed(self) -> bool:
        """Return true only when transcript and slots both agree."""
        return self.transcript_agreed and self.slots_agreed


@dataclass(frozen=True)
class LabelRecord:
    """One sealed label: source identity plus human-corrected content."""

    source_id: str
    call_id: str
    turn_index: int
    corrected_transcript: str
    slots: dict[str, Any]
    first_reviewer_id: str
    second_reviewer_id: str
    agreed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of the record."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LabelRecord:
        """Rebuild a record from its on-disk JSON object."""
        slots = data.get("slots", {})
        if not isinstance(slots, dict):
            raise ValidationError("sealed label slots must be an object")
        return cls(
            source_id=str(data["source_id"]),
            call_id=str(data["call_id"]),
            turn_index=int(data["turn_index"]),
            corrected_transcript=str(data["corrected_transcript"]),
            slots=dict(slots),
            first_reviewer_id=str(data["first_reviewer_id"]),
            second_reviewer_id=str(data["second_reviewer_id"]),
            agreed=bool(data["agreed"]),
        )


@dataclass(frozen=True)
class ReviewCoverage:
    """Second-review coverage and agreement counts for a sealed version."""

    total: int
    second_review_count: int
    agreement_count: int
    disagreement_count: int

    @property
    def ratio(self) -> float:
        """Return second reviews divided by total labels (0.0 when empty)."""
        return self.second_review_count / self.total if self.total else 0.0


class LabelTool(Protocol):
    """Stable boundary for a human labelling backend; no network in v1."""

    def push(self, items: Sequence[ExportRecord]) -> list[str]:
        """Push export items and return their tool item IDs."""
        ...

    def pull(self, item_ids: Sequence[str]) -> dict[str, tuple[LabelResult, ...]]:
        """Return all submitted labels per item ID, in submission order."""
        ...


class InMemoryLabelTool:
    """Fake labelling tool backed by dictionaries; suitable for unit tests."""

    def __init__(self) -> None:
        """Create an empty item store and result store."""
        self._items: dict[str, ExportRecord] = {}
        self._results: dict[str, list[LabelResult]] = {}

    def push(self, items: Sequence[ExportRecord]) -> list[str]:
        """Store export items and return their source IDs as item IDs."""
        for item in items:
            if item.source_id in self._items:
                raise ValidationError(f"duplicate export item: {item.source_id}")
            self._items[item.source_id] = item
            self._results[item.source_id] = []
        return [item.source_id for item in items]

    def put_result(
        self,
        source_id: str,
        *,
        reviewer_id: str,
        corrected_transcript: str,
        slots: dict[str, Any],
        routing: str | None = None,
    ) -> LabelResult:
        """Submit one reviewer result for an exported item."""
        if source_id not in self._items:
            raise ValidationError(f"unknown export item: {source_id}")
        result = LabelResult(
            source_id=source_id,
            reviewer_id=reviewer_id,
            corrected_transcript=corrected_transcript,
            slots=dict(slots),
            routing=routing,
            reviewed_at=utc_timestamp(),
        )
        self._results[source_id].append(result)
        return result

    def pull(self, item_ids: Sequence[str]) -> dict[str, tuple[LabelResult, ...]]:
        """Return copies of all submitted results for the requested items."""
        unknown = [item_id for item_id in item_ids if item_id not in self._items]
        if unknown:
            raise ValidationError(f"unknown export items: {', '.join(unknown)}")
        return {item_id: tuple(self._results[item_id]) for item_id in item_ids}


def build_label_tool(settings: GoldenSettings | None = None) -> LabelTool:
    """Build the settings-selected labelling backend (in-memory fake only)."""
    chosen: LabelToolName = (settings or GoldenSettings()).label_tool
    if chosen != "in_memory":
        raise ConfigError(f"unsupported label tool: {chosen!r}")
    return InMemoryLabelTool()


def export_to_tool(turns: Sequence[SourceTurn], tool: LabelTool) -> list[str]:
    """Export failure-blind source turns to the labelling tool."""
    items = [
        ExportRecord(
            source_id=turn.source_id,
            call_id=turn.call_id,
            turn_index=turn.turn_index,
            audio_uri=turn.audio_uri,
            hypothesis=turn.hypothesis,
        )
        for turn in turns
    ]
    return tool.push(items)


def import_labels(tool: LabelTool, item_ids: Sequence[str]) -> dict[str, LabelResult]:
    """Import the first human label submitted for each item."""
    primaries: dict[str, LabelResult] = {}
    for source_id, submissions in tool.pull(item_ids).items():
        if not submissions:
            raise ValidationError(f"no label imported for {source_id}")
        primaries[source_id] = submissions[0]
    return primaries


def import_second_reviews(
    tool: LabelTool,
    item_ids: Sequence[str],
    primaries: Mapping[str, LabelResult],
) -> dict[str, SecondReview]:
    """Import second submissions, requiring a distinct reviewer for each."""
    seconds: dict[str, SecondReview] = {}
    for source_id, submissions in tool.pull(item_ids).items():
        if not submissions:
            continue
        if len(submissions) < 2:
            raise ValidationError(f"missing second review for {source_id}")
        primary = primaries[source_id]
        seconds[source_id] = record_second_review(primary, submissions[1])
    return seconds


def record_second_review(primary: LabelResult, second: LabelResult) -> SecondReview:
    """Verify distinct reviewers and record transcript/slot agreement.

    Args:
        primary: First reviewer's imported label.
        second: Second reviewer's imported label.

    Returns:
        The second-review verdict, including separate agreement flags.

    Raises:
        ValidationError: If both labels belong to the same source item or
            reviewer, or if no second reviewer is identified.

    """
    if primary.source_id != second.source_id:
        raise ValidationError("second review must refer to the same source item")
    if not second.reviewer_id.strip() or second.reviewer_id == primary.reviewer_id:
        raise ValidationError("second reviewer must be a distinct identified human")
    return SecondReview(
        source_id=primary.source_id,
        first_reviewer_id=primary.reviewer_id,
        second_reviewer_id=second.reviewer_id,
        transcript_agreed=primary.corrected_transcript == second.corrected_transcript,
        slots_agreed=canonical_json(primary.slots) == canonical_json(second.slots),
        reviewed_at=second.reviewed_at or utc_timestamp(),
    )


def build_label_records(
    turns: Sequence[SourceTurn],
    primaries: Mapping[str, LabelResult],
    seconds: Mapping[str, SecondReview],
) -> tuple[LabelRecord, ...]:
    """Combine source IDs, imported labels and second reviews into seal input."""
    records: list[LabelRecord] = []
    for turn in turns:
        primary = primaries[turn.source_id]
        second = seconds.get(turn.source_id)
        records.append(
            LabelRecord(
                source_id=turn.source_id,
                call_id=turn.call_id,
                turn_index=turn.turn_index,
                corrected_transcript=primary.corrected_transcript,
                slots=dict(primary.slots),
                first_reviewer_id=primary.reviewer_id,
                second_reviewer_id=second.second_reviewer_id if second else "",
                agreed=second.agreed if second else False,
            )
        )
    return tuple(sorted(records, key=lambda record: record.source_id))


def compute_coverage(
    labels: Sequence[LabelRecord],
    seconds: Mapping[str, SecondReview],
) -> ReviewCoverage:
    """Count distinct-reviewer coverage and agreement across labels."""
    reviewed = [seconds[label.source_id] for label in labels if label.source_id in seconds]
    return ReviewCoverage(
        total=len(labels),
        second_review_count=len(reviewed),
        agreement_count=sum(1 for review in reviewed if review.agreed),
        disagreement_count=sum(1 for review in reviewed if not review.agreed),
    )


def synthetic_labels_from_turns(
    turns: Sequence[SourceTurn],
    *,
    primary_reviewer_id: str,
    second_reviewer_id: str,
) -> tuple[LabelRecord, ...]:
    """Create simulated labels from fixture expectations (NOT human review).

    This exists only to exercise the sealing workflow with the read-only
    synthetic fixture pool. The WAVs are tone/noise/silence carriers, so no
    human listened to these items and this output must never be used as a real
    golden set or for promotion decisions.
    """
    primaries: dict[str, LabelResult] = {}
    seconds: dict[str, SecondReview] = {}
    for turn in turns:
        label = LabelResult(
            source_id=turn.source_id,
            corrected_transcript=turn.corrected_transcript,
            slots=dict(turn.slots),
            reviewer_id=primary_reviewer_id,
        )
        primaries[turn.source_id] = label
        seconds[turn.source_id] = record_second_review(
            label,
            LabelResult(
                source_id=turn.source_id,
                corrected_transcript=turn.corrected_transcript,
                slots=dict(turn.slots),
                reviewer_id=second_reviewer_id,
            ),
        )
    return build_label_records(turns, primaries, seconds)
