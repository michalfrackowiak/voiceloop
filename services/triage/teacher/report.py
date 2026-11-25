"""Exploratory synthetic queue arithmetic and the internal teacher report.

Reviewed labels are injected through :class:`ReviewedLabelSource` and are used
**only** to compute synthetic arithmetic after selection. They never enter
batch transcription, disagreement scoring or selector logic. The sealed golden
stand-in is hash-verified read-only and is never read for labels or tuned on.

Every number here is explicitly EXPLORATORY: no operating point was pre-declared,
the reviewed labels are synthetic annotations rather than human-reviewed labels,
and no precision-improvement claim is made (P4-04 is NOT DONE).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

from services.datasets.golden.seal import verify_seal
from voiceloop_common import ValidationError

from .batch import (
    REPO_ROOT,
    BatchResult,
    TurnRecord,
    fixture_inputs_digest,
    resolve_repo_path,
)
from .scorer import QueueCandidate, QueueSelector
from .settings import ARTIFACT_SCHEMA_VERSION, ARTIFACT_TYPE, RUN_LABEL, TeacherSettings

FAILURE_TYPES = frozenset({"stt_wrong", "parser_wrong", "unusable_audio"})
TRANSCRIPTION_FAILURE_TYPES = frozenset({"stt_wrong", "unusable_audio"})
_REQUIRED_REPORT_KEYS = (
    "artifact_type",
    "artifact_schema_version",
    "status",
    "provenance",
    "settings",
    "batch",
    "signals",
    "precision",
    "invariants",
)


@dataclass(frozen=True)
class ReviewedLabel:
    """A synthetic reviewed label used only for precision accounting."""

    event_id: str
    call_id: str
    turn_index: int
    routing_label: str
    needs_review: bool
    needs_transcription_review: bool
    synthetic: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable label view."""
        return {
            "event_id": self.event_id,
            "call_id": self.call_id,
            "turn_index": self.turn_index,
            "routing_label": self.routing_label,
            "needs_review": self.needs_review,
            "needs_transcription_review": self.needs_transcription_review,
            "synthetic": self.synthetic,
        }


@runtime_checkable
class ReviewedLabelSource(Protocol):
    """Injectable seam supplying synthetic reviewed labels for evaluation."""

    def label_for(self, record: TurnRecord) -> ReviewedLabel:
        """Return the reviewed label for one turn."""
        ...


class FixtureReviewedLabelSource:
    """Read-only fake reading ``failure_type`` from the fixture labels manifest.

    Labels from this source are the evaluation ground truth for the rehearsal
    only. They are deliberately unavailable to :mod:`batch`,
    :mod:`disagreement` and :mod:`scorer`.
    """

    def __init__(
        self, turns_path: str | Path = "tests/fixtures/turns.jsonl", *, root: Path | None = None
    ) -> None:
        """Index the fixture manifest by ``call_id:turn_index``.

        Args:
            turns_path: Repo-relative or absolute labels manifest path.
            root: Base directory for relative paths; defaults to the repository root.

        """
        self._root = root or REPO_ROOT
        self._path = resolve_repo_path(turns_path, root=root)
        self._labels: dict[str, ReviewedLabel] = {}
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValidationError(f"cannot read reviewed labels: {self._path}") from exc
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"invalid fixture JSON at line {number}") from exc
            if not isinstance(row, Mapping):
                raise ValidationError(f"fixture line {number} is not an object")
            if row.get("synthetic") is not True:
                raise ValidationError(f"fixture line {number} is not labelled synthetic")
            call_id = str(row.get("call_id", ""))
            turn_index = row.get("turn_index")
            if not call_id or isinstance(turn_index, bool) or not isinstance(turn_index, int):
                raise ValidationError(f"fixture line {number} needs call_id and turn_index")
            routing = str(row.get("failure_type", "unknown"))
            key = f"{call_id}:{turn_index}"
            if key in self._labels:
                raise ValidationError(f"duplicate reviewed label for {key}")
            self._labels[key] = ReviewedLabel(
                event_id="",  # fixture manifest identifies rows by call/turn, not event_id
                call_id=call_id,
                turn_index=turn_index,
                routing_label=routing,
                needs_review=routing in FAILURE_TYPES,
                needs_transcription_review=routing in TRANSCRIPTION_FAILURE_TYPES,
            )

    def label_for(self, record: TurnRecord) -> ReviewedLabel:
        """Return the synthetic label matching ``call_id:turn_index``."""
        label = self._labels.get(record.source_id)
        if label is None:
            raise ValidationError(f"no reviewed label for {record.source_id}")
        return label


class InMemoryReviewedLabelSource:
    """Deterministic in-memory fake keyed by event ID, for unit tests."""

    def __init__(self, labels: Mapping[str, ReviewedLabel]) -> None:
        """Index caller-supplied labels.

        Args:
            labels: Event ID to synthetic reviewed label.

        """
        self._labels = dict(labels)

    def label_for(self, record: TurnRecord) -> ReviewedLabel:
        """Return the indexed label or raise :class:`ValidationError`."""
        label = self._labels.get(record.event_id)
        if label is None:
            raise ValidationError(f"no reviewed label for {record.event_id}")
        return label


@dataclass(frozen=True)
class PrecisionResult:
    """Precision and counts for one injected queue selector."""

    selector: str
    selected: tuple[str, ...]
    denominator: int
    tp_overall: int
    tp_transcription: int
    positives_overall: int
    positives_transcription: int
    precision_overall: float
    precision_transcription: float
    recall_overall: float
    recall_transcription: float

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable precision view."""
        return {
            "selector": self.selector,
            "selected": list(self.selected),
            "denominator": self.denominator,
            "tp_overall": self.tp_overall,
            "tp_transcription": self.tp_transcription,
            "positives_overall": self.positives_overall,
            "positives_transcription": self.positives_transcription,
            "precision_overall": self.precision_overall,
            "precision_transcription": self.precision_transcription,
            "recall_overall": self.recall_overall,
            "recall_transcription": self.recall_transcription,
        }


@dataclass(frozen=True)
class DepthComparison:
    """Exploratory synthetic arithmetic for both selectors at one queue depth."""

    queue_size: int
    baseline: PrecisionResult
    enhanced: PrecisionResult
    delta_overall: float
    delta_transcription: float

    def to_dict(self) -> dict[str, Any]:
        """Return an exploratory depth view with counts and Wilson intervals."""
        n = self.queue_size
        baseline_overall_ci = _wilson_interval(self.baseline.tp_overall, n)
        enhanced_overall_ci = _wilson_interval(self.enhanced.tp_overall, n)
        baseline_transcription_ci = _wilson_interval(self.baseline.tp_transcription, n)
        enhanced_transcription_ci = _wilson_interval(self.enhanced.tp_transcription, n)
        overall_overlap = _intervals_overlap(baseline_overall_ci, enhanced_overall_ci)
        transcription_overlap = _intervals_overlap(
            baseline_transcription_ci, enhanced_transcription_ci
        )
        overlap_note = (
            "Wilson 95% intervals overlap for both metrics"
            if overall_overlap and transcription_overlap
            else "Wilson 95% intervals do not overlap for every metric"
        )
        return {
            "queue_size": n,
            "n": n,
            "baseline_tp_overall": self.baseline.tp_overall,
            "enhanced_tp_overall": self.enhanced.tp_overall,
            "baseline_tp_transcription": self.baseline.tp_transcription,
            "enhanced_tp_transcription": self.enhanced.tp_transcription,
            "baseline_precision_overall": self.baseline.precision_overall,
            "baseline_precision_transcription": self.baseline.precision_transcription,
            "enhanced_precision_overall": self.enhanced.precision_overall,
            "enhanced_precision_transcription": self.enhanced.precision_transcription,
            "baseline_precision_overall_ci95": list(baseline_overall_ci),
            "enhanced_precision_overall_ci95": list(enhanced_overall_ci),
            "baseline_precision_transcription_ci95": list(baseline_transcription_ci),
            "enhanced_precision_transcription_ci95": list(enhanced_transcription_ci),
            "delta_overall": self.delta_overall,
            "delta_transcription": self.delta_transcription,
            "ci95_overlap": {
                "overall": overall_overlap,
                "transcription": transcription_overlap,
            },
            "confidence_note": (
                f"EXPLORATORY ONLY: n={n} synthetic labels; {overlap_note}; "
                "insufficient n and no pre-declared operating point for a precision claim."
            ),
        }


@dataclass(frozen=True)
class PrecisionComparison:
    """Exploratory synthetic arithmetic over identical synthetic labels.

    This is deliberately not a precision claim: no operating point was
    pre-declared, labels are synthetic annotations, and every delta is computed
    after seeing those labels.
    """

    queue_size: int
    turns: int
    calls: int
    baseline: PrecisionResult
    enhanced: PrecisionResult
    delta_overall: float
    delta_transcription: float
    depth_curve: tuple[DepthComparison, ...]
    claim: str

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable exploratory comparison view."""
        return {
            "queue_size": self.queue_size,
            "turns": self.turns,
            "calls": self.calls,
            "reviewed_labels_synthetic": True,
            "baseline": self.baseline.to_dict(),
            "teacher_enhanced": self.enhanced.to_dict(),
            "delta_overall": self.delta_overall,
            "delta_transcription": self.delta_transcription,
            "depth_curve": [point.to_dict() for point in self.depth_curve],
            "exploratory_note": (
                "EXPLORATORY ONLY: every depth was inspected after seeing the "
                "synthetic labels; no operating point was pre-declared, label "
                "counts are small, and Wilson 95% intervals overlap at these "
                "depths. Not a precision claim; P4-04 is NOT DONE pending a "
                "pre-declared operating point, human-reviewed labels and a real "
                "teacher model."
            ),
            "claim": self.claim,
        }


def _ratio(numerator: int, denominator: int) -> float:
    """Return a stable six-decimal ratio, or 0.0 for an empty denominator."""
    return round(numerator / denominator, 6) if denominator else 0.0


def _wilson_interval(successes: int, trials: int, *, z: float = 1.96) -> tuple[float, float]:
    """Return a rounded Wilson score interval for a binomial proportion.

    Args:
        successes: Number of positive selections (true positives).
        trials: Queue depth; zero trials yields the uninformative ``(0, 1)``.
        z: Normal quantile; 1.96 is the two-sided 95% default.

    Returns:
        A ``(low, high)`` pair clamped to ``[0, 1]``.

    """
    if trials <= 0:
        return (0.0, 1.0)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    low = max(0.0, centre - half_width)
    high = min(1.0, centre + half_width)
    return (round(low, 6), round(high, 6))


def _intervals_overlap(left: tuple[float, float], right: tuple[float, float]) -> bool:
    """Return whether two closed intervals share any point."""
    return not (left[1] < right[0] or right[1] < left[0])


def _evaluate_selector(
    selector: QueueSelector,
    candidates: Sequence[QueueCandidate],
    labels: Mapping[str, ReviewedLabel],
    limit: int,
) -> PrecisionResult:
    """Evaluate one selector against synthetic reviewed labels."""
    selected = selector.select(candidates, limit)
    if len(set(selected)) != len(selected):
        raise ValidationError(f"selector {selector.name} returned duplicate event IDs")
    if len(selected) > limit:
        raise ValidationError(f"selector {selector.name} exceeded the queue cap")
    by_id = {candidate.event_id: candidate for candidate in candidates}
    missing = [event_id for event_id in selected if event_id not in by_id]
    if missing:
        raise ValidationError(f"selector {selector.name} returned unknown event IDs: {missing}")
    positives_overall = sum(1 for label in labels.values() if label.needs_review)
    positives_transcription = sum(
        1 for label in labels.values() if label.needs_transcription_review
    )
    tp_overall = 0
    tp_transcription = 0
    for event_id in selected:
        label = labels.get(event_id)
        if label is None:
            raise ValidationError(f"no reviewed label for selected event {event_id}")
        if label.synthetic is not True:
            raise ValidationError(f"non-synthetic reviewed label for {event_id}")
        tp_overall += int(label.needs_review)
        tp_transcription += int(label.needs_transcription_review)
    denominator = len(selected)
    return PrecisionResult(
        selector=selector.name,
        selected=selected,
        denominator=denominator,
        tp_overall=tp_overall,
        tp_transcription=tp_transcription,
        positives_overall=positives_overall,
        positives_transcription=positives_transcription,
        precision_overall=_ratio(tp_overall, denominator),
        precision_transcription=_ratio(tp_transcription, denominator),
        recall_overall=_ratio(tp_overall, positives_overall),
        recall_transcription=_ratio(tp_transcription, positives_transcription),
    )


def evaluate_precision(
    candidates: Sequence[QueueCandidate],
    turns: Sequence[TurnRecord],
    labels: ReviewedLabelSource,
    baseline_selector: QueueSelector,
    enhanced_selector: QueueSelector,
    settings: TeacherSettings | None = None,
) -> PrecisionComparison:
    """Compute exploratory synthetic queue arithmetic against synthetic labels.

    No depth is pre-declared as an operating point and no improvement is
    claimed. The full curve is disclosed so post-hoc selection is visible.

    Args:
        candidates: Queue candidates produced without label access.
        turns: The same turns in the same order, for label lineage.
        labels: Synthetic annotation seam used only here; not human review.
        baseline_selector: Injected selector that ignores the teacher signal.
        enhanced_selector: Injected selector that consumes the teacher signal.
        settings: Typed settings; defaults to environment-backed values.

    Returns:
        An exploratory comparison with counts, precisions, recalls, deltas,
        Wilson intervals and explicit insufficient-n notes.

    Raises:
        ValidationError: If labels or selectors are incomplete.

    """
    config = settings or TeacherSettings()
    if len(candidates) != len(turns):
        raise ValidationError("candidates and turns must be 1:1")
    by_id: dict[str, ReviewedLabel] = {}
    for candidate, turn in zip(candidates, turns, strict=True):
        if candidate.event_id != turn.event_id:
            raise ValidationError("candidate and turn order disagree")
        label = labels.label_for(turn)
        if label.event_id not in ("", candidate.event_id):
            raise ValidationError(f"reviewed label event_id mismatch for {candidate.event_id}")
        by_id[candidate.event_id] = label
    # Report every depth up to the configured cap rather than choosing one.
    # The full curve is EXPLORATORY arithmetic on synthetic annotations, never
    # a pre-declared operating point or a precision-improvement claim.
    max_depth = min(config.queue_size, len(candidates))
    curve: list[DepthComparison] = []
    for depth in range(1, max_depth + 1):
        point_baseline = _evaluate_selector(baseline_selector, candidates, by_id, depth)
        point_enhanced = _evaluate_selector(enhanced_selector, candidates, by_id, depth)
        curve.append(
            DepthComparison(
                queue_size=depth,
                baseline=point_baseline,
                enhanced=point_enhanced,
                delta_overall=round(
                    point_enhanced.precision_overall - point_baseline.precision_overall, 6
                ),
                delta_transcription=round(
                    point_enhanced.precision_transcription - point_baseline.precision_transcription,
                    6,
                ),
            )
        )
    if curve:
        baseline, enhanced = curve[-1].baseline, curve[-1].enhanced
    else:
        baseline = _evaluate_selector(baseline_selector, candidates, by_id, config.queue_size)
        enhanced = _evaluate_selector(enhanced_selector, candidates, by_id, config.queue_size)
    return PrecisionComparison(
        queue_size=config.queue_size,
        turns=len(candidates),
        calls=len({candidate.call_id for candidate in candidates}),
        baseline=baseline,
        enhanced=enhanced,
        delta_overall=round(enhanced.precision_overall - baseline.precision_overall, 6),
        delta_transcription=round(
            enhanced.precision_transcription - baseline.precision_transcription, 6
        ),
        depth_curve=tuple(curve),
        claim=(
            "SYNTHETIC/FIXTURE STAND-IN: EXPLORATORY arithmetic against "
            "simulated fixture annotations only, not human-reviewed labels. "
            "Not a production precision measurement, not a promotion metric, "
            "and P4-04 is NOT DONE pending a pre-declared operating point, "
            "adequate human-reviewed labels and a real teacher model."
        ),
    )


@runtime_checkable
class ReportSink(Protocol):
    """Injectable destination for the internal teacher report."""

    def write(self, artifact: Mapping[str, Any]) -> str:
        """Persist the artifact and return an opaque location string."""
        ...


class InMemoryReportSink:
    """Deterministic in-memory fake :class:`ReportSink` for tests."""

    def __init__(self) -> None:
        """Start with no stored artifact."""
        self.artifacts: list[dict[str, Any]] = []

    @property
    def last(self) -> dict[str, Any]:
        """Return the most recently written artifact.

        Raises:
            ValidationError: If nothing has been written yet.

        """
        if not self.artifacts:
            raise ValidationError("in-memory report sink is empty")
        return self.artifacts[-1]

    def write(self, artifact: Mapping[str, Any]) -> str:
        """Store a JSON-round-tripped copy of the artifact."""
        self.artifacts.append(cast(dict[str, Any], json.loads(json.dumps(artifact))))
        return "memory://teacher-report"


class JsonFileReportSink:
    """Writes canonical UTF-8 JSON to a single configured path."""

    def __init__(self, path: str | Path, *, root: Path | None = None) -> None:
        """Set the only path this sink writes.

        Args:
            path: Repo-relative or absolute output path.
            root: Base directory for relative paths; defaults to the repository root.

        """
        base = root or REPO_ROOT
        candidate = Path(path)
        self.path = candidate if candidate.is_absolute() else base / candidate

    def write(self, artifact: Mapping[str, Any]) -> str:
        """Validate and write the artifact, returning its resolved path string."""
        validate_report(artifact)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return str(self.path.resolve())


def verify_read_only_inputs(settings: TeacherSettings) -> dict[str, Any]:
    """Hash the read-only fixture text inputs and verify the golden seal.

    Seal verification is called without audit context so no audit entry is
    appended. Golden labels are never loaded here: only the committed content
    hash is checked (context.md §2 invariant 5).

    Args:
        settings: Typed settings with fixture and golden paths.

    Returns:
        Provenance dict suitable for the report.

    Raises:
        ValidationError: If the seal, hash or synthetic flags do not match.

    """
    verification = verify_seal(
        resolve_repo_path(settings.golden_artifact_dir), settings.golden_version
    )
    verification.assert_ok()
    manifest = verification.manifest
    if manifest is None:  # pragma: no cover - guaranteed by a passing seal
        raise ValidationError("verified golden stand-in has no manifest")
    if manifest.content_hash != settings.expected_golden_content_hash:
        raise ValidationError("golden stand-in content hash changed; refusing to run")
    if not manifest.synthetic or not manifest.workflow_stand_in:
        raise ValidationError("golden stand-in is no longer labelled synthetic/workflow stand-in")
    if manifest.review_mode != "simulated_synthetic_labels":
        raise ValidationError("golden stand-in review mode is not the expected synthetic mode")
    return {
        "fixtures_path": settings.fixtures_path,
        "fixtures_sha256": fixture_inputs_digest(settings.fixtures_path),
        "golden_artifact_dir": settings.golden_artifact_dir,
        "golden_version": manifest.version,
        "golden_dataset_id": manifest.dataset_id,
        "golden_content_hash": manifest.content_hash,
        "golden_seal_ok": True,
        "golden_synthetic": manifest.synthetic,
        "golden_workflow_stand_in": manifest.workflow_stand_in,
        "golden_review_mode": manifest.review_mode,
        "golden_used_for_training_or_tuning": False,
        "golden_labels_loaded": False,
        "public_leaderboards_used": False,
        "real_teacher_model_used": False,
        "audio_bytes_in_artifact": False,
        "synthetic_standin": True,
    }


def build_report(
    settings: TeacherSettings,
    batch: BatchResult,
    candidates: Sequence[QueueCandidate],
    comparison: PrecisionComparison,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble and validate the internal teacher triage artifact.

    Args:
        settings: Typed settings used for this run.
        batch: Deterministic batch teacher output.
        candidates: Baseline/signal candidates in turn order.
        comparison: Synthetic precision comparison.
        provenance: Verified read-only input hashes and flags.

    Returns:
        A JSON-serialisable internal artifact (not a cross-service schema).

    """
    artifact: dict[str, Any] = {
        "artifact_type": ARTIFACT_TYPE,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": RUN_LABEL,
        "provenance": dict(provenance),
        "settings": {
            "teacher_model_id": settings.teacher_model_id,
            "queue_size": settings.queue_size,
            "signal_weight": settings.signal_weight,
            "low_confidence_ceiling": settings.low_confidence_ceiling,
            "no_speech_prob_ceiling": settings.no_speech_prob_ceiling,
            "hash_algorithm": settings.hash_algorithm,
            "fixtures_path": settings.fixtures_path,
            "output_path": settings.output_path,
        },
        "batch": {
            "run_label": batch.run_label,
            "teacher_model_id": batch.teacher_model_id,
            "hash_algorithm": batch.hash_algorithm,
            "fixtures_sha256": batch.fixtures_sha256,
            "turns": batch.turns,
            "calls": batch.calls,
            "records": [record.to_dict() for record in batch.records],
        },
        "baseline_scores": [candidate.baseline.to_dict() for candidate in candidates],
        "signals": [candidate.teacher_signal.to_dict() for candidate in candidates],
        "precision": comparison.to_dict(),
        "invariants": {
            "offline_only": True,
            "teacher_signal_is_hint_not_label": True,
            "p2_01_thresholds_modified": False,
            "p2_01_teacher_seam_used": False,
            "alias_or_gate_state_modified": False,
            "random_slice_or_queue_code_modified": False,
            "reviewed_labels_used_for_selection": False,
            "limitations": (
                "Tone/noise/silence carriers and a deterministic metadata fake, "
                "not a real teacher model, human review or production traffic."
            ),
        },
    }
    validate_report(artifact)
    return artifact


def validate_report(artifact: Mapping[str, Any]) -> None:
    """Validate the internal artifact shape and review-safety flags.

    Raises:
        ValidationError: If the artifact is missing fields or claims production use.

    """
    missing = [key for key in _REQUIRED_REPORT_KEYS if key not in artifact]
    if missing:
        raise ValidationError(f"teacher report missing: {', '.join(missing)}")
    if artifact["artifact_type"] != ARTIFACT_TYPE:
        raise ValidationError("unexpected teacher report artifact_type")
    if artifact["artifact_schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ValidationError("unsupported teacher report artifact_schema_version")
    provenance = artifact["provenance"]
    if not isinstance(provenance, Mapping):
        raise ValidationError("provenance must be an object")
    if provenance.get("golden_used_for_training_or_tuning") is not False:
        raise ValidationError("golden-set tuning flag must be false")
    if provenance.get("real_teacher_model_used") is not False:
        raise ValidationError("real teacher model flag must be false")
    if provenance.get("audio_bytes_in_artifact") is not False:
        raise ValidationError("report must not contain audio bytes")
    if provenance.get("fixtures_sha256") != artifact["batch"].get("fixtures_sha256"):
        raise ValidationError("fixture digest in provenance and batch disagrees")
    signals = artifact["signals"]
    records = artifact["batch"].get("records")
    if not isinstance(signals, list) or not isinstance(records, list):
        raise ValidationError("signals and batch records must be arrays")
    if len(signals) != len(records):
        raise ValidationError("signal count does not match teacher batch count")
    invariants = artifact["invariants"]
    if (
        not isinstance(invariants, Mapping)
        or invariants.get("p2_01_thresholds_modified") is not False
    ):
        raise ValidationError("report must not claim modified P2-01 thresholds")
    precision = artifact["precision"]
    if not isinstance(precision, Mapping) or precision.get("reviewed_labels_synthetic") is not True:
        raise ValidationError("precision must be labelled as synthetic")
    for banned in (
        "improved_at_smaller_depth",
        "improved_overall",
        "improved_transcription_review",
    ):
        if banned in precision:
            raise ValidationError(f"precision claim key not allowed: {banned}")
    curve = precision.get("depth_curve")
    if not isinstance(curve, list):
        raise ValidationError("exploratory depth_curve must be an array")
    for point in curve:
        if (
            not isinstance(point, Mapping)
            or point.get("n") != point.get("queue_size")
            or not isinstance(point.get("confidence_note"), str)
            or not isinstance(point.get("baseline_tp_transcription"), int)
            or not isinstance(point.get("enhanced_tp_transcription"), int)
        ):
            raise ValidationError("exploratory depth_curve must expose n, counts and notes")
