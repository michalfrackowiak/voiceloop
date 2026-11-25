"""Descriptive drift comparison between the sealed golden set and fixture traffic.

The incoming side is a synthetic fixture replay, not live traffic. Counts and
shares are descriptive only: this module makes no causal claim, no promotion
decision and no threshold change. Only failure-blind categorical metadata is
compared; transcripts, slot values, triage labels and audio are never read.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal

from voiceloop_common import ValidationError

from .candidates import (
    FixtureProjection,
    load_projected_turns,
    select_projected_calls,
    verify_fixture_hash,
)
from .settings import (
    ARTIFACT_LABEL,
    GoldenMaintenanceSettings,
    assert_outside_sealed,
    assert_safe_text,
    canonical_json,
    contains_pii_pattern,
    resolve_path,
    utc_timestamp,
)
from .versioning import SealedProvenance, sealed_provenance, sealed_source_keys

SnapshotSource = Literal["sealed_golden", "synthetic_incoming"]
_FEATURES: tuple[str, ...] = ("audio_kind", "sample_rate_hz", "slot_present")
HUMAN_REVIEW_GATE: Final = "required_before_any_golden_or_threshold_change"
"""Mandatory gate before any golden-set or threshold decision is made."""
_INTERPRETATION: dict[str, str] = {
    "mode": "descriptive_only",
    "causal_claim": "none",
    "promotion": "not_evaluated",
    "threshold_change": "none",
    "alarm": "not_evaluated",
    "human_review_gate": HUMAN_REVIEW_GATE,
    "note": "Fixture replay counts only; no live traffic, human labels or action taken.",
}


@dataclass(frozen=True)
class DistributionSnapshot:
    """Explicit sample counts and categorical distributions for one source."""

    source: SnapshotSource
    sample_count: int
    call_count: int
    features: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of the snapshot."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DistributionSnapshot:
        """Rebuild and validate a snapshot from a reported object."""
        try:
            snapshot = cls(
                source=data["source"],
                sample_count=int(data["sample_count"]),
                call_count=int(data["call_count"]),
                features={str(key): dict(value) for key, value in data["features"].items()},
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"malformed distribution snapshot: {exc}") from exc
        if snapshot.source not in ("sealed_golden", "synthetic_incoming"):
            raise ValidationError("unknown distribution snapshot source")
        if snapshot.sample_count < 1 or snapshot.call_count < 1:
            raise ValidationError("snapshot sample and call counts must be positive")
        if set(snapshot.features) != set(_FEATURES):
            raise ValidationError("snapshot features do not match the failure-blind allowlist")
        return snapshot


def snapshot_from_rows(
    rows: Sequence[FixtureProjection], source: SnapshotSource
) -> DistributionSnapshot:
    """Build a snapshot from in-memory fixture projections only.

    Args:
        rows: Failure-blind projected rows.
        source: Which side of the descriptive comparison this snapshot represents.

    Returns:
        Explicit turn and call counts plus sorted per-feature counts.

    """
    if not rows:
        raise ValidationError("cannot build a snapshot from zero turns")
    kinds = Counter(row.audio_kind for row in rows)
    rates = Counter(str(row.sample_rate_hz) for row in rows)
    slots = Counter("present" if row.slot_present else "absent" for row in rows)
    features: dict[str, dict[str, int]] = {
        "audio_kind": {key: kinds[key] for key in sorted(kinds)},
        "sample_rate_hz": {key: rates[key] for key in sorted(rates)},
        "slot_present": {key: slots[key] for key in sorted(slots)},
    }
    return DistributionSnapshot(
        source=source,
        sample_count=len(rows),
        call_count=len({row.call_id for row in rows}),
        features=features,
    )


def golden_snapshot(settings: GoldenMaintenanceSettings) -> DistributionSnapshot:
    """Snapshot the sealed stand-in by matching its source keys to fixture metadata.

    Sealed labels are opened read-only and only their ``source_id`` values are
    used. All 36 sealed source keys must exist in the pinned fixture projection;
    a mismatch fails closed rather than silently dropping a turn.

    Args:
        settings: Typed maintenance settings.

    Returns:
        Distribution snapshot of the sealed source turns.

    Raises:
        ValidationError: If the sealed hash pin fails or source keys are missing.

    """
    sealed_provenance(settings)
    sealed_keys = set(sealed_source_keys(settings))
    rows = load_projected_turns(settings, require_training_consent=False)
    by_source = {row.source_id: row for row in rows}
    missing = sorted(sealed_keys - set(by_source))
    if missing:
        raise ValidationError("sealed source keys are missing from the pinned fixture pool")
    selected = tuple(by_source[key] for key in sorted(sealed_keys))
    return snapshot_from_rows(selected, "sealed_golden")


def incoming_snapshot(settings: GoldenMaintenanceSettings) -> DistributionSnapshot:
    """Snapshot a deterministic whole-call synthetic fixture replay.

    Args:
        settings: Typed maintenance settings supplying the replay seed and call cap.

    Returns:
        Distribution snapshot of the synthetic incoming stand-in.

    """
    verify_fixture_hash(settings)
    rows = load_projected_turns(settings)
    selection = select_projected_calls(
        rows, seed=settings.drift_seed, max_calls=settings.drift_calls
    )
    return snapshot_from_rows(selection.rows, "synthetic_incoming")


def compare_snapshots(
    golden: DistributionSnapshot, incoming: DistributionSnapshot
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Return descriptive count and share deltas; no alarm or decision is made."""
    deltas: dict[str, dict[str, dict[str, float | int]]] = {}
    for feature in _FEATURES:
        golden_counts = golden.features.get(feature, {})
        incoming_counts = incoming.features.get(feature, {})
        feature_deltas: dict[str, dict[str, float | int]] = {}
        for value in sorted(set(golden_counts) | set(incoming_counts)):
            golden_count = golden_counts.get(value, 0)
            incoming_count = incoming_counts.get(value, 0)
            golden_share = golden_count / golden.sample_count
            incoming_share = incoming_count / incoming.sample_count
            feature_deltas[value] = {
                "golden_count": golden_count,
                "incoming_count": incoming_count,
                "golden_share": round(golden_share, 6),
                "incoming_share": round(incoming_share, 6),
                "share_delta": round(incoming_share - golden_share, 6),
            }
        deltas[feature] = feature_deltas
    return deltas


def build_drift_report(settings: GoldenMaintenanceSettings) -> dict[str, Any]:
    """Build a PII-free, descriptive-only drift report from fixture stand-ins.

    Args:
        settings: Typed maintenance settings.

    Returns:
        Report with explicit sample counts, distributions, deltas and a fixed
        interpretation that rules out causation, promotion and threshold changes.

    """
    provenance: SealedProvenance = sealed_provenance(settings)
    golden = golden_snapshot(settings)
    incoming = incoming_snapshot(settings)
    report: dict[str, Any] = {
        "label": ARTIFACT_LABEL,
        "generated_at": utc_timestamp(),
        "sealed_version": provenance.version,
        "sealed_content_hash": provenance.content_hash,
        "sealed_manifest_hash": provenance.manifest_hash,
        "fixture_sha256": provenance.fixture_sha256,
        "sample_counts": {
            "sealed_golden_turns": golden.sample_count,
            "sealed_golden_calls": golden.call_count,
            "synthetic_incoming_turns": incoming.sample_count,
            "synthetic_incoming_calls": incoming.call_count,
        },
        "golden": golden.to_dict(),
        "incoming": incoming.to_dict(),
        "distribution_deltas": compare_snapshots(golden, incoming),
        "interpretation": dict(_INTERPRETATION),
        "promotion": "not_evaluated",
        "threshold_change": "none",
        "human_review_gate": HUMAN_REVIEW_GATE,
    }
    errors = validate_drift_report(report)
    if errors:
        raise ValidationError("drift report failed safety validation: " + "; ".join(errors))
    return report


def validate_drift_report(report: Mapping[str, Any]) -> tuple[str, ...]:
    """Return structural and privacy errors for a drift report object."""
    errors: list[str] = []
    if report.get("label") != ARTIFACT_LABEL:
        errors.append("drift report lacks the mandatory synthetic stand-in label")
    if report.get("interpretation") != _INTERPRETATION:
        errors.append("drift interpretation is not the fixed descriptive-only text")
    if report.get("human_review_gate") != HUMAN_REVIEW_GATE:
        errors.append("drift report lacks the mandatory human-review gate")
    if report.get("promotion") != "not_evaluated" or report.get("threshold_change") != "none":
        errors.append("drift report must not make a promotion or threshold decision")
    golden: DistributionSnapshot | None = None
    incoming: DistributionSnapshot | None = None
    try:
        golden = DistributionSnapshot.from_dict(report["golden"])
        incoming = DistributionSnapshot.from_dict(report["incoming"])
    except (KeyError, ValidationError) as exc:
        errors.append(str(exc))
    counts = report.get("sample_counts")
    if not isinstance(counts, dict):
        errors.append("drift report has no sample counts")
    elif golden is not None and incoming is not None:
        expected = {
            "sealed_golden_turns": golden.sample_count,
            "sealed_golden_calls": golden.call_count,
            "synthetic_incoming_turns": incoming.sample_count,
            "synthetic_incoming_calls": incoming.call_count,
        }
        if counts != expected:
            errors.append("sample counts do not match the snapshots")
    if "distribution_deltas" not in report:
        errors.append("drift report has no distribution deltas")
    text = canonical_json(report)
    if contains_pii_pattern(text):
        errors.append("drift report contains a PII-like pattern")
    for key, value in report.items():
        if isinstance(value, str):
            try:
                assert_safe_text(value, key)
            except ValidationError as exc:
                errors.append(str(exc))
    return tuple(errors)


def write_drift_report(
    report: dict[str, Any], path: str | Path, settings: GoldenMaintenanceSettings
) -> Path:
    """Validate and write a drift report outside the sealed artifact directory.

    Args:
        report: Report from :func:`build_drift_report`.
        path: Destination JSON path.
        settings: Typed maintenance settings for the sealed-path guard.

    Returns:
        The written path.

    Raises:
        ValidationError: If the report is unsafe or the destination is sealed.

    """
    errors = validate_drift_report(report)
    if errors:
        raise ValidationError("cannot write an unsafe drift report: " + "; ".join(errors))
    assert_outside_sealed(path, settings)
    destination = resolve_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(canonical_json(report) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination
