"""Outcome-based, paired and descriptive canary metrics.

The primary signal is end-to-end booking-field accuracy from final booking
records (``context.md`` §2 invariant 4): each served call's predicted booking
fields are paired with that call's final record, both joined by pseudonymous
``call_id`` alone. A candidate baseline for the same call is supplied by an
injected seam (P3-07 shadow replay in production); if that baseline or the
outcome is absent or incomplete, the evidence is marked incomplete and the
decision fails closed. WER is never read, computed or reported.

All values stored here are rates, sample counts and SHA-256 digests. Raw
transcripts, audio, names, emails, phone numbers and card digits never enter
this module's outputs. Nothing here promotes a model or claims a champion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from services.release.shadow.runner import field_digests as digest_fields
from services.release.shadow.scoring import OutcomeRecord

from .settings import CanarySettings
from .split import Arm

EvidenceStatus = Literal["complete", "insufficient", "incomplete"]


@dataclass(frozen=True)
class PredictionRecord:
    """PII-safe record of one arm's predicted booking fields for a call."""

    call_id: str
    arm: Arm
    field_digests: Mapping[str, str]
    synthetic: bool
    observed_at: str

    @classmethod
    def from_fields(
        cls,
        call_id: str,
        arm: Arm,
        fields: Mapping[str, Any],
        ignored_fields: frozenset[str],
        *,
        observed_at: datetime,
        synthetic: bool = True,
    ) -> PredictionRecord:
        """Project raw booking fields into digests immediately.

        Args:
            call_id: Pseudonymous join key supplied by the injected event source.
            arm: Arm whose prediction this is.
            fields: In-memory booking fields, which may contain PII.
            ignored_fields: Top-level keys excluded from comparison.
            observed_at: Fake-clock timestamp of the observation.
            synthetic: Whether the source is a labelled fixture stand-in.

        Returns:
            A record containing only digests and allowlisted metadata.

        """
        return cls(
            call_id=call_id,
            arm=arm,
            field_digests=digest_fields(fields, ignored_fields),
            synthetic=synthetic,
            observed_at=observed_at.isoformat().replace("+00:00", "Z"),
        )


@dataclass(frozen=True)
class CanaryPair:
    """One paired observation for a single pseudonymous call.

    ``champion`` is a baseline prediction for the *same* call (replay or P3-07
    shadow output), not a model that served the canary call. Any missing side
    makes the pair incomplete; :func:`score_canary` excludes it and reports the
    exclusion in sample counts.
    """

    call_id: str
    arm: Arm
    canary: PredictionRecord | None
    champion: PredictionRecord | None
    outcome: OutcomeRecord | None


@dataclass(frozen=True)
class CanaryMetrics:
    """Descriptive, paired outcome metrics with explicit sample counts."""

    evidence_label: str
    synthetic: bool
    sample_counts: Mapping[str, int]
    canary_call_accuracy: float
    champion_call_accuracy: float
    canary_field_accuracy: float
    champion_field_accuracy: float
    paired_delta_call_accuracy: float
    paired_delta_field_accuracy: float
    canary_error_rate: float
    champion_error_rate: float
    paired_call_wins: int
    paired_call_losses: int
    paired_call_ties: int

    def to_dict(self) -> dict[str, Any]:
        """Return a PII-safe, labelled report fragment."""
        return {
            "evidence_label": self.evidence_label,
            "synthetic": self.synthetic,
            "primary_metric": "e2e_booking_field_accuracy_from_final_record",
            "sample_counts": dict(self.sample_counts),
            "canary_call_accuracy": self.canary_call_accuracy,
            "champion_call_accuracy": self.champion_call_accuracy,
            "canary_field_accuracy": self.canary_field_accuracy,
            "champion_field_accuracy": self.champion_field_accuracy,
            "paired_delta_call_accuracy": self.paired_delta_call_accuracy,
            "paired_delta_field_accuracy": self.paired_delta_field_accuracy,
            "canary_error_rate": self.canary_error_rate,
            "champion_error_rate": self.champion_error_rate,
            "paired_call_wins": self.paired_call_wins,
            "paired_call_losses": self.paired_call_losses,
            "paired_call_ties": self.paired_call_ties,
            "metric_semantics": (
                "Per-call prediction paired with that call's final booking record; "
                "champion baseline is the same call via replay/shadow; deltas are "
                "descriptive candidate-minus-champion differences, not gate CIs."
            ),
        }


@dataclass(frozen=True)
class RegressionVerdict:
    """Fail-closed decision describing whether a bounded rollback is required."""

    rollback: bool
    reason: str
    evidence_status: EvidenceStatus
    sample_counts: Mapping[str, int]
    evidence_label: str

    def to_dict(self) -> dict[str, Any]:
        """Return a labelled, PII-safe report fragment."""
        return {
            "rollback_required": self.rollback,
            "reason": self.reason,
            "evidence_status": self.evidence_status,
            "sample_counts": dict(self.sample_counts),
            "evidence_label": self.evidence_label,
            "promotion_decision": "not_evaluated",
        }


def _rate(numerator: int, denominator: int) -> float:
    """Return a bounded rate, using zero when no evidence is available."""
    return numerator / denominator if denominator else 0.0


def score_canary(pairs: Sequence[CanaryPair], settings: CanarySettings) -> CanaryMetrics:
    """Score paired, outcome-based canary evidence.

    Args:
        pairs: Bounded observations joined by pseudonymous call ID.
        settings: Scorable statuses and the evidence label.

    Returns:
        Descriptive rates, paired deltas and all sample/exclusion counts.

    """
    counts = {
        "calls_observed": len(pairs),
        "calls_canary": sum(1 for pair in pairs if pair.arm is Arm.CANARY),
        "calls_control": sum(1 for pair in pairs if pair.arm is Arm.CHAMPION),
        "calls_missing_canary_prediction": 0,
        "calls_missing_baseline_prediction": 0,
        "calls_missing_outcome": 0,
        "calls_excluded_by_status": 0,
        "calls_empty_final_record": 0,
        "calls_identity_mismatch": 0,
        "calls_scored": 0,
    }
    scorable = settings.scorable_statuses()
    scored: list[tuple[bool, bool, int, int, int]] = []

    for pair in pairs:
        observed_ids = [
            record.call_id
            for record in (pair.canary, pair.champion, pair.outcome)
            if record is not None
        ]
        if any(call_id != pair.call_id for call_id in observed_ids):
            counts["calls_identity_mismatch"] += 1
            continue
        if pair.arm is not Arm.CANARY:
            # Control traffic is counted but never used to judge the canary arm.
            continue
        if pair.canary is None:
            counts["calls_missing_canary_prediction"] += 1
            continue
        if pair.champion is None:
            counts["calls_missing_baseline_prediction"] += 1
            continue
        if pair.outcome is None:
            counts["calls_missing_outcome"] += 1
            continue
        if pair.outcome.status not in scorable:
            counts["calls_excluded_by_status"] += 1
            continue
        if not pair.outcome.field_digests:
            counts["calls_empty_final_record"] += 1
            continue

        reference = pair.outcome.field_digests
        canary = pair.canary.field_digests
        champion = pair.champion.field_digests
        fields_total = len(set(reference) | set(canary) | set(champion))
        canary_matched = sum(
            1 for name in set(reference) | set(canary) if reference.get(name) == canary.get(name)
        )
        champion_matched = sum(
            1
            for name in set(reference) | set(champion)
            if reference.get(name) == champion.get(name)
        )
        scored.append(
            (
                canary == reference,
                champion == reference,
                canary_matched,
                champion_matched,
                fields_total,
            )
        )

    counts["calls_scored"] = len(scored)
    canary_correct = sum(1 for row in scored if row[0])
    champion_correct = sum(1 for row in scored if row[1])
    canary_matched = sum(row[2] for row in scored)
    champion_matched = sum(row[3] for row in scored)
    fields_total = sum(row[4] for row in scored)

    canary_call_accuracy = _rate(canary_correct, len(scored))
    champion_call_accuracy = _rate(champion_correct, len(scored))
    canary_field_accuracy = _rate(canary_matched, fields_total)
    champion_field_accuracy = _rate(champion_matched, fields_total)

    return CanaryMetrics(
        evidence_label=settings.evidence_label,
        synthetic=True,
        sample_counts=counts,
        canary_call_accuracy=canary_call_accuracy,
        champion_call_accuracy=champion_call_accuracy,
        canary_field_accuracy=canary_field_accuracy,
        champion_field_accuracy=champion_field_accuracy,
        # With the same denominator on both sides, the difference of means is
        # the mean of the paired per-call differences.
        paired_delta_call_accuracy=canary_call_accuracy - champion_call_accuracy,
        paired_delta_field_accuracy=canary_field_accuracy - champion_field_accuracy,
        canary_error_rate=1.0 - canary_call_accuracy if scored else 0.0,
        champion_error_rate=1.0 - champion_call_accuracy if scored else 0.0,
        paired_call_wins=sum(1 for row in scored if row[0] and not row[1]),
        paired_call_losses=sum(1 for row in scored if not row[0] and row[1]),
        paired_call_ties=sum(1 for row in scored if row[0] == row[1]),
    )


def detect_regression(metrics: CanaryMetrics, settings: CanarySettings) -> RegressionVerdict:
    """Return a descriptive, fail-closed verdict for the current evidence.

    Missing, empty or mismatched outcome evidence rolls back rather than being
    read as "no regression". Too little paired evidence is not a pass: the
    runner rolls back at the configured window boundary if it stays insufficient.
    A strictly negative paired delta on either outcome metric is a regression.

    Args:
        metrics: Descriptive metrics from :func:`score_canary`.
        settings: Typed settings supplying the sample-count floor and label.

    Returns:
        A verdict whose ``rollback`` flag drives the bounded controller.

    """
    counts = metrics.sample_counts
    # A non-scorable status is an explicit exclusion, counted below as
    # insufficient evidence; it is not a missing/incomplete outcome.
    incomplete = (
        counts["calls_missing_canary_prediction"]
        + counts["calls_missing_baseline_prediction"]
        + counts["calls_missing_outcome"]
        + counts["calls_empty_final_record"]
        + counts["calls_identity_mismatch"]
    )
    if incomplete:
        return RegressionVerdict(
            rollback=True,
            reason="incomplete_outcome_evidence",
            evidence_status="incomplete",
            sample_counts=counts,
            evidence_label=metrics.evidence_label,
        )
    if counts["calls_scored"] < settings.min_paired_calls:
        return RegressionVerdict(
            rollback=False,
            reason="insufficient_paired_evidence",
            evidence_status="insufficient",
            sample_counts=counts,
            evidence_label=metrics.evidence_label,
        )
    if metrics.paired_delta_call_accuracy < 0.0 or metrics.paired_delta_field_accuracy < 0.0:
        return RegressionVerdict(
            rollback=True,
            reason="outcome_metric_regression",
            evidence_status="complete",
            sample_counts=counts,
            evidence_label=metrics.evidence_label,
        )
    return RegressionVerdict(
        rollback=False,
        reason="no_regression_observed",
        evidence_status="complete",
        sample_counts=counts,
        evidence_label=metrics.evidence_label,
    )
