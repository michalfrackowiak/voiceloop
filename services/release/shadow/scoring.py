"""Outcome-based candidate-vs-champion scoring on shadow results.

Outcomes are joined **by ``call_id`` only** through an injectable
:class:`OutcomeSource`. No transcript, audio, event ID, turn index, name or
other identifier participates in the join. Both sides' booking fields are
compared as SHA-256 digests against the call's final booking record, so raw
guest data is never copied into a comparison artifact.

This module produces evidence for the P3-08 gate. It never reads or writes an
MLflow alias, promotion threshold or gate rule.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .mirror import MirrorStats
from .runner import (
    CandidateRunner,
    ChampionRunner,
    ModelResult,
    ModelSource,
    ResultStatus,
    field_digests,
)
from .settings import ShadowSettings

JOIN_KEY = "call_id"


@dataclass(frozen=True)
class OutcomeRecord:
    """PII-safe call-level outcome snapshot used for scoring."""

    call_id: str
    status: str
    field_digests: Mapping[str, str]
    synthetic: bool
    updated_at: str

    @classmethod
    def from_payload(
        cls,
        call_id: str,
        payload: Mapping[str, Any],
        ignored_fields: frozenset[str],
        synthetic: bool = True,
    ) -> OutcomeRecord:
        """Project a raw outcome payload into digests and safe metadata.

        Args:
            call_id: Pseudonymous join key supplied by the source.
            payload: Outcome block with ``status`` and optional
                ``final_record`` from the booking connector.
            ignored_fields: Final-record keys that are not booking slots.
            synthetic: Flag copied to the artifact provenance.

        Returns:
            An outcome containing only digests and allowlisted metadata.

        """
        final = payload.get("final_record", {})
        fields: Mapping[str, Any] = final if isinstance(final, Mapping) else {}
        return cls(
            call_id=call_id,
            status=str(payload.get("status", "unknown")),
            field_digests=field_digests(fields, ignored_fields),
            synthetic=bool(payload.get("synthetic", synthetic)),
            updated_at=str(payload.get("updated_at", "")),
        )


class OutcomeSource(Protocol):
    """Late-arriving outcome seam, keyed only by ``call_id``."""

    def for_call(self, call_id: str) -> OutcomeRecord | None:
        """Return the outcome for a call, or None when none has arrived."""
        ...


class FakeOutcomeSource:
    """Deterministic in-memory FAKE outcome source with lookup accounting."""

    def __init__(self, outcomes: Mapping[str, OutcomeRecord]) -> None:
        """Store outcomes by the single supported join key."""
        self._outcomes = dict(outcomes)
        self.lookups: list[str] = []

    def add(self, call_id: str, record: OutcomeRecord) -> None:
        """Add or replace one outcome by its single supported join key."""
        self._outcomes[call_id] = record

    def for_call(self, call_id: str) -> OutcomeRecord | None:
        """Record the lookup and return the matching outcome, if any."""
        self.lookups.append(call_id)
        return self._outcomes.get(call_id)


@dataclass(frozen=True)
class ScoredCall:
    """Per-call comparison kept in memory; never serialised artifact payload."""

    call_id: str
    candidate_correct: bool
    champion_correct: bool
    candidate_matched: int
    champion_matched: int
    fields_total: int


@dataclass(frozen=True)
class ComparisonData:
    """In-memory comparison result ready for the internal report builder."""

    settings: ShadowSettings
    mirror_stats: MirrorStats
    results: tuple[ModelResult, ...]
    sample_counts: Mapping[str, int]
    metrics: Mapping[str, Mapping[str, float]]
    deltas: Mapping[str, float]
    scored_calls: tuple[ScoredCall, ...]
    candidate_artifact_uri: str
    champion_artifact_uri: str
    candidate_model_id: str
    champion_model_id: str
    synthetic: bool


def _latest_by_call(results: Sequence[ModelResult], source: ModelSource) -> dict[str, ModelResult]:
    """Select the final successful result per call for one model side."""
    latest: dict[str, ModelResult] = {}
    for result in results:
        if result.source is not source or result.status is not ResultStatus.OK:
            continue
        current = latest.get(result.call_id)
        key = (result.turn_index, result.completed_at)
        if current is None or key > (current.turn_index, current.completed_at):
            latest[result.call_id] = result
    return latest


def _rate(numerator: int, denominator: int) -> float:
    """Return a bounded rate, using zero when there is no evidence."""
    return numerator / denominator if denominator else 0.0


def score_comparison(
    *,
    settings: ShadowSettings,
    results: Sequence[ModelResult],
    outcome_source: OutcomeSource,
    mirror_stats: MirrorStats,
    candidate: CandidateRunner,
    champion: ChampionRunner,
    synthetic: bool = True,
) -> ComparisonData:
    """Score candidate against champion using call-level booking outcomes.

    Args:
        settings: Scorable-status and ignored-field configuration.
        results: Digest-only results from background shadow runs.
        outcome_source: Injectable source queried only with ``call_id``.
        mirror_stats: Call-path counters used as isolation evidence.
        candidate: Candidate runner supplying artifact metadata.
        champion: Champion runner supplying artifact metadata.
        synthetic: Whether all inputs are fixture stand-ins (default True).

    Returns:
        Aggregate metrics, candidate-minus-champion deltas and sample counts.

    """
    candidate_latest = _latest_by_call(results, ModelSource.CANDIDATE)
    champion_latest = _latest_by_call(results, ModelSource.CHAMPION)
    all_calls = set(candidate_latest) | set(champion_latest)
    scorable = settings.scorable_statuses()

    calls_with_outcome = 0
    calls_excluded_by_status = 0
    calls_empty_final_record = 0
    calls_missing_model_result = 0
    scored_calls: list[ScoredCall] = []

    for call_id in sorted(all_calls):
        outcome = outcome_source.for_call(call_id)
        if outcome is None:
            continue
        calls_with_outcome += 1
        if outcome.status not in scorable:
            calls_excluded_by_status += 1
            continue
        if not outcome.field_digests:
            calls_empty_final_record += 1
            continue
        candidate_result = candidate_latest.get(call_id)
        champion_result = champion_latest.get(call_id)
        if candidate_result is None or champion_result is None:
            calls_missing_model_result += 1
            continue

        reference = outcome.field_digests
        candidate_prediction = candidate_result.field_digests
        champion_prediction = champion_result.field_digests
        fields_total = len(set(reference) | set(candidate_prediction) | set(champion_prediction))
        candidate_matched = sum(
            1
            for name in set(reference) | set(candidate_prediction)
            if reference.get(name) == candidate_prediction.get(name)
        )
        champion_matched = sum(
            1
            for name in set(reference) | set(champion_prediction)
            if reference.get(name) == champion_prediction.get(name)
        )
        scored_calls.append(
            ScoredCall(
                call_id=call_id,
                candidate_correct=candidate_prediction == reference,
                champion_correct=champion_prediction == reference,
                candidate_matched=candidate_matched,
                champion_matched=champion_matched,
                fields_total=fields_total,
            )
        )

    scored = len(scored_calls)
    candidate_correct = sum(1 for call in scored_calls if call.candidate_correct)
    champion_correct = sum(1 for call in scored_calls if call.champion_correct)
    candidate_fields = sum(call.candidate_matched for call in scored_calls)
    champion_fields = sum(call.champion_matched for call in scored_calls)
    fields_total = sum(call.fields_total for call in scored_calls)

    candidate_accuracy = _rate(candidate_correct, scored)
    champion_accuracy = _rate(champion_correct, scored)
    candidate_field_accuracy = _rate(candidate_fields, fields_total)
    champion_field_accuracy = _rate(champion_fields, fields_total)
    # With no scorable calls, both error rates are 0.0 (no evidence), not 1.0.
    candidate_error_rate = 1.0 - candidate_accuracy if scored else 0.0
    champion_error_rate = 1.0 - champion_accuracy if scored else 0.0

    candidate_metrics = {
        "outcome_call_accuracy": candidate_accuracy,
        "outcome_error_rate": candidate_error_rate,
        "outcome_field_accuracy": candidate_field_accuracy,
    }
    champion_metrics = {
        "outcome_call_accuracy": champion_accuracy,
        "outcome_error_rate": champion_error_rate,
        "outcome_field_accuracy": champion_field_accuracy,
    }
    deltas = {name: candidate_metrics[name] - champion_metrics[name] for name in candidate_metrics}

    sample_counts = {
        "turns_observed": mirror_stats.observed,
        "turns_mirrored": mirror_stats.accepted,
        "turns_dropped": mirror_stats.dropped,
        "sink_errors": mirror_stats.sink_errors,
        "results_candidate": sum(1 for result in results if result.source is ModelSource.CANDIDATE),
        "results_champion": sum(1 for result in results if result.source is ModelSource.CHAMPION),
        "failed_results": sum(1 for result in results if result.status is ResultStatus.FAILED),
        "calls_observed": len(all_calls),
        "calls_with_outcome": calls_with_outcome,
        "calls_scored": scored,
        "calls_missing_outcome": len(all_calls) - calls_with_outcome,
        "calls_excluded_by_status": calls_excluded_by_status,
        "calls_empty_final_record": calls_empty_final_record,
        "calls_missing_model_result": calls_missing_model_result,
    }

    return ComparisonData(
        settings=settings,
        mirror_stats=mirror_stats,
        results=tuple(results),
        sample_counts=sample_counts,
        metrics={"candidate": candidate_metrics, "champion": champion_metrics},
        deltas=deltas,
        scored_calls=tuple(scored_calls),
        candidate_artifact_uri=candidate.artifact_uri,
        champion_artifact_uri=champion.artifact_uri,
        candidate_model_id=candidate.model_id,
        champion_model_id=champion.model_id,
        synthetic=synthetic,
    )
