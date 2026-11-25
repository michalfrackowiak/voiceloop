"""Assemble, validate and write the internal P2-04 review metrics report.

The report is an internal artifact, not a versioned cross-service contract. It
contains counts, rates, pseudonymous IDs and fixed explanatory strings only:
no hypothesis, corrected text, slot values, audio URI, credential or free-form
reviewer note can reach it. Its only intended consumer is the later P4-03
go/no-go discussion; it makes no UI, promotion or production decision.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from services.review_adapter.interface import ROUTING_LABELS
from voiceloop_common import ValidationError

from .agreement import (
    GOLDEN_CANDIDATE_SOURCE,
    NOT_HUMAN_RELIABILITY,
    SCRIPTED_GOLDEN_REVIEWERS,
    AgreementMetrics,
    GoldenCandidateSource,
    compute_inter_reviewer_agreement,
)
from .anchoring import AnchoringMetrics, compare_prefill_anchoring
from .edit_distance import compute_edit_distances, summarize_edit_distances
from .events import (
    CandidatePairSource,
    Clock,
    ReportSink,
    ReviewResultLike,
    ReviewResultSource,
    resolve_path,
    rfc3339,
    sha256_directory,
    sha256_file,
)
from .settings import ReviewMetricsSettings
from .timing import compute_item_times, summarize_times

ARTIFACT_TYPE = "voiceloop.review.metrics"
REPORT_SCHEMA_VERSION = "internal-1.0.0"
STATUS = "SYNTHETIC/FIXTURE STAND-IN - NOT A PRODUCTION REVIEW"
PURPOSE = "input_to_p4_03_go_no_go_only"
CUSTOM_UI_DECISION = "not_made"
PROMOTION = "not_evaluated"
ALIAS_THRESHOLD_CHANGE = "none"
PRODUCTION_READINESS = "not_claimed"
REVIEW_SOURCE = "fixture_review_result_source"
CLOCK_SOURCE = "injected_fake_clock"

LIMITATIONS: tuple[str, ...] = (
    "synthetic_fake_reviews_only_not_human_labels",
    "no_clickstream_or_ui_timing_measured",
    "prefill_agreement_is_descriptive_not_causal",
    "golden_candidates_are_sealed_synthetic_stand_in_only",
    "no_promotion_or_alias_or_threshold_change",
    "no_production_readiness_claim",
)

SAFE_MESSAGES: frozenset[str] = frozenset(
    {
        ARTIFACT_TYPE,
        REPORT_SCHEMA_VERSION,
        STATUS,
        PURPOSE,
        CUSTOM_UI_DECISION,
        PROMOTION,
        ALIAS_THRESHOLD_CHANGE,
        PRODUCTION_READINESS,
        REVIEW_SOURCE,
        CLOCK_SOURCE,
        GOLDEN_CANDIDATE_SOURCE,
        NOT_HUMAN_RELIABILITY,
        SCRIPTED_GOLDEN_REVIEWERS,
        "simulated_synthetic_labels",
        "descriptive_only_no_causal_claim",
        *ROUTING_LABELS,
        *LIMITATIONS,
    }
)
"""The only free-form strings the report writer is allowed to emit."""

ID_KEYS: frozenset[str] = frozenset(
    {
        "event_id",
        "call_id",
        "reviewer_id",
        "generated_at",
        "fixtures_path",
        "golden_artifact_dir",
        "golden_version",
        "fixture_sha256",
        "golden_content_hash",
    }
)
"""String fields constrained to pseudonymous IDs, paths or hex digests."""

ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*\Z")

ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "artifact_type",
        "report_schema_version",
        "generated_at",
        "status",
        "scope",
        "purpose",
        "custom_ui_decision",
        "promotion",
        "alias_threshold_change",
        "production_readiness",
        "provenance",
        "fixtures_path",
        "fixture_sha256",
        "fixture_hash_unchanged",
        "golden_artifact_dir",
        "golden_version",
        "golden_content_hash",
        "golden_seal_ok",
        "golden_files_hash_unchanged",
        "golden_synthetic",
        "golden_workflow_stand_in",
        "golden_review_mode",
        "golden_used_for_training_or_tuning",
        "real_reviewers_used",
        "audio_bytes_in_report",
        "network_used",
        "settings",
        "review_source",
        "clock_source",
        "scripted_duration_seconds",
        "sections",
        "per_item",
        "item_times",
        "time_summary",
        "edit_distances",
        "edit_distance_summary",
        "routing_label_distribution",
        "prefill_anchoring",
        "golden_inter_reviewer",
        "event_id",
        "call_id",
        "turn_index",
        "reviewer_id",
        "blind",
        "seconds",
        "count",
        "total_seconds",
        "mean_seconds",
        "median_seconds",
        "min_seconds",
        "max_seconds",
        "distance",
        "hypothesis_length",
        "corrected_length",
        "total_distance",
        "mean_distance",
        "median_distance",
        "min_distance",
        "max_distance",
        "paired_items",
        "unpaired_events",
        "transcript_agreement_count",
        "slots_agreement_count",
        "both_agreement_count",
        "transcript_agreement_rate",
        "slots_agreement_rate",
        "both_agreement_rate",
        "interpretation",
        "candidates",
        "sealed_agreement_count",
        "reviewers",
        "source",
        "limitations",
        *ROUTING_LABELS,
    }
)


@dataclass(frozen=True)
class MetricsInputs:
    """All injectable seams needed to generate one fixture-only report."""

    settings: ReviewMetricsSettings
    results: ReviewResultSource
    pairs: CandidatePairSource
    golden: GoldenCandidateSource
    clock: Clock
    sink: ReportSink


def label_distribution(results: Iterable[ReviewResultLike]) -> dict[str, int]:
    """Count routing labels with zero entries for labels absent from fixtures.

    Args:
        results: Committed fake review results.

    Returns:
        Counts keyed by the four OpenAPI 1.0.0 routing labels.

    Raises:
        ValidationError: If a result carries an unknown routing label.

    """
    counts = {label: 0 for label in ROUTING_LABELS}
    for result in results:
        if result.routing not in counts:
            raise ValidationError(f"unknown routing label: {result.routing!r}")
        counts[result.routing] += 1
    return counts


def validate_report_safety(report: Mapping[str, Any]) -> None:
    """Fail closed if a report contains unknown keys or unsafe string values.

    Every key must be in :data:`ALLOWED_KEYS`. Values for :data:`ID_KEYS` must be
    pseudonymous IDs, paths or hex digests; every other string must be one of the
    fixed messages in :data:`SAFE_MESSAGES`. This prevents hypothesis text,
    corrected transcripts, slot values, names, emails, phone numbers, card
    digits or credentials from entering the report through an injected source.

    Args:
        report: Candidate report object.

    Raises:
        ValidationError: If shape or string values are not report-safe.

    """

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if key not in ALLOWED_KEYS:
                    raise ValidationError(f"unsafe report key: {key!r}")
                if isinstance(value, str):
                    if key in ID_KEYS:
                        if ID_PATTERN.fullmatch(value) is None:
                            raise ValidationError(f"unsafe identifier in report field {key!r}")
                    elif value not in SAFE_MESSAGES:
                        raise ValidationError(f"unsafe free text in report field {key!r}")
                elif isinstance(value, (list, tuple, Mapping)):
                    walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)

    walk(report)


def build_metrics_report(data: MetricsInputs) -> dict[str, Any]:
    """Compute all metrics and assemble the internal review metrics report.

    Args:
        data: Injectable settings, sources, fake clock and report sink.

    Returns:
        A JSON-serialisable report validated by :func:`validate_report_safety`.

    Raises:
        ValidationError: If input hashes change during the run, or if any
            computed value or string is not report-safe.

    """
    settings = data.settings
    fixture_path = resolve_path(settings.fixtures_path)
    golden_dir = resolve_path(settings.golden_artifact_dir)
    fixture_before = sha256_file(fixture_path)
    golden_before = sha256_directory(golden_dir)

    results = tuple(data.results.iter_results())
    times = compute_item_times(results)
    distances = compute_edit_distances(results)
    labels = label_distribution(results)
    anchoring: AnchoringMetrics = compare_prefill_anchoring(data.pairs, results)
    agreement: AgreementMetrics = compute_inter_reviewer_agreement(data.golden)

    fixture_after = sha256_file(fixture_path)
    golden_after = sha256_directory(golden_dir)
    if fixture_before != fixture_after or golden_before != golden_after:
        raise ValidationError("read-only fixture or golden artifacts changed during the run")

    report: dict[str, Any] = {
        "artifact_type": ARTIFACT_TYPE,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": rfc3339(data.clock()),
        "status": STATUS,
        "scope": {
            "purpose": PURPOSE,
            "custom_ui_decision": CUSTOM_UI_DECISION,
            "promotion": PROMOTION,
            "alias_threshold_change": ALIAS_THRESHOLD_CHANGE,
            "production_readiness": PRODUCTION_READINESS,
        },
        "provenance": {
            "fixtures_path": settings.fixtures_path,
            "fixture_sha256": fixture_before,
            "fixture_hash_unchanged": fixture_before == fixture_after,
            "golden_artifact_dir": settings.golden_artifact_dir,
            "golden_version": settings.golden_version,
            "golden_content_hash": settings.expected_golden_content_hash,
            "golden_seal_ok": True,
            "golden_files_hash_unchanged": golden_before == golden_after,
            "golden_synthetic": True,
            "golden_workflow_stand_in": True,
            "golden_review_mode": "simulated_synthetic_labels",
            "golden_used_for_training_or_tuning": False,
            "real_reviewers_used": False,
            "audio_bytes_in_report": False,
            "network_used": False,
        },
        "settings": {
            "review_source": REVIEW_SOURCE,
            "clock_source": CLOCK_SOURCE,
            "scripted_duration_seconds": settings.scripted_duration_seconds,
        },
        "sections": {
            "per_item": {
                "item_times": [
                    {
                        "event_id": item.event_id,
                        "call_id": item.call_id,
                        "turn_index": item.turn_index,
                        "reviewer_id": item.reviewer_id,
                        "blind": item.blind,
                        "seconds": item.seconds,
                    }
                    for item in times
                ],
                "time_summary": summarize_times(times).to_dict(),
                "edit_distances": [
                    {
                        "event_id": item.event_id,
                        "call_id": item.call_id,
                        "turn_index": item.turn_index,
                        "reviewer_id": item.reviewer_id,
                        "blind": item.blind,
                        "distance": item.distance,
                        "hypothesis_length": item.hypothesis_length,
                        "corrected_length": item.corrected_length,
                    }
                    for item in distances
                ],
                "edit_distance_summary": summarize_edit_distances(distances).to_dict(),
                "routing_label_distribution": labels,
            },
            "prefill_anchoring": anchoring.to_dict(),
            "golden_inter_reviewer": agreement.to_dict(),
        },
        "limitations": list(LIMITATIONS),
    }
    validate_report_safety(report)
    return report


def write_metrics_report(report: Mapping[str, Any], sink: ReportSink) -> str:
    """Validate and hand ``report`` to an injectable sink.

    Args:
        report: Report from :func:`build_metrics_report`.
        sink: Destination implementing :class:`~services.review_adapter.metrics.events.ReportSink`.

    Returns:
        The sink's destination label.

    """
    validate_report_safety(report)
    return sink.write(report)


def generate_metrics_report(data: MetricsInputs) -> tuple[dict[str, Any], str]:
    """Build and write one fixture-only report.

    Args:
        data: Injectable settings, sources, fake clock and sink.

    Returns:
        The report object and the sink's destination label.

    """
    report = build_metrics_report(data)
    destination = write_metrics_report(report, data.sink)
    return report, destination
