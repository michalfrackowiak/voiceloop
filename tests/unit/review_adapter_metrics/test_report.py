"""Whole-report assembly, provenance and privacy validation tests."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from services.review_adapter.interface import ROUTING_LABELS
from services.review_adapter.metrics.agreement import SealedGoldenCandidateSource
from services.review_adapter.metrics.events import (
    CandidatePair,
    FixedClock,
    InMemoryReportSink,
    ResultPairSource,
    ReviewResultLike,
    SequenceReviewResultSource,
    sha256_directory,
    sha256_file,
)
from services.review_adapter.metrics.report import (
    ARTIFACT_TYPE,
    CUSTOM_UI_DECISION,
    PRODUCTION_READINESS,
    PROMOTION,
    REPORT_SCHEMA_VERSION,
    STATUS,
    MetricsInputs,
    build_metrics_report,
    generate_metrics_report,
    label_distribution,
    validate_report_safety,
)
from services.review_adapter.metrics.settings import ReviewMetricsSettings
from tests.unit.review_adapter_metrics.conftest import make_result
from voiceloop_common import ValidationError


def _inputs(metrics_settings: ReviewMetricsSettings) -> tuple[MetricsInputs, InMemoryReportSink]:
    """Build the real fixture-backed seams with an in-memory sink."""
    from services.review_adapter.metrics.events import FixtureReviewResultSource, SequenceClock

    results = FixtureReviewResultSource(
        metrics_settings,
        SequenceClock(
            metrics_settings.start_at,
            step_seconds=metrics_settings.scripted_duration_seconds,
        ),
    )
    sink = InMemoryReportSink()
    return (
        MetricsInputs(
            settings=metrics_settings,
            results=SequenceReviewResultSource(results.iter_results()),
            pairs=ResultPairSource(results),
            golden=SealedGoldenCandidateSource(metrics_settings),
            clock=FixedClock(metrics_settings.start_at),
            sink=sink,
        ),
        sink,
    )


def test_label_distribution_counts_every_contract_label() -> None:
    """All four routing labels appear, with zero counts where fixtures lack them."""
    results = tuple(
        make_result(f"event-{index}", routing=label) for index, label in enumerate(ROUTING_LABELS)
    ) + (make_result("event-4", routing="both_fine"),)

    counts = label_distribution(results)

    assert counts == {label: 2 if label == "both_fine" else 1 for label in ROUTING_LABELS}
    assert sum(counts.values()) == 5


def test_label_distribution_rejects_unknown_label() -> None:
    """A fabricated routing label fails closed."""
    invalid = cast(ReviewResultLike, SimpleNamespace(routing="made_up"))
    with pytest.raises(ValidationError, match="unknown routing label"):
        label_distribution((invalid,))


def test_report_safety_blocks_unknown_keys_and_free_text() -> None:
    """Only allowlisted keys and fixed messages can reach a report."""
    with pytest.raises(ValidationError, match="unsafe report key"):
        validate_report_safety({"corrected_transcript": "private fixture text"})

    with pytest.raises(ValidationError, match="unsafe free text"):
        validate_report_safety({"status": "Ada Fixture"})

    with pytest.raises(ValidationError, match="unsafe identifier"):
        validate_report_safety({"call_id": "ada.fixture@example.com"})

    validate_report_safety({"status": STATUS, "call_id": "call-0001"})


def test_generated_report_is_synthetic_labeled_and_pii_free(
    metrics_settings: ReviewMetricsSettings, repo_root: Path
) -> None:
    """The full report is fixture-only, hash-pinned and contains no fixture content."""
    fixture_before = sha256_file(repo_root / metrics_settings.fixtures_path)
    golden_before = sha256_directory(repo_root / metrics_settings.golden_artifact_dir)
    inputs, sink = _inputs(metrics_settings)

    report, destination = generate_metrics_report(inputs)

    fixture_after = sha256_file(repo_root / metrics_settings.fixtures_path)
    golden_after = sha256_directory(repo_root / metrics_settings.golden_artifact_dir)
    assert destination == "in-memory://review-metrics"
    assert sink.report == report
    assert fixture_before == fixture_after
    assert golden_before == golden_after
    assert report["artifact_type"] == ARTIFACT_TYPE
    assert report["report_schema_version"] == REPORT_SCHEMA_VERSION
    assert report["status"] == STATUS
    assert report["scope"]["custom_ui_decision"] == CUSTOM_UI_DECISION
    assert report["scope"]["promotion"] == PROMOTION
    assert report["scope"]["production_readiness"] == PRODUCTION_READINESS
    assert report["provenance"]["fixture_hash_unchanged"] is True
    assert report["provenance"]["golden_files_hash_unchanged"] is True
    assert report["provenance"]["golden_seal_ok"] is True
    assert report["provenance"]["real_reviewers_used"] is False
    assert report["provenance"]["network_used"] is False

    per_item = report["sections"]["per_item"]
    assert per_item["time_summary"]["count"] == 72
    assert sum(per_item["routing_label_distribution"].values()) == 72
    assert report["sections"]["prefill_anchoring"]["paired_items"] == 36
    assert report["sections"]["prefill_anchoring"]["unpaired_events"] == 0
    assert report["sections"]["golden_inter_reviewer"]["candidates"] == 36

    serialized = json.dumps(report)
    assert "ada.fixture@example.com" not in serialized
    assert "+1 555 010 0001" not in serialized
    assert "Ada Fixture" not in serialized
    assert "Hi, I'd like to book a room for December." not in serialized


def test_report_generation_is_deterministic(metrics_settings: ReviewMetricsSettings) -> None:
    """Repeated runs with the same injected seams produce identical reports."""
    first, _ = _inputs(metrics_settings)
    second, _ = _inputs(metrics_settings)

    first_report, _ = generate_metrics_report(first)
    second_report, _ = generate_metrics_report(second)

    assert first_report == second_report


class EmptyPairSource:
    """Pair source used only to isolate the input-hash guard."""

    def iter_pairs(self) -> Iterator[CandidatePair]:
        """Yield no pairs."""
        return iter(())


class MutatingResultSource:
    """Fake source that changes its temporary fixture copy while being consumed."""

    def __init__(self, path: Path, results: tuple[ReviewResultLike, ...]) -> None:
        """Bind a writable copy and the fake results to yield."""
        self._path = path
        self._results = results

    def iter_results(self) -> Iterator[ReviewResultLike]:
        """Append one byte before yielding the frozen fake results."""
        with self._path.open("ab") as handle:
            handle.write(b"\n")
        return iter(self._results)


def test_build_report_rejects_changed_inputs(
    metrics_settings: ReviewMetricsSettings, tmp_path: Path
) -> None:
    """A source that mutates a read-only input copy mid-run fails the whole report."""
    fixture_copy = tmp_path / "turns.jsonl"
    fixture_copy.write_bytes(Path(metrics_settings.fixtures_path).resolve().read_bytes())
    changed = metrics_settings.model_copy(update={"fixtures_path": str(fixture_copy)})
    inputs = MetricsInputs(
        settings=changed,
        results=MutatingResultSource(fixture_copy, (make_result(),)),
        pairs=EmptyPairSource(),
        golden=SealedGoldenCandidateSource(changed),
        clock=FixedClock(changed.start_at),
        sink=InMemoryReportSink(),
    )

    with pytest.raises(ValidationError, match="changed during the run"):
        build_metrics_report(inputs)
