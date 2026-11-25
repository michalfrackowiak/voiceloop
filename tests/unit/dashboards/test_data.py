"""Tests for fixture-derived failure rate, injected seams and provenance."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infra.dashboards.data import (
    UNKNOWN_PROMOTION,
    UNKNOWN_VERSION,
    FailureSnapshot,
    FakeModelStateProvider,
    FakeQueueDepthProvider,
    FakeReviewThroughputProvider,
    FixtureFailureProvider,
    JsonSnapshotProvider,
    PromotionRecord,
    UnavailableQueueDepthProvider,
    collect_dashboard,
    failure_rate,
)
from infra.dashboards.settings import DashboardSettings
from tests.unit.dashboards.conftest import metric_named
from voiceloop_common import ValidationError

FRESH = "2026-11-25T11:00:00Z"
OLD = "2026-11-23T00:00:00Z"


def test_failure_rate_comes_from_readonly_fixture_labels() -> None:
    provider = FixtureFailureProvider("tests/fixtures/turns.jsonl")
    snapshot = provider.failure_snapshot()
    assert snapshot.total_turns == 36
    assert snapshot.failed_turns == 10
    assert snapshot.synthetic is True
    assert snapshot.observed_at is None
    assert snapshot.source == "tests/fixtures/turns.jsonl"
    assert failure_rate(snapshot) == pytest.approx(10 / 36)


def test_failure_rate_handles_empty_denominator() -> None:
    snapshot = FailureSnapshot(total_turns=0, failed_turns=0, source="fake://empty")
    assert failure_rate(snapshot) is None
    with pytest.raises(ValidationError):
        FailureSnapshot(total_turns=1, failed_turns=2, source="fake://bad")


def test_fixture_provider_rejects_unlabelled_or_non_synthetic_rows(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text('{"failure_type": "clean", "synthetic": false}\n', encoding="utf-8")
    with pytest.raises(ValidationError, match="synthetic"):
        FixtureFailureProvider(str(path)).failure_snapshot()
    path.write_text('{"synthetic": true}\n', encoding="utf-8")
    with pytest.raises(ValidationError, match="failure_type"):
        FixtureFailureProvider(str(path)).failure_snapshot()
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="invalid JSON"):
        FixtureFailureProvider(str(path)).failure_snapshot()


def test_committed_example_snapshot_is_synthetic_and_has_no_model_facts() -> None:
    provider = JsonSnapshotProvider("infra/dashboards/examples/synthetic-snapshot.json")
    queue = provider.queue_snapshot()
    review = provider.review_snapshot()
    model = provider.model_snapshot()
    assert queue.depth == 3 and queue.synthetic is True
    assert review.completed == 5 and review.window_start == "2026-11-24T00:00:00Z"
    assert model.version_in_service is None
    assert model.promotion is None
    assert model.synthetic is True
    assert "SYNTHETIC" in (queue.note + review.note + model.note)


def test_snapshot_provider_validates_envelope_and_promotion(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    path.write_text('{"queue": {"depth": 1}}', encoding="utf-8")
    with pytest.raises(ValidationError, match="synthetic"):
        JsonSnapshotProvider(str(path)).queue_snapshot()
    payload = {
        "synthetic": True,
        "queue": {"depth": 1},
        "review": {"completed": 1},
        "model": {
            "version_in_service": None,
            "promotion": {"version": "v1", "promoted_at": "2026-11-25T00:00:00Z"},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError, match="source"):
        JsonSnapshotProvider(str(path)).model_snapshot()


def test_injected_queue_and_review_providers_are_used(
    dashboard_settings: DashboardSettings,
) -> None:
    data = collect_dashboard(
        settings=dashboard_settings,
        now=datetime(2026, 11, 25, 12, 0, tzinfo=UTC),
        queue_provider=FakeQueueDepthProvider(depth=9, source="fake://queue-a"),
        review_provider=FakeReviewThroughputProvider(
            completed=7,
            window_start="2026-11-25T10:00:00Z",
            window_end="2026-11-25T11:00:00Z",
        ),
    )
    assert metric_named(data, "Queue depth").value == "9"
    assert metric_named(data, "Queue depth").provenance.source == "fake://queue-a"
    assert (
        "7 completed | 2026-11-25T10:00:00Z to 2026-11-25T11:00:00Z"
        == metric_named(data, "Review throughput (completed in window)").value
    )


def test_unavailable_provider_never_reports_zero(
    dashboard_settings: DashboardSettings,
) -> None:
    data = collect_dashboard(
        settings=dashboard_settings,
        now=datetime(2026, 11, 25, 12, 0, tzinfo=UTC),
        queue_provider=UnavailableQueueDepthProvider(),
    )
    assert metric_named(data, "Queue depth").value.startswith("unavailable")
    assert "0" not in metric_named(data, "Queue depth").value


def test_staleness_is_explicit_for_fresh_old_and_missing_timestamps(
    dashboard_settings: DashboardSettings,
) -> None:
    now = datetime(2026, 11, 25, 12, 0, tzinfo=UTC)
    outcomes = []
    for observed in (FRESH, OLD, None):
        data = collect_dashboard(
            settings=dashboard_settings,
            now=now,
            queue_provider=FakeQueueDepthProvider(depth=1, observed_at=observed),
        )
        outcomes.append(metric_named(data, "Queue depth").provenance)
    assert [item.stale for item in outcomes] == [False, True, True]
    assert [item.observed_at for item in outcomes] == [FRESH, OLD, None]


def test_model_state_uses_injected_provenance_and_unknown_fallback(
    dashboard_settings: DashboardSettings,
) -> None:
    now = datetime(2026, 11, 25, 12, 0, tzinfo=UTC)
    empty = collect_dashboard(settings=dashboard_settings, now=now)
    assert metric_named(empty, "Model version in service").value == UNKNOWN_VERSION
    assert metric_named(empty, "Last promotion").value == UNKNOWN_PROMOTION
    promo_metric = metric_named(empty, "Last promotion")
    assert promo_metric.provenance.source == "infra/dashboards/examples/synthetic-snapshot.json"

    promotion = PromotionRecord(
        version="test-promoted-v2",
        promoted_at="2026-11-24T09:00:00Z",
        source="fake://release/aliases",
        actor="test-reviewer",
        audit_ref="audit-42",
    )
    data = collect_dashboard(
        settings=dashboard_settings,
        now=now,
        model_provider=FakeModelStateProvider(
            version_in_service="test-serving-v1",
            promotion=promotion,
            source="fake://release/state",
            observed_at=FRESH,
        ),
    )
    version_metric = metric_named(data, "Model version in service")
    assert version_metric.value == "test-serving-v1"
    assert version_metric.provenance.source == "fake://release/state"
    promo_metric = metric_named(data, "Last promotion")
    assert "test-promoted-v2" in promo_metric.value
    assert "2026-11-24T09:00:00Z" in promo_metric.value
    assert "fake://release/aliases" in promo_metric.provenance.source
    assert promo_metric.provenance.stale is False


def test_dashboard_data_is_json_serialisable(dashboard_settings: DashboardSettings) -> None:
    data = collect_dashboard(
        settings=dashboard_settings, now=datetime(2026, 11, 25, 12, 0, tzinfo=UTC)
    )
    payload = json.loads(json.dumps(data.to_dict()))
    assert payload["rendered_at"] == "2026-11-25T12:00:00Z"
    assert len(payload["sections"]) == 2
    assert payload["sections"][0]["metrics"][0]["provenance"]["synthetic"] is True
