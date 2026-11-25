"""Tests for background-only canary orchestration and the done-when rehearsal."""

from __future__ import annotations

from typing import Any

from services.evalgate.gate.aliases import CHAMPION_ALIAS, InMemoryAliasStore
from services.evalgate.gate.audit import PromotionRecord
from services.release.canary import (
    SYNTHETIC_LABEL,
    CanaryCallObservation,
    CanaryRunner,
    CanarySettings,
    CollectingCanaryAudit,
    FakeEventSource,
    InMemoryRollbackAudit,
    run_fixture_rehearsal,
)
from services.release.shadow.mirror import FakeClock
from services.release.shadow.scoring import FakeOutcomeSource, OutcomeRecord

IGNORED = frozenset({"call_id", "status", "updated_at"})
GOOD = {"guests": 2, "time": "19:00"}
BAD = {"guests": -1, "time": "19:00"}
CANARY_IDS = ("call-0002", "call-0004", "call-0005")


def make_outcome(call_id: str, fields: dict[str, Any]) -> OutcomeRecord:
    """Build one final-record outcome for the given call."""
    return OutcomeRecord.from_payload(
        call_id,
        {"status": "confirmed", "final_record": fields, "updated_at": ""},
        IGNORED,
        True,
    )


def wire(
    settings: CanarySettings,
    clock: FakeClock,
    promotion: PromotionRecord,
    alias_store: InMemoryAliasStore,
    audit: CollectingCanaryAudit,
    rollback_audit: InMemoryRollbackAudit,
    call_ids: tuple[str, ...] = CANARY_IDS,
    *,
    canary_fields: dict[str, Any] = BAD,
    champion_fields: dict[str, Any] = GOOD,
    with_outcomes: bool = True,
    max_observations: int | None = None,
) -> CanaryRunner:
    """Wire one deterministic runner with injected fakes."""
    selected = call_ids[:max_observations] if max_observations else call_ids
    observations = [
        CanaryCallObservation(
            call_id=call_id,
            canary_fields=canary_fields,
            champion_fields=champion_fields,
            observed_at=clock.now(),
        )
        for call_id in selected
    ]
    outcomes = (
        {call_id: make_outcome(call_id, champion_fields) for call_id in selected}
        if with_outcomes
        else {}
    )
    return CanaryRunner(
        settings=settings,
        promotion=promotion,
        event_source=FakeEventSource(observations),
        outcome_source=FakeOutcomeSource(outcomes),
        alias_store=alias_store,
        audit=audit,
        clock=clock,
        rollback_audit=rollback_audit,
    )


def test_simulated_regression_triggers_bounded_rollback(
    settings: CanarySettings,
    clock: FakeClock,
    promotion: PromotionRecord,
    alias_store: InMemoryAliasStore,
    canary_audit: CollectingCanaryAudit,
    rollback_audit: InMemoryRollbackAudit,
) -> None:
    """Done-when: a simulated regression applies one audited alias rollback."""
    runner = wire(settings, clock, promotion, alias_store, canary_audit, rollback_audit)
    result = runner.tick()

    assert result.metrics.sample_counts["calls_scored"] == 3
    assert result.verdict.reason == "outcome_metric_regression"
    assert result.rollback.status == "applied"
    assert result.rollback.writes == 1
    assert alias_store.get(CHAMPION_ALIAS) == "v1"
    assert len(rollback_audit.entries()) == 1
    assert len(canary_audit.events) == 1
    assert canary_audit.events[0].status == "applied"
    assert canary_audit.events[0].writes == 1


def test_no_regression_does_not_flip_alias(
    settings: CanarySettings,
    clock: FakeClock,
    promotion: PromotionRecord,
    alias_store: InMemoryAliasStore,
    canary_audit: CollectingCanaryAudit,
    rollback_audit: InMemoryRollbackAudit,
) -> None:
    """Equal predictions produce no rollback and no alias write."""
    runner = wire(
        settings,
        clock,
        promotion,
        alias_store,
        canary_audit,
        rollback_audit,
        canary_fields=GOOD,
    )
    result = runner.tick()
    assert result.verdict.rollback is False
    assert result.rollback.status == "not_needed"
    assert result.rollback.writes == 0
    assert alias_store.get(CHAMPION_ALIAS) == "v2"
    assert rollback_audit.entries() == ()


def test_missing_outcomes_fail_closed(
    settings: CanarySettings,
    clock: FakeClock,
    promotion: PromotionRecord,
    alias_store: InMemoryAliasStore,
    canary_audit: CollectingCanaryAudit,
    rollback_audit: InMemoryRollbackAudit,
) -> None:
    """Missing final records trigger the bounded rollback rather than a pass."""
    runner = wire(
        settings,
        clock,
        promotion,
        alias_store,
        canary_audit,
        rollback_audit,
        with_outcomes=False,
    )
    result = runner.tick()
    assert result.verdict.reason == "incomplete_outcome_evidence"
    assert result.rollback.status == "applied"
    assert result.rollback.writes == 1


def test_window_expiry_rolls_back_without_decision(
    settings: CanarySettings,
    clock: FakeClock,
    promotion: PromotionRecord,
    alias_store: InMemoryAliasStore,
    canary_audit: CollectingCanaryAudit,
    rollback_audit: InMemoryRollbackAudit,
) -> None:
    """At the window boundary, insufficient evidence still fails closed."""
    runner = wire(
        settings,
        clock,
        promotion,
        alias_store,
        canary_audit,
        rollback_audit,
        canary_fields=GOOD,
        max_observations=1,
    )
    first = runner.tick()
    assert first.verdict.evidence_status == "insufficient"
    assert first.rollback.status == "not_needed"
    clock.advance(settings.window_seconds)
    result = runner.tick()
    assert result.verdict.evidence_status == "insufficient"
    assert result.rollback.status == "applied"
    assert result.rollback.writes == 1
    assert result.observed_at == result.window_deadline


def test_before_window_boundary_insufficient_evidence_waits(
    settings: CanarySettings,
    clock: FakeClock,
    promotion: PromotionRecord,
    alias_store: InMemoryAliasStore,
    canary_audit: CollectingCanaryAudit,
    rollback_audit: InMemoryRollbackAudit,
) -> None:
    """Insufficient paired evidence before the deadline does not yet flip."""
    runner = wire(
        settings,
        clock,
        promotion,
        alias_store,
        canary_audit,
        rollback_audit,
        canary_fields=GOOD,
        max_observations=1,
    )
    result = runner.tick()
    assert result.verdict.evidence_status == "insufficient"
    assert result.rollback.status == "not_needed"
    assert alias_store.get(CHAMPION_ALIAS) == "v2"


def test_calls_per_tick_is_bounded(
    settings: CanarySettings,
    clock: FakeClock,
    promotion: PromotionRecord,
    alias_store: InMemoryAliasStore,
    canary_audit: CollectingCanaryAudit,
    rollback_audit: InMemoryRollbackAudit,
) -> None:
    """The runner never consumes more than the configured batch size."""
    ids = tuple(f"call-{index:04d}" for index in range(1, 41))
    runner = wire(settings, clock, promotion, alias_store, canary_audit, rollback_audit, ids)
    result = runner.tick()
    assert result.calls_observed == settings.max_calls_per_tick
    assert result.calls_canary + result.calls_control == settings.max_calls_per_tick


def test_second_tick_is_idempotent_after_rollback(
    settings: CanarySettings,
    clock: FakeClock,
    promotion: PromotionRecord,
    alias_store: InMemoryAliasStore,
    canary_audit: CollectingCanaryAudit,
    rollback_audit: InMemoryRollbackAudit,
) -> None:
    """A later tick after an applied rollback issues no further alias write."""
    runner = wire(settings, clock, promotion, alias_store, canary_audit, rollback_audit)
    first = runner.tick()
    second = runner.tick()
    assert first.rollback.status == "applied"
    assert second.rollback.status == "idempotent"
    assert second.rollback.writes == 0
    assert len(canary_audit.events) == 2
    assert len(rollback_audit.entries()) == 1


def test_report_is_labelled_and_makes_no_promotion_claim(
    settings: CanarySettings,
    clock: FakeClock,
    promotion: PromotionRecord,
    alias_store: InMemoryAliasStore,
    canary_audit: CollectingCanaryAudit,
    rollback_audit: InMemoryRollbackAudit,
) -> None:
    """Reports carry the synthetic label and no promotion/champion claim."""
    report = (
        wire(settings, clock, promotion, alias_store, canary_audit, rollback_audit)
        .tick()
        .to_report()
    )
    assert report["evidence_label"] == SYNTHETIC_LABEL
    assert report["synthetic"] is True
    assert report["real_traffic"] is False
    assert report["promotion_decision"] == "not_evaluated"
    assert report["champion_claim"] is False
    assert report["gate_edits"] == 0
    assert report["thresholds_changed"] is False
    assert report["alias_writes"] == 1
    assert "wer_used" not in report["metrics"]
    assert "wer" not in str(report["metrics"]).lower()
    assert "SYNTHETIC" in report["evidence_label"]


def test_fixture_rehearsal_applies_bounded_rollback() -> None:
    """The one-command fixture rehearsal demonstrates the done-when condition."""
    result = run_fixture_rehearsal()
    assert result.rollback.status == "applied"
    assert result.rollback.writes == 1
    assert result.rollback.record is not None
    assert result.evidence_label == SYNTHETIC_LABEL
    assert result.metrics.sample_counts["calls_canary"] >= 2
    assert result.metrics.sample_counts["calls_scored"] >= 2
    assert result.promotion_decision == "not_evaluated"
    report = result.to_report()
    assert report["real_traffic"] is False
    assert report["gate_edits"] == 0


def test_fixture_rehearsal_is_deterministic() -> None:
    """Two rehearsal runs produce identical digests, counts and rollback status."""
    first = run_fixture_rehearsal().to_report()
    second = run_fixture_rehearsal().to_report()
    assert first == second
