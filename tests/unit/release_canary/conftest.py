"""Shared deterministic fake world for P4-01 tests (no sleeps or services)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Literal

import pytest

from services.evalgate.gate.aliases import InMemoryAliasStore
from services.evalgate.gate.audit import PromotionRecord
from services.release.canary import (
    CanaryDeployment,
    CanarySettings,
    CollectingCanaryAudit,
    InMemoryRollbackAudit,
    RegressionVerdict,
)
from services.release.shadow.mirror import FakeClock
from services.release.shadow.scoring import OutcomeRecord


@pytest.fixture
def settings() -> CanarySettings:
    """Return deterministic settings with a fixed salt and small bounds."""
    return CanarySettings(
        canary_fraction=0.20,
        split_salt="voiceloop-canary-demo-0",
        window_seconds=60.0,
        min_paired_calls=2,
        max_calls_per_tick=16,
        max_pairs=32,
    )


@pytest.fixture
def start() -> datetime:
    """Return the fixed fake-clock start instant."""
    return datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def clock(start: datetime) -> FakeClock:
    """Return a fake clock at the fixed start instant."""
    return FakeClock(start)


@pytest.fixture
def promotion() -> PromotionRecord:
    """Return a read-only fake gate promotion record."""
    return PromotionRecord(
        actor_id="fake-gate-promoter",
        action="promote",
        reason="synthetic canary rehearsal",
        before_version="v1",
        after_version="v2",
        task_id="P4-01",
    )


@pytest.fixture
def deployment(
    promotion: PromotionRecord, start: datetime, settings: CanarySettings
) -> CanaryDeployment:
    """Return the immutable deployment context derived from the promotion."""
    return CanaryDeployment.from_promotion(promotion, start, settings.evidence_label)


@pytest.fixture
def alias_store() -> InMemoryAliasStore:
    """Return an alias store currently serving the canary version."""
    return InMemoryAliasStore({"@champion": "v2"})


@pytest.fixture
def rollback_audit() -> InMemoryRollbackAudit:
    """Return an empty in-memory append-only audit sink."""
    return InMemoryRollbackAudit()


@pytest.fixture
def canary_audit() -> CollectingCanaryAudit:
    """Return an empty collecting structured audit sink."""
    return CollectingCanaryAudit()


def make_outcome(
    call_id: str,
    fields: Mapping[str, Any],
    *,
    status: str = "confirmed",
) -> OutcomeRecord:
    """Build a PII-safe outcome record from raw fields using digests."""
    return OutcomeRecord.from_payload(
        call_id,
        {"status": status, "final_record": dict(fields), "updated_at": ""},
        frozenset({"call_id", "status", "updated_at"}),
        True,
    )


def make_verdict(
    *,
    rollback: bool = True,
    reason: str = "outcome_metric_regression",
    evidence_status: Literal["complete", "insufficient", "incomplete"] = "complete",
) -> RegressionVerdict:
    """Build a verdict with explicit sample counts for controller tests."""
    return RegressionVerdict(
        rollback=rollback,
        reason=reason,
        evidence_status=evidence_status,
        sample_counts={"calls_scored": 3},
        evidence_label="SYNTHETIC/FIXTURE STAND-IN - NOT LIVE CANARY EVIDENCE",
    )
