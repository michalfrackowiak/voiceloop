"""Tests for bounded, idempotent, audited one-step rollback."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from services.evalgate.gate.aliases import CHAMPION_ALIAS, InMemoryAliasStore
from services.evalgate.gate.audit import PromotionRecord
from services.release.canary import (
    CanaryDeployment,
    CanarySettings,
    CollectingRollbackAudit,
    InMemoryRollbackAudit,
    RegressionVerdict,
    RollbackAuditLog,
    RollbackController,
    RollbackRecord,
    canonical_json,
    sha256_hex,
)
from services.release.sync.registry import ReadOnlyAliasStore
from voiceloop_common import ValidationError


class FixedRegistry:
    """Read-only alias view that can be made inconsistent with the writer."""

    def __init__(self, value: str | None) -> None:
        """Store the version that this view reports."""
        self.value = value

    def get(self, alias: str) -> str | None:
        """Return the configured version for the requested alias."""
        return self.value


class FailingAliasStore:
    """Alias store that raises on read, write, or both."""

    def __init__(
        self, value: str | None, *, fail_read: bool = False, fail_write: bool = False
    ) -> None:
        """Configure injected failures for controller tests."""
        self.value = value
        self.fail_read = fail_read
        self.fail_write = fail_write
        self.writes: list[tuple[str, str]] = []

    def get(self, alias: str) -> str | None:
        """Return the stored version or raise the injected read error."""
        if self.fail_read:
            raise RuntimeError("boom-read")
        return self.value

    def set(self, alias: str, version: str) -> None:
        """Raise on injected write failure, otherwise record one flip."""
        if self.fail_write:
            raise RuntimeError("boom-write")
        self.writes.append((alias, version))
        self.value = version


class FailingAudit:
    """Audit sink that fails on the first append."""

    def __init__(self) -> None:
        """Initialise with no entries."""
        self.records: list[RollbackRecord] = []

    def head_hash(self) -> str:
        """Return the empty chain head."""
        return ""

    def entries(self) -> tuple[RollbackRecord, ...]:
        """Return all appended records."""
        return tuple(self.records)

    def record(self, record: RollbackRecord) -> RollbackRecord:
        """Raise to simulate an unavailable audit backend."""
        raise RuntimeError("boom-audit")


def controller(
    settings: CanarySettings,
    store: Any,
    audit: Any,
    registry: ReadOnlyAliasStore | None = None,
) -> RollbackController:
    """Wire a controller with injected fakes."""
    return RollbackController(settings, store, audit, registry)


def verdict(rollback: bool = True, reason: str = "outcome_metric_regression") -> RegressionVerdict:
    """Build a regression verdict for controller tests."""
    return RegressionVerdict(
        rollback=rollback,
        reason=reason,
        evidence_status="complete",
        sample_counts={"calls_scored": 3},
        evidence_label="SYNTHETIC/FIXTURE STAND-IN - NOT LIVE CANARY EVIDENCE",
    )


def test_applied_rollback_writes_once_and_audits(
    settings: CanarySettings,
    deployment: CanaryDeployment,
    alias_store: InMemoryAliasStore,
    rollback_audit: InMemoryRollbackAudit,
    start: datetime,
) -> None:
    """A regression during the window flips the alias once and audits it."""
    ctl = controller(settings, alias_store, rollback_audit)
    result = ctl.rollback(verdict(), deployment, start)

    assert result.status == "applied"
    assert result.writes == 1
    assert alias_store.get(CHAMPION_ALIAS) == "v1"
    assert result.current_version == "v2"
    assert result.target_version == "v1"
    assert result.evidence_label == settings.evidence_label
    assert len(rollback_audit.entries()) == 1
    assert rollback_audit.verify_chain() == ()
    assert rollback_audit.entries()[0].previous_version == "v1"
    assert rollback_audit.entries()[0].canary_version == "v2"
    assert rollback_audit.entries()[0].task_id == "P4-01"
    assert rollback_audit.entries()[0].actor_id == settings.rollback_actor_id


def test_rollback_is_idempotent(
    settings: CanarySettings,
    deployment: CanaryDeployment,
    alias_store: InMemoryAliasStore,
    rollback_audit: InMemoryRollbackAudit,
    start: datetime,
) -> None:
    """Repeating the same rollback makes no additional write or audit entry."""
    ctl = controller(settings, alias_store, rollback_audit)
    first = ctl.rollback(verdict(), deployment, start)
    second = ctl.rollback(verdict(), deployment, start)

    assert first.status == "applied"
    assert second.status == "idempotent"
    assert second.writes == 0
    assert len(rollback_audit.entries()) == 1


def test_alias_already_at_target_is_idempotent(
    settings: CanarySettings,
    deployment: CanaryDeployment,
    rollback_audit: InMemoryRollbackAudit,
    start: datetime,
) -> None:
    """If the alias already points at the previous version, nothing is written."""
    store = InMemoryAliasStore({CHAMPION_ALIAS: "v1"})
    result = controller(settings, store, rollback_audit).rollback(verdict(), deployment, start)
    assert result.status == "idempotent"
    assert result.writes == 0
    assert rollback_audit.entries() == ()


def test_outside_window_refuses_rollback(
    settings: CanarySettings,
    deployment: CanaryDeployment,
    alias_store: InMemoryAliasStore,
    rollback_audit: InMemoryRollbackAudit,
    start: datetime,
) -> None:
    """An observation after the configured deadline performs no write."""
    too_late = start + timedelta(seconds=settings.window_seconds + 1)
    result = controller(settings, alias_store, rollback_audit).rollback(
        verdict(), deployment, too_late
    )
    assert result.status == "outside_window"
    assert result.writes == 0
    assert alias_store.get(CHAMPION_ALIAS) == "v2"
    assert rollback_audit.entries() == ()


def test_boundary_instant_is_allowed(
    settings: CanarySettings,
    deployment: CanaryDeployment,
    alias_store: InMemoryAliasStore,
    rollback_audit: InMemoryRollbackAudit,
    start: datetime,
) -> None:
    """The exact deadline instant is still inside the configured window."""
    at_deadline = start + timedelta(seconds=settings.window_seconds)
    result = controller(settings, alias_store, rollback_audit).rollback(
        verdict(), deployment, at_deadline
    )
    assert result.status == "applied"
    assert result.writes == 1


def test_unbound_and_stale_alias_refuse_rollback(
    settings: CanarySettings,
    deployment: CanaryDeployment,
    rollback_audit: InMemoryRollbackAudit,
    start: datetime,
) -> None:
    """Unbound or unexpected alias versions stop the single flip."""
    unbound = controller(settings, InMemoryAliasStore({}), rollback_audit).rollback(
        verdict(), deployment, start
    )
    assert unbound.status == "failed"
    assert unbound.reason == "alias_unbound"
    assert unbound.writes == 0

    stale = controller(
        settings, InMemoryAliasStore({CHAMPION_ALIAS: "v9"}), rollback_audit
    ).rollback(verdict(), deployment, start)
    assert stale.status == "stale_alias_version"
    assert stale.writes == 0


def test_registry_mismatch_refuses_rollback(
    settings: CanarySettings,
    deployment: CanaryDeployment,
    alias_store: InMemoryAliasStore,
    rollback_audit: InMemoryRollbackAudit,
    start: datetime,
) -> None:
    """A disagreeing read-only registry view prevents an audited flip."""
    mismatched: ReadOnlyAliasStore = FixedRegistry("v3")
    result = controller(settings, alias_store, rollback_audit, mismatched).rollback(
        verdict(), deployment, start
    )
    assert result.status == "failed"
    assert result.reason == "registry_alias_mismatch"
    assert result.writes == 0
    assert alias_store.get(CHAMPION_ALIAS) == "v2"


def test_alias_read_and_write_failures_are_reported(
    settings: CanarySettings,
    deployment: CanaryDeployment,
    rollback_audit: InMemoryRollbackAudit,
    start: datetime,
) -> None:
    """Injected alias source errors never raise into the caller."""
    read_error = controller(
        settings,
        FailingAliasStore("v2", fail_read=True),
        rollback_audit,
    ).rollback(verdict(), deployment, start)
    assert read_error.status == "failed"
    assert read_error.reason == "alias_read_failed"
    assert read_error.error_type == "RuntimeError"
    assert read_error.writes == 0

    write_store = FailingAliasStore("v2", fail_write=True)
    write_error = controller(settings, write_store, rollback_audit).rollback(
        verdict(), deployment, start
    )
    assert write_error.status == "failed"
    assert write_error.reason == "alias_write_failed"
    assert write_error.writes == 0
    assert write_store.writes == []


def test_audit_failure_restores_alias_without_an_entry(
    settings: CanarySettings,
    deployment: CanaryDeployment,
    start: datetime,
) -> None:
    """An audit failure compensates the flip and reports typed failure."""
    store = FailingAliasStore("v2")
    result = controller(settings, store, FailingAudit()).rollback(verdict(), deployment, start)
    assert result.status == "failed"
    assert result.reason == "audit_write_failed"
    assert result.writes == 2
    assert store.value == "v2"


def test_no_rollback_when_verdict_says_no(
    settings: CanarySettings,
    deployment: CanaryDeployment,
    alias_store: InMemoryAliasStore,
    rollback_audit: InMemoryRollbackAudit,
    start: datetime,
) -> None:
    """A non-regression verdict produces no alias write."""
    result = controller(settings, alias_store, rollback_audit).rollback(
        verdict(rollback=False, reason="no_regression_observed"), deployment, start
    )
    assert result.status == "not_needed"
    assert result.writes == 0
    assert alias_store.get(CHAMPION_ALIAS) == "v2"


def test_chain_hash_helpers_are_stable() -> None:
    """Canonical JSON and SHA-256 helpers produce stable digests."""
    payload = {"b": 2, "a": 1}
    encoded = canonical_json(payload)
    assert encoded == '{"a":1,"b":2}'
    assert len(sha256_hex(encoded)) == 64


def test_rollback_audit_log_persists_jsonl(
    settings: CanarySettings,
    deployment: CanaryDeployment,
    start: datetime,
    tmp_path: Path,
) -> None:
    """The file-backed audit appends canonical JSONL and loads it back."""
    audit = RollbackAuditLog(tmp_path / "audit.jsonl")
    store = InMemoryAliasStore({CHAMPION_ALIAS: "v2"})
    result = controller(settings, store, audit).rollback(verdict(), deployment, start)
    assert result.status == "applied"
    loaded = audit.entries()
    assert len(loaded) == 1
    assert result.record is not None
    assert loaded[0].entry_hash == result.record.entry_hash
    assert audit.head_hash() == loaded[0].entry_hash
    assert len(audit.path.read_text(encoding="utf-8").splitlines()) == 1


def test_deployment_requires_a_promote_record(
    promotion: PromotionRecord, start: datetime, settings: CanarySettings
) -> None:
    """A rollback audit record cannot seed a canary deployment."""
    invalid = PromotionRecord(
        actor_id="fake",
        action="rollback",
        reason="not a promote",
        before_version="v2",
        after_version="v1",
        task_id="P4-01",
    )
    with pytest.raises(ValidationError):
        CanaryDeployment.from_promotion(invalid, start, settings.evidence_label)
    assert deployment_from(promotion, start, settings).previous_version == "v1"


def deployment_from(
    promotion: PromotionRecord, start: datetime, settings: CanarySettings
) -> CanaryDeployment:
    """Build a deployment from a valid promote record."""
    return CanaryDeployment.from_promotion(promotion, start, settings.evidence_label)


def test_collecting_alias_is_in_memory() -> None:
    """The collecting rollback audit alias behaves like the in-memory sink."""

    audit = CollectingRollbackAudit()
    assert isinstance(audit, InMemoryRollbackAudit)
    assert audit.entries() == ()
