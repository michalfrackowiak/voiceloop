"""P2-09.4: @challenger → @champion flip, audit and one-step rollback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from services.evalgate.gate import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    GateError,
    InMemoryAliasStore,
    PromotionAuditLog,
    PromotionRecord,
    promote,
    rollback,
    rollback_promotion,
)
from voiceloop_common import ValidationError


class FailingAuditLog:
    """Audit fake that always fails, to exercise alias restoration."""

    def record(
        self,
        *,
        actor_id: str,
        action: Any,
        reason: str,
        before_version: str,
        after_version: str,
        task_id: str = "P2-09",
    ) -> PromotionRecord:
        """Refuse to write anything."""
        raise OSError("audit unavailable")


def _store() -> InMemoryAliasStore:
    """Return a store with champion v3 and challenger v4."""
    return InMemoryAliasStore({CHAMPION_ALIAS: "v3", CHALLENGER_ALIAS: "v4"})


def test_promote_flips_champion_and_audits(tmp_path: Path) -> None:
    store = _store()
    log = PromotionAuditLog(tmp_path / "audit.jsonl")

    record = promote(
        store,
        "v4",
        actor_id="ml-engineer",
        reason="gate passed on fixtures",
        audit_log=log,
    )

    assert store.get(CHAMPION_ALIAS) == "v4"
    assert store.get(CHALLENGER_ALIAS) == "v4"
    assert record.action == "promote"
    assert (record.actor_id, record.reason) == ("ml-engineer", "gate passed on fixtures")
    assert (record.before_version, record.after_version) == ("v3", "v4")
    assert log.verify_chain() == ()
    assert len(log.entries()) == 1


def test_promote_refuses_stale_challenger(tmp_path: Path) -> None:
    store = _store()
    log = PromotionAuditLog(tmp_path / "audit.jsonl")

    with pytest.raises(GateError, match="@challenger"):
        promote(
            store,
            "v9",
            actor_id="ml-engineer",
            reason="stale version",
            audit_log=log,
        )

    assert store.get(CHAMPION_ALIAS) == "v3"
    assert log.entries() == ()


def test_promote_refuses_missing_champion(tmp_path: Path) -> None:
    store = InMemoryAliasStore({CHALLENGER_ALIAS: "v4"})
    log = PromotionAuditLog(tmp_path / "audit.jsonl")

    with pytest.raises(GateError, match="no current @champion"):
        promote(
            store,
            "v4",
            actor_id="ml-engineer",
            reason="no baseline",
            audit_log=log,
        )


def test_rollback_restores_previous_champion(tmp_path: Path) -> None:
    store = _store()
    log = PromotionAuditLog(tmp_path / "audit.jsonl")
    promotion = promote(store, "v4", actor_id="ml-engineer", reason="promote", audit_log=log)

    record = rollback_promotion(
        store, promotion, actor_id="oncall", reason="canary regression", audit_log=log
    )

    assert store.get(CHAMPION_ALIAS) == "v3"
    assert (record.action, record.before_version, record.after_version) == (
        "rollback",
        "v4",
        "v3",
    )
    assert [entry.action for entry in log.entries()] == ["promote", "rollback"]
    assert log.verify_chain() == ()


def test_rollback_checks_expected_version(tmp_path: Path) -> None:
    store = _store()
    log = PromotionAuditLog(tmp_path / "audit.jsonl")

    with pytest.raises(GateError, match="not expected version"):
        rollback(
            store,
            previous_version="v3",
            actor_id="oncall",
            reason="wrong version",
            audit_log=log,
            expected_current_version="v4",
        )
    assert store.get(CHAMPION_ALIAS) == "v3"
    assert log.entries() == ()


def test_rollback_refuses_noop_and_non_promote_record(tmp_path: Path) -> None:
    store = _store()
    log = PromotionAuditLog(tmp_path / "audit.jsonl")

    with pytest.raises(GateError, match="nothing to roll back"):
        rollback(
            store,
            previous_version="v3",
            actor_id="oncall",
            reason="noop",
            audit_log=log,
        )
    rollback_record = PromotionRecord(
        actor_id="oncall",
        action="rollback",
        reason="earlier rollback",
        before_version="v4",
        after_version="v3",
    )
    with pytest.raises(GateError, match="requires a promote record"):
        rollback_promotion(store, rollback_record, actor_id="oncall", reason="retry", audit_log=log)


def test_failed_audit_write_restores_alias(tmp_path: Path) -> None:
    store = _store()

    with pytest.raises(OSError, match="audit unavailable"):
        promote(
            store,
            "v4",
            actor_id="ml-engineer",
            reason="audit failure",
            audit_log=FailingAuditLog(),
        )

    assert store.get(CHAMPION_ALIAS) == "v3"


def test_blank_version_is_rejected() -> None:
    store = _store()

    with pytest.raises(ValidationError, match="challenger_version"):
        promote(
            store,
            " ",
            actor_id="ml-engineer",
            reason="bad version",
            audit_log=FailingAuditLog(),
        )
