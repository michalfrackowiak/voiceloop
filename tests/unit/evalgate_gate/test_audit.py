"""P2-09.4: immutable, hash-chained promotion audit records."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path

import pytest

from services.evalgate.gate import PromotionAuditLog
from voiceloop_common import ValidationError, frozen_clock, use_clock

FROZEN_TIME = datetime(2026, 11, 25, 1, 2, 3, tzinfo=UTC)


@pytest.fixture
def frozen() -> Iterator[None]:
    """Freeze voiceloop_common.utcnow() for deterministic audit timestamps."""
    with use_clock(frozen_clock(FROZEN_TIME)):
        yield


def test_record_fields_time_and_hash_chain(tmp_path: Path, frozen: None) -> None:
    log = PromotionAuditLog(tmp_path / "audit.jsonl")
    first = log.record(
        actor_id="ml-engineer",
        action="promote",
        reason="gate passed",
        before_version="v3",
        after_version="v4",
    )
    second = log.record(
        actor_id="oncall",
        action="rollback",
        reason="canary regression",
        before_version="v4",
        after_version="v3",
    )

    assert first.timestamp == "2026-11-25T01:02:03Z"
    assert first.task_id == "P2-09"
    assert (first.actor_id, first.reason, first.before_version, first.after_version) == (
        "ml-engineer",
        "gate passed",
        "v3",
        "v4",
    )
    assert second.previous_hash == first.entry_hash
    assert log.head_hash() == second.entry_hash
    assert len(log.entries()) == 2
    assert log.verify_chain() == ()


def test_records_are_frozen_and_round_trip(tmp_path: Path) -> None:
    log = PromotionAuditLog(tmp_path / "audit.jsonl")
    record = log.record(
        actor_id="reviewer",
        action="promote",
        reason="ok",
        before_version="v1",
        after_version="v2",
    )

    with pytest.raises(FrozenInstanceError):
        record.actor_id = "someone-else"  # type: ignore[misc]
    assert PromotionAuditLog(tmp_path / "audit.jsonl").entries()[0].to_dict() == record.to_dict()


def test_tampering_is_detected(tmp_path: Path) -> None:
    log = PromotionAuditLog(tmp_path / "audit.jsonl")
    log.record(
        actor_id="ml-engineer",
        action="promote",
        reason="original reason",
        before_version="v3",
        after_version="v4",
    )
    path = tmp_path / "audit.jsonl"
    path.write_text(
        path.read_text(encoding="utf-8").replace("original reason", "changed reason"),
        encoding="utf-8",
    )

    assert any("hash mismatch" in error for error in log.verify_chain())


def test_blank_identity_reason_or_version_is_rejected(tmp_path: Path) -> None:
    log = PromotionAuditLog(tmp_path / "audit.jsonl")

    with pytest.raises(ValidationError, match="actor_id"):
        log.record(
            actor_id=" ",
            action="promote",
            reason="ok",
            before_version="v1",
            after_version="v2",
        )
    with pytest.raises(ValidationError, match="after_version"):
        log.record(
            actor_id="ml-engineer",
            action="promote",
            reason="ok",
            before_version="v1",
            after_version="",
        )


def test_unknown_action_is_rejected(tmp_path: Path) -> None:
    log = PromotionAuditLog(tmp_path / "audit.jsonl")

    with pytest.raises(ValidationError, match="Unknown promotion action"):
        log.record(
            actor_id="ml-engineer",
            action="delete",  # type: ignore[arg-type]
            reason="ok",
            before_version="v1",
            after_version="v2",
        )
