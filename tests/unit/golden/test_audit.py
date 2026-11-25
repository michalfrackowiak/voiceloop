"""P1-06.3: access audit records reader/writer, time, purpose and task ID."""

from __future__ import annotations

from pathlib import Path

import pytest

from services.datasets.golden import AccessAuditLog
from voiceloop_common import ValidationError


def test_entry_fields_and_hash_chain(tmp_path: Path, frozen: None) -> None:
    log = AccessAuditLog(tmp_path / "audit.jsonl")
    first = log.record(
        actor_id="reviewer-a",
        action="read",
        purpose="inspect candidates",
        task_id="P1-06",
        dataset_id="golden.v1.synthetic",
        version="golden.v1",
    )
    second = log.record(
        actor_id="data-worker",
        action="seal",
        purpose="seal workflow stand-in",
        task_id="P1-06",
        dataset_id="golden.v1.synthetic",
        version="golden.v1",
        content_hash="abc123",
    )
    assert first.timestamp == "2026-11-25T00:00:00Z"
    assert (first.actor_id, first.purpose, first.task_id) == (
        "reviewer-a",
        "inspect candidates",
        "P1-06",
    )
    assert second.previous_hash == first.entry_hash
    assert len(log.entries()) == 2
    assert log.head_hash() == second.entry_hash
    assert log.verify_chain() == ()


def test_append_only_tampering_is_detected(tmp_path: Path, frozen: None) -> None:
    log = AccessAuditLog(tmp_path / "audit.jsonl")
    log.record(
        actor_id="reader",
        action="read",
        purpose="original purpose",
        task_id="P1-06",
        dataset_id="golden.v1.synthetic",
        version="golden.v1",
    )
    path = tmp_path / "audit.jsonl"
    path.write_text(
        path.read_text(encoding="utf-8").replace("original purpose", "tampered"), encoding="utf-8"
    )
    assert any("hash mismatch" in error for error in log.verify_chain())


def test_blank_identity_purpose_or_task_is_rejected(tmp_path: Path) -> None:
    log = AccessAuditLog(tmp_path / "audit.jsonl")
    with pytest.raises(ValidationError, match="actor_id"):
        log.record(
            actor_id=" ",
            action="read",
            purpose="p",
            task_id="P1-06",
            dataset_id="golden.v1",
            version="golden.v1",
        )
