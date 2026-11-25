"""P1-06.3: versioned, hashed, immutable golden set with second-review coverage."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from services.datasets.golden import (
    AccessAuditLog,
    LabelRecord,
    amend_sealed_set,
    assert_sealed,
    compute_content_hash,
    seal_golden_set,
    verify_seal,
)
from voiceloop_common import ValidationError

ARTIFACT_DIR = "services/datasets/golden/artifacts"


def _kwargs() -> dict[str, Any]:
    """Return common seal arguments for unit-test artifacts."""
    return {
        "dataset_id": "golden.v1.test",
        "actor_id": "data-worker",
        "reviewer_id": "reviewer-b",
        "task_id": "P1-06",
        "purpose": "unit test seal",
        "synthetic": True,
        "workflow_stand_in": True,
        "review_mode": "simulated_synthetic_labels",
    }


def _tamper_labels(path: Path, field: str, value: Any) -> None:
    """Change exactly one field of the first label in a sealed document."""
    document = json.loads(path.read_text(encoding="utf-8"))
    document["labels"][0][field] = value
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_id", "call-0001:99"),
        ("call_id", "call-0002"),
        ("turn_index", 99),
        ("corrected_transcript", "tampered transcript"),
        ("slots", {"guests": 2}),
        ("first_reviewer_id", "reviewer-z"),
        ("second_reviewer_id", "reviewer-z"),
        ("agreed", False),
    ],
)
def test_content_hash_covers_every_label_field(
    field: str, value: Any, small_labels: tuple[LabelRecord, ...]
) -> None:
    original = compute_content_hash(small_labels)
    assert original == compute_content_hash(tuple(reversed(small_labels)))
    changed = (replace(small_labels[0], **{field: value}), *small_labels[1:])
    assert compute_content_hash(changed) != original


def test_seal_and_verify_success(
    tmp_path: Path, small_labels: tuple[LabelRecord, ...], frozen: None
) -> None:
    artifact = seal_golden_set(tmp_path, small_labels, **_kwargs())
    assert artifact.manifest.count == 2
    assert artifact.manifest.second_review_count == 2
    assert artifact.manifest.second_review_coverage == 1.0
    assert artifact.manifest.agreement_count == 2
    assert artifact.manifest.audit_pointer.entries_at_seal == 2
    assert len(artifact.seal.manifest_hash) == 64
    report = verify_seal(tmp_path)
    report.assert_ok()
    assert assert_sealed(tmp_path).content_hash == compute_content_hash(small_labels)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_id", "call-0001:99"),
        ("call_id", "call-0002"),
        ("turn_index", 99),
        ("corrected_transcript", "tampered transcript"),
        ("slots", {"guests": 2}),
        ("first_reviewer_id", "reviewer-z"),
        ("second_reviewer_id", "reviewer-z"),
        ("agreed", False),
    ],
)
def test_single_field_label_tampering_fails_verification(
    field: str,
    value: Any,
    tmp_path: Path,
    small_labels: tuple[LabelRecord, ...],
    frozen: None,
) -> None:
    artifact = seal_golden_set(tmp_path, small_labels, **_kwargs())
    assert verify_seal(tmp_path).ok
    _tamper_labels(artifact.labels_path, field, value)
    report = verify_seal(tmp_path)
    assert not report.ok
    assert any("labels were modified" in error for error in report.errors)


def test_labels_document_version_tampering_fails_verification(
    tmp_path: Path, small_labels: tuple[LabelRecord, ...], frozen: None
) -> None:
    artifact = seal_golden_set(tmp_path, small_labels, **_kwargs())
    document = json.loads(artifact.labels_path.read_text(encoding="utf-8"))
    document["version"] = "golden.v2"
    artifact.labels_path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    report = verify_seal(tmp_path)
    assert not report.ok
    assert any("labels document version" in error for error in report.errors)


def test_modified_manifest_fails_verification(
    tmp_path: Path, small_labels: tuple[LabelRecord, ...], frozen: None
) -> None:
    artifact = seal_golden_set(tmp_path, small_labels, **_kwargs())
    document = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    document["count"] = 99
    artifact.manifest_path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    report = verify_seal(tmp_path)
    assert not report.ok
    assert any("manifest was modified" in error for error in report.errors)


def test_modified_audit_prefix_fails_verification(
    tmp_path: Path, small_labels: tuple[LabelRecord, ...], frozen: None
) -> None:
    seal_golden_set(tmp_path, small_labels, **_kwargs())
    log_path = tmp_path / "audit.jsonl"
    log_path.write_text(
        log_path.read_text(encoding="utf-8").replace("unit test seal", "tampered"), encoding="utf-8"
    )
    report = verify_seal(tmp_path)
    assert not report.ok
    assert any("audit" in error for error in report.errors)


def test_explicit_verification_is_audited(
    tmp_path: Path, small_labels: tuple[LabelRecord, ...], frozen: None
) -> None:
    seal_golden_set(tmp_path, small_labels, **_kwargs())
    log = AccessAuditLog(tmp_path / "audit.jsonl")
    before = len(log.entries())
    assert verify_seal(tmp_path, actor_id="verifier-a", purpose="quality review", task_id="CP-1").ok
    assert_sealed(tmp_path, actor_id="verifier-b", purpose="merge gate", task_id="CP-1")
    entries = log.entries()
    assert len(entries) == before + 2
    audit_fields = [
        (entry.action, entry.actor_id, entry.purpose, entry.task_id) for entry in entries[-2:]
    ]
    assert audit_fields == [
        ("verify", "verifier-a", "quality review", "CP-1"),
        ("verify", "verifier-b", "merge gate", "CP-1"),
    ]
    assert entries[-1].timestamp == "2026-11-25T00:00:00Z"
    assert log.verify_chain() == ()
    assert verify_seal(tmp_path).ok


def test_implicit_verification_is_read_only(
    tmp_path: Path, small_labels: tuple[LabelRecord, ...], frozen: None
) -> None:
    seal_golden_set(tmp_path, small_labels, **_kwargs())
    log = AccessAuditLog(tmp_path / "audit.jsonl")
    before = len(log.entries())
    assert verify_seal(tmp_path).ok
    assert_sealed(tmp_path)
    assert len(log.entries()) == before


def test_partial_audit_context_is_rejected(
    tmp_path: Path, small_labels: tuple[LabelRecord, ...], frozen: None
) -> None:
    seal_golden_set(tmp_path, small_labels, **_kwargs())
    with pytest.raises(ValidationError, match="actor_id, purpose and task_id"):
        verify_seal(tmp_path, actor_id="verifier-a")


def test_existing_seal_is_never_overwritten(
    tmp_path: Path, small_labels: tuple[LabelRecord, ...], frozen: None
) -> None:
    seal_golden_set(tmp_path, small_labels, **_kwargs())
    with pytest.raises(ValidationError, match="refusing to overwrite"):
        seal_golden_set(tmp_path, small_labels, **_kwargs())


def test_seal_requires_full_second_review(
    tmp_path: Path, small_labels: tuple[LabelRecord, ...], frozen: None
) -> None:
    partial = (replace(small_labels[0], second_reviewer_id=""), *small_labels[1:])
    with pytest.raises(ValidationError, match="second review"):
        seal_golden_set(tmp_path, partial, **_kwargs())


def test_amendment_is_audited_and_preserves_predecessor(
    tmp_path: Path, small_labels: tuple[LabelRecord, ...], frozen: None
) -> None:
    original = seal_golden_set(tmp_path, small_labels, **_kwargs())
    changed = (
        replace(small_labels[0], corrected_transcript="audited correction"),
        *small_labels[1:],
    )
    amended = amend_sealed_set(
        tmp_path,
        changed,
        previous_version="golden.v1",
        new_version="golden.v2",
        **{**_kwargs(), "dataset_id": "golden.v2.test"},
    )
    assert amended.manifest.predecessor_hash == original.manifest.content_hash
    assert assert_sealed(tmp_path, "golden.v1").count == 2
    assert assert_sealed(tmp_path, "golden.v2").count == 2
    v1_document = json.loads((tmp_path / "golden.v1.labels.json").read_text(encoding="utf-8"))
    v2_document = json.loads((tmp_path / "golden.v2.labels.json").read_text(encoding="utf-8"))
    assert v1_document["version"] == "golden.v1"
    assert v2_document["version"] == "golden.v2"
    assert verify_seal(tmp_path, "golden.v2").ok
    assert any(
        entry.action == "amend" for entry in AccessAuditLog(tmp_path / "audit.jsonl").entries()
    )
    assert verify_seal(tmp_path).ok


def test_committed_synthetic_v1_artifact_is_verified() -> None:
    report = verify_seal(ARTIFACT_DIR)
    report.assert_ok()
    manifest = assert_sealed(ARTIFACT_DIR)
    assert manifest.count == 36
    assert len(manifest.call_ids) == 6
    assert manifest.second_review_count == 36
    assert manifest.second_review_coverage == 1.0
    assert manifest.synthetic and manifest.workflow_stand_in
    assert manifest.review_mode == "simulated_synthetic_labels"
    labels_path = Path(ARTIFACT_DIR) / "golden.v1.labels.json"
    document = json.loads(labels_path.read_text(encoding="utf-8"))
    assert document["version"] == "golden.v1"


def test_verify_cli_reports_ok_and_audits_access(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    small_labels: tuple[LabelRecord, ...],
    frozen: None,
) -> None:
    from services.datasets.golden.seal import main

    seal_golden_set(tmp_path, small_labels, **_kwargs())
    assert (
        main(
            [
                "verify",
                "--dir",
                str(tmp_path),
                "--actor",
                "cli-verifier",
                "--purpose",
                "release check",
                "--task-id",
                "CP-1",
            ]
        )
        == 0
    )
    assert '"ok":true' in capsys.readouterr().out
    assert any(
        entry.action == "verify" and entry.actor_id == "cli-verifier"
        for entry in AccessAuditLog(tmp_path / "audit.jsonl").entries()
    )
