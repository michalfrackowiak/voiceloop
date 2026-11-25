"""P3-09.1: two distinct scripted reviewers with a fail-closed audit trail."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from services.datasets.golden.maintenance import (
    ARTIFACT_LABEL,
    CandidateSet,
    GoldenMaintenanceSettings,
    ReviewAuditLog,
    ReviewBundle,
    answer_digest,
    assert_review_approved,
    record_candidate_build,
    run_scripted_review,
    verify_review_bundle,
)
from voiceloop_common import ValidationError

TIMESTAMP = "2026-11-25T00:00:00Z"


def test_two_distinct_reviewers_are_recorded_and_audited(
    settings: GoldenMaintenanceSettings,
    reviewed_candidate: tuple[CandidateSet, ReviewBundle, ReviewAuditLog],
) -> None:
    """Each candidate receives two distinct, hashed and timestamped answers."""
    candidate_set, bundle, audit = reviewed_candidate
    entries = audit.entries()
    assert len(entries) == 1 + 2 * len(candidate_set.records)
    assert entries[0].action == "candidate_build"
    assert bundle.first_reviewer_id == settings.first_reviewer_id
    assert bundle.second_reviewer_id == settings.second_reviewer_id
    assert bundle.first_reviewer_id != bundle.second_reviewer_id

    for index, pair in enumerate(bundle.pairs):
        primary, secondary = pair
        assert primary.reviewer_id == settings.first_reviewer_id
        assert secondary.reviewer_id == settings.second_reviewer_id
        assert primary.candidate_id == secondary.candidate_id
        assert primary.decision == secondary.decision == "include"
        assert primary.submitted_at == secondary.submitted_at == TIMESTAMP
        assert primary.answer_hash == answer_digest(primary.candidate_id, "include")
        assert secondary.answer_hash == answer_digest(secondary.candidate_id, "include")
        first_entry, second_entry = entries[1 + 2 * index : 3 + 2 * index]
        assert (first_entry.action, second_entry.action) == ("first_review", "second_review")
        assert first_entry.actor_id == primary.reviewer_id
        assert second_entry.actor_id == secondary.reviewer_id
        assert first_entry.content_hash == primary.answer_hash
        assert second_entry.content_hash == secondary.answer_hash

    for entry in entries:
        assert entry.label == ARTIFACT_LABEL
        assert entry.timestamp == TIMESTAMP
        assert entry.task_id == settings.task_id
        assert len(entry.content_hash) == 64
        assert len(entry.entry_hash) == 64

    assert audit.verify_chain() == ()
    assert verify_review_bundle(bundle, candidate_set.records, audit, settings) == ()
    assert_review_approved(bundle, candidate_set.records, audit, settings)


def test_review_timestamps_are_bound_to_submissions(
    settings: GoldenMaintenanceSettings,
    candidate_set: CandidateSet,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ticking clock cannot detach an audit entry from its review answer."""
    from services.datasets.golden.maintenance import review as review_module

    ticks = iter(range(1, 100))

    def ticking_clock() -> str:
        return f"2026-11-25T00:00:{next(ticks):02d}Z"

    monkeypatch.setattr(review_module, "utc_timestamp", ticking_clock)
    audit = ReviewAuditLog(tmp_path / "audit.jsonl")
    record_candidate_build(
        audit, candidate_set, settings, actor_id="data-steward", purpose="P3-09 rehearsal"
    )
    bundle = run_scripted_review(candidate_set, settings, audit)

    first_entries = [entry for entry in audit.entries() if entry.action == "first_review"]
    assert [entry.timestamp for entry in first_entries] == [
        primary.submitted_at for primary, _ in bundle.pairs
    ]
    assert verify_review_bundle(bundle, candidate_set.records, audit, settings) == ()


def test_missing_audit_entry_fails_closed(
    settings: GoldenMaintenanceSettings,
    reviewed_candidate: tuple[CandidateSet, ReviewBundle, ReviewAuditLog],
) -> None:
    """Removing one reviewer entry invalidates the approval."""
    candidate_set, bundle, audit = reviewed_candidate
    lines = audit.path.read_text(encoding="utf-8").splitlines()
    audit.path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    errors = verify_review_bundle(bundle, candidate_set.records, audit, settings)
    assert any("audited second_review missing" in error for error in errors)


def test_tampered_audit_prefix_fails_closed(
    settings: GoldenMaintenanceSettings,
    reviewed_candidate: tuple[CandidateSet, ReviewBundle, ReviewAuditLog],
) -> None:
    """Changing an audited reason breaks the chain and review approval."""
    candidate_set, bundle, audit = reviewed_candidate
    original = audit.path.read_text(encoding="utf-8")
    audit.path.write_text(
        original.replace("scripted synthetic review", "tampered synthetic review", 1),
        encoding="utf-8",
    )
    errors = verify_review_bundle(bundle, candidate_set.records, audit, settings)
    assert any("hash mismatch" in error or "broken chain" in error for error in errors)
    with pytest.raises(ValidationError):
        assert_review_approved(bundle, candidate_set.records, audit, settings)


def test_single_reviewer_bundle_fails_closed(
    settings: GoldenMaintenanceSettings,
    reviewed_candidate: tuple[CandidateSet, ReviewBundle, ReviewAuditLog],
) -> None:
    """A bundle that reuses one actor cannot pass the distinct-reviewer check."""
    _, bundle, audit = reviewed_candidate
    single_pairs = tuple(
        (primary, replace(secondary, reviewer_id=primary.reviewer_id))
        for primary, secondary in bundle.pairs
    )
    single_bundle = replace(
        bundle,
        second_reviewer_id=bundle.first_reviewer_id,
        pairs=single_pairs,
    )
    errors = verify_review_bundle(single_bundle, (), audit, settings)
    assert any("single reviewer" in error for error in errors)


def test_build_actor_must_be_allowlisted_and_not_a_reviewer(
    settings: GoldenMaintenanceSettings, candidate_set: CandidateSet, tmp_path: Path
) -> None:
    """The audited build entry cannot be attributed to an untrusted actor."""
    audit = ReviewAuditLog(tmp_path / "audit.jsonl")
    with pytest.raises(ValidationError, match="allowlist"):
        record_candidate_build(
            audit, candidate_set, settings, actor_id="unknown-actor-999", purpose="P3-09 rehearsal"
        )
    with pytest.raises(ValidationError, match="reviewer"):
        record_candidate_build(
            audit,
            candidate_set,
            settings,
            actor_id=settings.first_reviewer_id,
            purpose="P3-09 rehearsal",
        )
    assert not audit.path.exists()


def test_audit_entry_label_is_enforced(tmp_path: Path) -> None:
    """An audit entry missing the mandatory stand-in label cannot be replayed."""
    from services.datasets.golden.maintenance import AuditEntry

    entry = AuditEntry(
        actor_id="reviewer-x",
        action="first_review",
        reason="scripted synthetic review",
        task_id="P3-09",
        timestamp=TIMESTAMP,
        version="candidate.v1",
        content_hash="0" * 64,
    )
    raw = entry.to_dict()
    raw["label"] = "human-reviewed golden set"
    (tmp_path / "audit.jsonl").write_text(json.dumps(raw) + "\n", encoding="utf-8")
    audit = ReviewAuditLog(tmp_path / "audit.jsonl")
    with pytest.raises(ValidationError, match="label"):
        audit.entries()
