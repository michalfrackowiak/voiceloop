"""P3-09.1: versioned candidate artifacts preserve reproducible predecessors."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from services.datasets.golden.maintenance import (
    ARTIFACT_LABEL,
    SEALED_CONTENT_HASH,
    CandidateSet,
    GoldenMaintenanceSettings,
    ReviewAuditLog,
    ReviewBundle,
    build_candidate_set,
    candidate_seal_hash,
    canonical_json,
    create_candidate_version,
    record_candidate_build,
    run_scripted_review,
    sha256_file,
    verify_candidate_version,
)
from voiceloop_common import ValidationError

PURPOSE = "P3-09 synthetic rehearsal"
SEALED_FILES = (
    "services/datasets/golden/artifacts/audit.jsonl",
    "services/datasets/golden/artifacts/golden.v1.json",
    "services/datasets/golden/artifacts/golden.v1.labels.json",
    "services/datasets/golden/artifacts/golden.v1.seal.json",
)


def _sealed_hashes() -> dict[str, str]:
    """Return raw SHA-256 digests of every sealed artifact file."""
    return {path: sha256_file(path) for path in SEALED_FILES}


def _reviewed(
    settings: GoldenMaintenanceSettings,
) -> tuple[CandidateSet, ReviewBundle, ReviewAuditLog]:
    """Build and review a candidate version using the shared audit log."""
    candidate_set = build_candidate_set(settings)
    audit = ReviewAuditLog(settings.output_dir + "/review_audit.jsonl")
    record_candidate_build(audit, candidate_set, settings, actor_id="data-steward", purpose=PURPOSE)
    bundle = run_scripted_review(candidate_set, settings, audit)
    return candidate_set, bundle, audit


def test_candidate_version_is_sealed_verified_and_leaves_sealed_files_unchanged(
    settings: GoldenMaintenanceSettings,
    reviewed_candidate: tuple[CandidateSet, ReviewBundle, ReviewAuditLog],
) -> None:
    """A new version records provenance and only writes to the output directory."""
    candidate_set, bundle, audit = reviewed_candidate
    before = _sealed_hashes()
    artifact = create_candidate_version(
        settings.output_dir,
        candidate_set,
        bundle,
        settings=settings,
        actor_id="data-steward",
        purpose=PURPOSE,
    )
    assert artifact.manifest.version == "candidate.v1"
    assert artifact.manifest.label == ARTIFACT_LABEL
    assert artifact.manifest.sealed_content_hash == SEALED_CONTENT_HASH
    assert artifact.manifest.predecessor_version is None
    assert artifact.manifest.audit_entries_at_seal == len(audit.entries())
    assert artifact.seal.immutable is True
    assert artifact.manifest_path.parent == artifact.candidates_path.parent

    report = verify_candidate_version(settings.output_dir, "candidate.v1", settings)
    assert report.ok is True
    assert report.errors == ()
    assert _sealed_hashes() == before


def test_overwrite_and_sealed_output_are_rejected(
    settings: GoldenMaintenanceSettings,
    reviewed_candidate: tuple[CandidateSet, ReviewBundle, ReviewAuditLog],
) -> None:
    """No candidate can amend an existing version or write into sealed artifacts."""
    candidate_set, bundle, _ = reviewed_candidate
    create_candidate_version(
        settings.output_dir,
        candidate_set,
        bundle,
        settings=settings,
        actor_id="data-steward",
        purpose=PURPOSE,
    )
    with pytest.raises(ValidationError, match="overwrite"):
        create_candidate_version(
            settings.output_dir,
            candidate_set,
            bundle,
            settings=settings,
            actor_id="data-steward",
            purpose=PURPOSE,
        )
    with pytest.raises(ValidationError, match="sealed golden artifacts"):
        create_candidate_version(
            "services/datasets/golden/artifacts",
            candidate_set,
            bundle,
            settings=settings,
            actor_id="data-steward",
            purpose=PURPOSE,
        )


def test_tampered_manifest_is_detected(
    settings: GoldenMaintenanceSettings,
    reviewed_candidate: tuple[CandidateSet, ReviewBundle, ReviewAuditLog],
) -> None:
    """Changing a sealed manifest after writing invalidates its seal."""
    candidate_set, bundle, _ = reviewed_candidate
    artifact = create_candidate_version(
        settings.output_dir,
        candidate_set,
        bundle,
        settings=settings,
        actor_id="data-steward",
        purpose=PURPOSE,
    )
    payload = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    payload["candidate_count"] = 999
    artifact.manifest_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    report = verify_candidate_version(settings.output_dir, "candidate.v1", settings)
    assert report.ok is False
    assert any("manifest was modified" in error for error in report.errors)


def _file_snapshot(directory: str | Path) -> dict[str, bytes]:
    """Return every file under a directory as relative path to raw bytes."""
    base = Path(directory)
    return {
        path.relative_to(base).as_posix(): path.read_bytes()
        for path in base.rglob("*")
        if path.is_file()
    }


def test_tampered_seal_second_reviewer_fails_closed_without_writes(
    settings: GoldenMaintenanceSettings,
    reviewed_candidate: tuple[CandidateSet, ReviewBundle, ReviewAuditLog],
) -> None:
    """A seal swapped to an untrusted reviewer fails verification without writing."""
    candidate_set, bundle, _ = reviewed_candidate
    artifact = create_candidate_version(
        settings.output_dir,
        candidate_set,
        bundle,
        settings=settings,
        actor_id="data-steward",
        purpose=PURPOSE,
    )
    assert verify_candidate_version(settings.output_dir, "candidate.v1", settings).ok is True

    seal_payload = json.loads(artifact.seal_path.read_text(encoding="utf-8"))
    seal_payload["second_reviewer_id"] = "attacker-b"
    artifact.seal_path.write_text(canonical_json(seal_payload) + "\n", encoding="utf-8")
    assert seal_payload["second_reviewer_id"] != settings.second_reviewer_id

    before = _file_snapshot(settings.output_dir)
    sealed_before = _sealed_hashes()

    report = verify_candidate_version(settings.output_dir, "candidate.v1", settings)
    assert report.ok is False
    assert "candidate seal second reviewer is not the configured reviewer pin" in report.errors

    after = _file_snapshot(settings.output_dir)
    assert after == before
    assert set(after) == {
        "candidate.v1.json",
        "candidate.v1.candidates.json",
        "candidate.v1.seal.json",
        "review_audit.jsonl",
    }
    assert _sealed_hashes() == sealed_before


def test_shortened_audit_prefix_is_detected(
    settings: GoldenMaintenanceSettings,
    reviewed_candidate: tuple[CandidateSet, ReviewBundle, ReviewAuditLog],
) -> None:
    """Removing an entry behind the sealed audit pointer fails verification."""
    candidate_set, bundle, audit = reviewed_candidate
    create_candidate_version(
        settings.output_dir,
        candidate_set,
        bundle,
        settings=settings,
        actor_id="data-steward",
        purpose=PURPOSE,
    )
    lines = audit.path.read_text(encoding="utf-8").splitlines()
    audit.path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    report = verify_candidate_version(settings.output_dir, "candidate.v1", settings)
    assert report.ok is False
    assert any(
        "audit log is shorter" in error or "sealed audit prefix" in error for error in report.errors
    )


def _version_files(output_dir: str) -> tuple[Path, Path, Path]:
    """Return the three files a version write would create."""
    base = Path(output_dir)
    return (
        base / "candidate.v1.json",
        base / "candidate.v1.candidates.json",
        base / "candidate.v1.seal.json",
    )


def test_predecessor_is_reproducible_for_v2(settings: GoldenMaintenanceSettings) -> None:
    """A second version records and re-verifies the previous seal exactly."""
    first_set, first_bundle, _ = _reviewed(settings)
    first = create_candidate_version(
        settings.output_dir,
        first_set,
        first_bundle,
        settings=settings,
        actor_id="data-steward",
        purpose=PURPOSE,
    )

    second_settings = GoldenMaintenanceSettings(
        output_dir=settings.output_dir,
        candidate_version="candidate.v2",
        candidate_calls=3,
        candidate_seed=12,
    )
    second_set, second_bundle, _ = _reviewed(second_settings)
    second = create_candidate_version(
        second_settings.output_dir,
        second_set,
        second_bundle,
        settings=second_settings,
        actor_id="data-steward",
        purpose=PURPOSE,
        predecessor_version="candidate.v1",
    )
    assert second.manifest.predecessor_version == "candidate.v1"
    assert second.manifest.predecessor_content_hash == first.manifest.content_hash
    assert second.manifest.predecessor_seal_hash == candidate_seal_hash(first.seal)
    assert second_set.content_hash != first_set.content_hash
    assert verify_candidate_version(second_settings.output_dir, "candidate.v2", second_settings).ok


def test_duplicate_pair_bundle_is_rejected_before_any_version_write(
    settings: GoldenMaintenanceSettings,
    reviewed_candidate: tuple[CandidateSet, ReviewBundle, ReviewAuditLog],
) -> None:
    """Twelve pairs with one duplicated candidate cannot seal a version."""
    candidate_set, bundle, _ = reviewed_candidate
    duplicated = replace(bundle, pairs=bundle.pairs[:-2] + (bundle.pairs[-1], bundle.pairs[-1]))
    assert len(duplicated.pairs) == len(candidate_set.records)

    with pytest.raises(ValidationError, match="duplicate candidate pairs"):
        create_candidate_version(
            settings.output_dir,
            candidate_set,
            duplicated,
            settings=settings,
            actor_id="data-steward",
            purpose=PURPOSE,
        )
    assert not any(path.exists() for path in _version_files(settings.output_dir))
    assert {path.name for path in Path(settings.output_dir).iterdir()} == {"review_audit.jsonl"}


def test_unapproved_write_fails_closed(
    settings: GoldenMaintenanceSettings,
    reviewed_candidate: tuple[CandidateSet, ReviewBundle, ReviewAuditLog],
) -> None:
    """An incomplete second review cannot seal a version or leave partial files."""
    candidate_set, bundle, audit = reviewed_candidate
    lines = audit.path.read_text(encoding="utf-8").splitlines()
    audit.path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="review not approved"):
        create_candidate_version(
            settings.output_dir,
            candidate_set,
            bundle,
            settings=settings,
            actor_id="data-steward",
            purpose=PURPOSE,
        )
    assert not any(path.exists() for path in _version_files(settings.output_dir))


def test_alternate_reviewers_fail_verification_under_trusted_pins(
    settings: GoldenMaintenanceSettings,
) -> None:
    """An artifact sealed by other reviewers cannot verify with trusted settings."""
    alternate = GoldenMaintenanceSettings(
        output_dir=settings.output_dir,
        first_reviewer_id="attacker-a",
        second_reviewer_id="attacker-b",
    )
    candidate_set, bundle, _ = _reviewed(alternate)
    create_candidate_version(
        alternate.output_dir,
        candidate_set,
        bundle,
        settings=alternate,
        actor_id="data-steward",
        purpose=PURPOSE,
    )

    report = verify_candidate_version(settings.output_dir, "candidate.v1", settings)
    assert report.ok is False
    assert any("configured reviewer pin" in error for error in report.errors)
