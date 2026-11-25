"""Versioned, audited candidate artifacts with reproducible predecessors.

A candidate version is a new ``candidate.vN`` artifact set under the maintenance
output directory. It records the read-only sealed golden hash and seal, the
previous candidate hash and seal, and a pointer into the two-reviewer audit log.
Nothing is ever amended in place and the sealed P1-06 artifacts are verified
read-only on every write and every verification.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from services.datasets.golden import assert_sealed
from services.datasets.golden.seal import load_labels
from services.datasets.golden.seal import manifest_hash as sealed_manifest_hash
from voiceloop_common import ValidationError

from .candidates import (
    CandidateSet,
    candidate_document,
    load_candidate_document,
    verify_fixture_hash,
)
from .review import (
    ReviewAuditLog,
    ReviewBundle,
    assert_review_approved,
    verify_review_bundle,
)
from .settings import (
    ARTIFACT_LABEL,
    GoldenMaintenanceSettings,
    assert_allowed_actor,
    assert_outside_sealed,
    assert_safe_text,
    assert_valid_hash,
    assert_version,
    canonical_json,
    resolve_path,
    sha256_text,
    utc_timestamp,
    version_number,
)

REVIEW_MODE = "scripted_synthetic_reviewers"


@dataclass(frozen=True)
class SealedProvenance:
    """Read-only hashes of the sealed golden stand-in and its source fixture."""

    version: str
    content_hash: str
    manifest_hash: str
    fixture_sha256: str


@dataclass(frozen=True)
class CandidateManifest:
    """Versioned manifest for one immutable synthetic candidate artifact."""

    version: str
    label: str
    candidate_count: int
    call_refs: tuple[str, ...]
    content_hash: str
    created_at: str
    synthetic: bool
    workflow_stand_in: bool
    review_mode: str
    first_reviewer_id: str
    second_reviewer_id: str
    candidate_source_sha256: str
    sealed_version: str
    sealed_content_hash: str
    sealed_manifest_hash: str
    predecessor_version: str | None
    predecessor_content_hash: str | None
    predecessor_seal_hash: str | None
    audit_path: str
    audit_entries_at_seal: int
    audit_head_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of the manifest."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CandidateManifest:
        """Rebuild and validate a manifest from its on-disk JSON object."""
        try:
            manifest = cls(
                version=str(data["version"]),
                label=str(data["label"]),
                candidate_count=int(data["candidate_count"]),
                call_refs=tuple(str(item) for item in data["call_refs"]),
                content_hash=str(data["content_hash"]),
                created_at=str(data["created_at"]),
                synthetic=bool(data["synthetic"]),
                workflow_stand_in=bool(data["workflow_stand_in"]),
                review_mode=str(data["review_mode"]),
                first_reviewer_id=str(data["first_reviewer_id"]),
                second_reviewer_id=str(data["second_reviewer_id"]),
                candidate_source_sha256=str(data["candidate_source_sha256"]),
                sealed_version=str(data["sealed_version"]),
                sealed_content_hash=str(data["sealed_content_hash"]),
                sealed_manifest_hash=str(data["sealed_manifest_hash"]),
                predecessor_version=data.get("predecessor_version"),
                predecessor_content_hash=data.get("predecessor_content_hash"),
                predecessor_seal_hash=data.get("predecessor_seal_hash"),
                audit_path=str(data["audit_path"]),
                audit_entries_at_seal=int(data["audit_entries_at_seal"]),
                audit_head_hash=str(data["audit_head_hash"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"malformed candidate manifest: {exc}") from exc
        assert_version(manifest.version)
        assert_valid_hash(manifest.content_hash, "content_hash")
        assert_valid_hash(manifest.candidate_source_sha256, "candidate_source_sha256")
        assert_valid_hash(manifest.sealed_content_hash, "sealed_content_hash")
        assert_valid_hash(manifest.sealed_manifest_hash, "sealed_manifest_hash")
        for name in ("predecessor_content_hash", "predecessor_seal_hash"):
            value = getattr(manifest, name)
            if value is not None:
                assert_valid_hash(str(value), name)
        return manifest


@dataclass(frozen=True)
class CandidateSeal:
    """Seal over the canonical bytes of one candidate manifest."""

    version: str
    label: str
    manifest_hash: str
    content_hash: str
    sealed_at: str
    sealed_by: str
    second_reviewer_id: str
    predecessor_content_hash: str | None
    immutable: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of the seal."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CandidateSeal:
        """Rebuild and validate a seal from its on-disk JSON object."""
        try:
            seal = cls(
                version=str(data["version"]),
                label=str(data["label"]),
                manifest_hash=str(data["manifest_hash"]),
                content_hash=str(data["content_hash"]),
                sealed_at=str(data["sealed_at"]),
                sealed_by=str(data["sealed_by"]),
                second_reviewer_id=str(data["second_reviewer_id"]),
                predecessor_content_hash=data.get("predecessor_content_hash"),
                immutable=bool(data.get("immutable", True)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"malformed candidate seal: {exc}") from exc
        assert_version(seal.version)
        assert_valid_hash(seal.manifest_hash, "manifest_hash")
        assert_valid_hash(seal.content_hash, "content_hash")
        return seal


@dataclass(frozen=True)
class SealedCandidateArtifact:
    """Paths and parsed objects returned by a successful candidate version write."""

    manifest: CandidateManifest
    seal: CandidateSeal
    manifest_path: Path
    candidates_path: Path
    seal_path: Path
    audit_path: Path


@dataclass(frozen=True)
class CandidateVerificationReport:
    """Result of verifying a candidate version and its predecessor chain."""

    ok: bool
    errors: tuple[str, ...] = ()
    manifest: CandidateManifest | None = None
    seal: CandidateSeal | None = None

    def assert_ok(self) -> None:
        """Raise :class:`ValidationError` when any integrity check failed."""
        if not self.ok:
            raise ValidationError("candidate verification failed: " + "; ".join(self.errors))


def _paths(output_dir: str | Path, version: str) -> tuple[Path, Path, Path, Path, Path]:
    """Return base, manifest, candidates, seal and audit paths for a version."""
    base = resolve_path(output_dir)
    return (
        base,
        base / f"{version}.json",
        base / f"{version}.candidates.json",
        base / f"{version}.seal.json",
        base / "review_audit.jsonl",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write canonical JSON with a trailing newline via a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _read_object(path: Path, kind: str, errors: list[str]) -> dict[str, Any] | None:
    """Read one JSON object, recording an error on any failure."""
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read {kind}: {exc}")
        return None
    if not isinstance(decoded, dict):
        errors.append(f"{kind} must be a JSON object")
        return None
    return decoded


def sealed_provenance(settings: GoldenMaintenanceSettings) -> SealedProvenance:
    """Verify the sealed synthetic stand-in read-only and return its hashes.

    Args:
        settings: Typed maintenance settings with the committed hash pins.

    Returns:
        Version, content hash, manifest hash and fixture hash of the sealed set.

    Raises:
        ValidationError: If the sealed artifact or fixture no longer matches.

    """
    manifest = assert_sealed(settings.sealed_artifact_dir, settings.sealed_version)
    if manifest.content_hash != settings.expected_sealed_content_hash:
        raise ValidationError("sealed golden content hash no longer matches the committed pin")
    if not manifest.synthetic or not manifest.workflow_stand_in:
        raise ValidationError("sealed predecessor is not labelled as a synthetic stand-in")
    return SealedProvenance(
        version=manifest.version,
        content_hash=manifest.content_hash,
        manifest_hash=sealed_manifest_hash(manifest),
        fixture_sha256=verify_fixture_hash(settings),
    )


def candidate_manifest_hash(manifest: CandidateManifest) -> str:
    """Return the SHA-256 hash of a candidate manifest's canonical bytes."""
    return sha256_text(canonical_json(manifest.to_dict()))


def candidate_seal_hash(seal: CandidateSeal) -> str:
    """Return the SHA-256 hash of a candidate seal's canonical bytes."""
    return sha256_text(canonical_json(seal.to_dict()))


def create_candidate_version(
    output_dir: str | Path,
    candidate_set: CandidateSet,
    bundle: ReviewBundle,
    *,
    settings: GoldenMaintenanceSettings,
    actor_id: str,
    purpose: str,
    task_id: str | None = None,
    predecessor_version: str | None = None,
) -> SealedCandidateArtifact:
    """Seal a new candidate version after a fully audited second review.

    Args:
        output_dir: Maintenance-owned destination; never the sealed artifact dir.
        candidate_set: Separate synthetic candidate set to seal.
        bundle: Two scripted reviews per candidate.
        settings: Typed maintenance settings.
        actor_id: Pseudonymous operator performing the write.
        purpose: PII-screened reason for the version.
        task_id: Authorising roadmap task; defaults to the setting.
        predecessor_version: Previous candidate version, required for ``v2+``.

    Returns:
        Paths and parsed objects for the new version.

    Raises:
        ValidationError: On incomplete review, broken predecessor, hash drift,
            missing audit context or an attempt to overwrite an existing version.

    """
    assert_outside_sealed(output_dir, settings)
    assert_allowed_actor(actor_id, settings)
    assert_safe_text(purpose, "purpose")
    effective_task = task_id or settings.task_id
    assert_safe_text(effective_task, "task_id")
    if candidate_set.version != settings.candidate_version:
        raise ValidationError("candidate set version does not match settings")
    number = version_number(candidate_set.version)
    if number == 1 and predecessor_version is not None:
        raise ValidationError("the first candidate version cannot have a predecessor")
    if number > 1 and predecessor_version is None:
        raise ValidationError("versions after candidate.v1 require a verified predecessor")
    predecessor: SealedCandidateArtifact | None = None
    if predecessor_version is not None:
        assert_version(predecessor_version, "predecessor_version")
        if version_number(predecessor_version) != number - 1:
            raise ValidationError("predecessor must be the immediately previous version")
        previous = verify_candidate_version(output_dir, predecessor_version, settings)
        previous.assert_ok()
        if previous.manifest is None or previous.seal is None:  # pragma: no cover - guarded
            raise ValidationError("predecessor report has no manifest or seal")
        predecessor = SealedCandidateArtifact(
            previous.manifest,
            previous.seal,
            _paths(output_dir, predecessor_version)[1],
            _paths(output_dir, predecessor_version)[2],
            _paths(output_dir, predecessor_version)[3],
            _paths(output_dir, predecessor_version)[4],
        )

    _, manifest_path, candidates_path, seal_path, audit_path = _paths(
        output_dir, candidate_set.version
    )
    existing = [path for path in (manifest_path, candidates_path, seal_path) if path.exists()]
    if existing:
        raise ValidationError("refusing to overwrite an existing candidate version")

    log = ReviewAuditLog(audit_path)
    assert_review_approved(bundle, candidate_set.records, log, settings)
    sealed = sealed_provenance(settings)
    log.record(
        actor_id=actor_id,
        action="version_write",
        reason=purpose,
        task_id=effective_task,
        version=candidate_set.version,
        content_hash=candidate_set.content_hash,
    )
    log.record(
        actor_id=actor_id,
        action="version_seal",
        reason=purpose,
        task_id=effective_task,
        version=candidate_set.version,
        content_hash=candidate_set.content_hash,
    )
    entries = log.entries()
    manifest = CandidateManifest(
        version=candidate_set.version,
        label=ARTIFACT_LABEL,
        candidate_count=len(candidate_set.records),
        call_refs=candidate_set.call_refs,
        content_hash=candidate_set.content_hash,
        created_at=utc_timestamp(),
        synthetic=True,
        workflow_stand_in=True,
        review_mode=REVIEW_MODE,
        first_reviewer_id=bundle.first_reviewer_id,
        second_reviewer_id=bundle.second_reviewer_id,
        candidate_source_sha256=candidate_set.source_sha256,
        sealed_version=sealed.version,
        sealed_content_hash=sealed.content_hash,
        sealed_manifest_hash=sealed.manifest_hash,
        predecessor_version=predecessor.manifest.version if predecessor else None,
        predecessor_content_hash=predecessor.manifest.content_hash if predecessor else None,
        predecessor_seal_hash=candidate_seal_hash(predecessor.seal) if predecessor else None,
        audit_path="review_audit.jsonl",
        audit_entries_at_seal=len(entries),
        audit_head_hash=entries[-1].entry_hash,
    )
    _write_json(candidates_path, candidate_document(candidate_set, (bundle.to_dict(),)))
    _write_json(manifest_path, manifest.to_dict())
    seal = CandidateSeal(
        version=candidate_set.version,
        label=ARTIFACT_LABEL,
        manifest_hash=candidate_manifest_hash(manifest),
        content_hash=candidate_set.content_hash,
        sealed_at=utc_timestamp(),
        sealed_by=actor_id,
        second_reviewer_id=bundle.second_reviewer_id,
        predecessor_content_hash=manifest.predecessor_content_hash,
    )
    _write_json(seal_path, seal.to_dict())
    artifact = SealedCandidateArtifact(
        manifest, seal, manifest_path, candidates_path, seal_path, log.path
    )
    verify_candidate_version(output_dir, candidate_set.version, settings).assert_ok()
    return artifact


def verify_candidate_version(
    output_dir: str | Path,
    version: str,
    settings: GoldenMaintenanceSettings,
) -> CandidateVerificationReport:
    """Verify one candidate version, its audit, seal and predecessor chain.

    Appended audit entries after sealing are allowed; changing the sealed prefix
    is not. The read-only sealed golden set is re-verified and its hashes are
    compared with the values recorded at seal time.

    Args:
        output_dir: Maintenance-owned directory holding the candidate version.
        version: ``candidate.vN`` version to verify.
        settings: Typed maintenance settings.

    Returns:
        A report with all detected integrity errors.

    """
    try:
        assert_version(version)
    except ValidationError as exc:
        return CandidateVerificationReport(ok=False, errors=(str(exc),))
    _, manifest_path, candidates_path, seal_path, audit_path = _paths(output_dir, version)
    errors: list[str] = []
    manifest_data = _read_object(manifest_path, "candidate manifest", errors)
    seal_data = _read_object(seal_path, "candidate seal", errors)
    manifest: CandidateManifest | None = None
    seal: CandidateSeal | None = None
    if manifest_data is not None and seal_data is not None:
        try:
            manifest = CandidateManifest.from_dict(manifest_data)
            seal = CandidateSeal.from_dict(seal_data)
        except ValidationError as exc:
            errors.append(str(exc))
            manifest, seal = None, None

    if manifest is not None and seal is not None:
        if manifest.version != version or seal.version != version:
            errors.append("version mismatch between path, manifest and seal")
        if manifest.label != ARTIFACT_LABEL or seal.label != ARTIFACT_LABEL:
            errors.append("candidate artifacts lack the mandatory synthetic stand-in label")
        if not manifest.synthetic or not manifest.workflow_stand_in:
            errors.append("candidate manifest is not labelled as a synthetic stand-in")
        if manifest.review_mode != REVIEW_MODE:
            errors.append("candidate review mode is not the scripted stand-in mode")
        if manifest.audit_path != "review_audit.jsonl":
            errors.append("candidate manifest points at an unexpected audit log")
        if manifest.first_reviewer_id != settings.first_reviewer_id:
            errors.append("manifest first reviewer is not the configured reviewer pin")
        if manifest.second_reviewer_id != settings.second_reviewer_id:
            errors.append("manifest second reviewer is not the configured reviewer pin")
        if seal.second_reviewer_id != settings.second_reviewer_id:
            errors.append("candidate seal second reviewer is not the configured reviewer pin")
        if seal.manifest_hash != candidate_manifest_hash(manifest):
            errors.append("candidate manifest was modified after sealing")
        if seal.content_hash != manifest.content_hash:
            errors.append("candidate seal content hash does not match the manifest")
        if not seal.immutable:
            errors.append("candidate seal is marked mutable")
        if seal.sealed_by in (manifest.first_reviewer_id, manifest.second_reviewer_id):
            errors.append("sealing actor must not be a scripted reviewer")
        if seal.sealed_by not in settings.allowed_actor_ids:
            errors.append("sealing actor is not in the configured allowlist")

        candidate_set: CandidateSet | None = None
        review_documents: tuple[dict[str, Any], ...] = ()
        try:
            candidate_set, review_documents = load_candidate_document(candidates_path)
        except ValidationError as exc:
            errors.append(str(exc))
        if candidate_set is not None:
            if candidate_set.version != version:
                errors.append("candidate document version does not match the manifest")
            if candidate_set.content_hash != manifest.content_hash:
                errors.append("candidate document content hash does not match the manifest")
            if len(candidate_set.records) != manifest.candidate_count:
                errors.append("candidate count does not match the manifest")
            if candidate_set.call_refs != manifest.call_refs:
                errors.append("candidate call pseudonyms do not match the manifest")
            if candidate_set.source_sha256 != manifest.candidate_source_sha256:
                errors.append("candidate fixture hash does not match the manifest")
            if len(review_documents) != 1:
                errors.append("candidate document must carry exactly one review bundle")
            else:
                bundle: ReviewBundle | None = None
                try:
                    bundle = ReviewBundle.from_dict(review_documents[0])
                except ValidationError as exc:
                    errors.append(str(exc))
                if bundle is not None:
                    if bundle.first_reviewer_id != settings.first_reviewer_id:
                        errors.append("review bundle first reviewer is not the configured pin")
                    if bundle.second_reviewer_id != settings.second_reviewer_id:
                        errors.append("review bundle second reviewer is not the configured pin")
                    if bundle.first_reviewer_id != manifest.first_reviewer_id:
                        errors.append("manifest first reviewer does not match the review bundle")
                    if bundle.second_reviewer_id != manifest.second_reviewer_id:
                        errors.append("manifest second reviewer does not match the review bundle")
                    if manifest.first_reviewer_id == manifest.second_reviewer_id:
                        errors.append("manifest has a single scripted reviewer")
                    try:
                        review_settings = settings.model_copy(update={"candidate_version": version})
                        errors.extend(
                            verify_review_bundle(
                                bundle,
                                candidate_set.records,
                                ReviewAuditLog(audit_path),
                                review_settings,
                            )
                        )
                    except ValidationError as exc:
                        errors.append(str(exc))

        try:
            provenance = sealed_provenance(settings)
        except ValidationError as exc:
            errors.append(f"sealed provenance check failed: {exc}")
        else:
            if manifest.sealed_version != provenance.version:
                errors.append("recorded sealed version does not match the read-only artifact")
            if manifest.sealed_content_hash != provenance.content_hash:
                errors.append("sealed golden content changed after this candidate was sealed")
            if manifest.sealed_manifest_hash != provenance.manifest_hash:
                errors.append("sealed golden manifest changed after this candidate was sealed")
            if manifest.candidate_source_sha256 != provenance.fixture_sha256:
                errors.append("recorded fixture hash does not match the read-only source")

        pointer_errors = _verify_audit_pointer(manifest, audit_path)
        errors.extend(pointer_errors)
        errors.extend(_verify_predecessor(manifest, seal, output_dir, settings, version))

    return CandidateVerificationReport(
        ok=not errors,
        errors=tuple(errors),
        manifest=manifest,
        seal=seal,
    )


def _verify_audit_pointer(manifest: CandidateManifest, audit_path: Path) -> tuple[str, ...]:
    """Return errors if the sealed audit prefix is missing or modified."""
    errors: list[str] = []
    try:
        entries = ReviewAuditLog(audit_path).entries()
    except ValidationError as exc:
        return (str(exc),)
    if len(entries) < manifest.audit_entries_at_seal:
        errors.append("audit log is shorter than the sealed pointer")
    elif (
        manifest.audit_entries_at_seal > 0
        and entries[manifest.audit_entries_at_seal - 1].entry_hash != manifest.audit_head_hash
    ):
        errors.append("sealed audit prefix was modified")
    return tuple(errors)


def _verify_predecessor(
    manifest: CandidateManifest,
    seal: CandidateSeal,
    output_dir: str | Path,
    settings: GoldenMaintenanceSettings,
    version: str,
) -> tuple[str, ...]:
    """Return errors if a recorded predecessor is absent, broken or tampered."""
    number = version_number(version)
    previous = manifest.predecessor_version
    if number == 1:
        if previous is not None:
            return ("the first candidate version must not record a predecessor",)
        return ()
    if previous is None:
        return ("versions after candidate.v1 must record a predecessor",)
    if version_number(previous) != number - 1:
        return ("recorded predecessor is not the immediately previous version",)
    report = verify_candidate_version(output_dir, previous, settings)
    if not report.ok:
        return (f"predecessor {previous} is not reproducible",)
    if report.manifest is None or report.seal is None:  # pragma: no cover - guarded
        return ("predecessor report has no manifest or seal",)
    errors: list[str] = []
    if manifest.predecessor_content_hash != report.manifest.content_hash:
        errors.append("recorded predecessor content hash is wrong")
    if manifest.predecessor_seal_hash != candidate_seal_hash(report.seal):
        errors.append("recorded predecessor seal hash is wrong")
    if seal.predecessor_content_hash != report.manifest.content_hash:
        errors.append("seal predecessor content hash is wrong")
    return tuple(errors)


def assert_candidate_version(
    output_dir: str | Path, version: str, settings: GoldenMaintenanceSettings
) -> CandidateManifest:
    """Return the verified manifest or raise :class:`ValidationError`."""
    report = verify_candidate_version(output_dir, version, settings)
    report.assert_ok()
    if report.manifest is None:  # pragma: no cover - guarded by a passing report
        raise ValidationError("verified candidate report has no manifest")
    return report.manifest


def sealed_source_keys(settings: GoldenMaintenanceSettings) -> tuple[str, ...]:
    """Return sealed label source keys for drift comparison, read-only.

    Only ``source_id`` values are read from the sealed labels; transcripts and
    slot values are never copied out of the sealed artifact.

    Args:
        settings: Typed maintenance settings.

    Returns:
        Sorted pseudonymous source keys, e.g. ``call-0001:0``.

    """
    manifest = assert_sealed(settings.sealed_artifact_dir, settings.sealed_version)
    labels = load_labels(
        resolve_path(settings.sealed_artifact_dir) / manifest.labels_path,
        expected_version=manifest.version,
    )
    return tuple(sorted({label.source_id for label in labels}))


def load_candidate_bundle(output_dir: str | Path, version: str) -> ReviewBundle:
    """Read and validate only the review bundle of a candidate version."""
    _, _, candidates_path, _, _ = _paths(output_dir, version)
    _, reviews = load_candidate_document(candidates_path)
    if len(reviews) != 1:
        raise ValidationError("candidate document must carry exactly one review bundle")
    return ReviewBundle.from_dict(reviews[0])
