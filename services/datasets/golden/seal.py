"""Sealing, content hashing and verification of golden-set versions.

A sealed version is three files: canonical labels, a versioned manifest and a
seal file whose manifest hash detects any unaudited manifest edit. The manifest
carries ``content_hash`` over canonical labels plus source IDs, second-review
coverage and a pointer into the append-only access audit log. Sealed files are
never overwritten: changes go through :func:`amend_sealed_set` with a distinct
second reviewer and an audit entry.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from voiceloop_common import ValidationError, utcnow

from .audit import (
    AccessAuditLog,
    AuditPointer,
    canonical_json,
    rel_posix,
    resolve_path,
    sha256_hex,
    utc_timestamp,
)
from .labelling import LabelRecord, ReviewCoverage, SecondReview, compute_coverage
from .settings import GoldenSettings

DEFAULT_SAMPLING_POLICY = (
    "Whole calls are sampled with a seeded RNG (stratified by call, never by "
    "utterance or failure label); no failure-based filtering is applied. "
    "Invariant 6's unbiased random slice is the separate 3-5% of live turns "
    "that bypasses the failure filter so real error rates stay measurable; "
    "golden candidates are drawn from all eligible calls for the same reason."
)


class SealVerificationError(ValidationError):
    """Raised when a sealed golden set fails an integrity check."""


@dataclass(frozen=True)
class GoldenManifest:
    """Versioned manifest for one immutable golden-set artifact."""

    version: str
    dataset_id: str
    content_hash: str
    count: int
    call_ids: tuple[str, ...]
    created_at: str
    second_review_count: int
    second_review_total: int
    second_review_coverage: float
    agreement_count: int
    disagreement_count: int
    audit_pointer: AuditPointer
    labels_path: str
    synthetic: bool
    workflow_stand_in: bool
    review_mode: str
    sampling_policy: str
    predecessor_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of the manifest."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoldenManifest:
        """Rebuild a manifest from its on-disk JSON object."""
        return cls(
            version=str(data["version"]),
            dataset_id=str(data["dataset_id"]),
            content_hash=str(data["content_hash"]),
            count=int(data["count"]),
            call_ids=tuple(str(call_id) for call_id in data["call_ids"]),
            created_at=str(data["created_at"]),
            second_review_count=int(data["second_review_count"]),
            second_review_total=int(data["second_review_total"]),
            second_review_coverage=float(data["second_review_coverage"]),
            agreement_count=int(data["agreement_count"]),
            disagreement_count=int(data["disagreement_count"]),
            audit_pointer=AuditPointer.from_dict(data["audit_pointer"]),
            labels_path=str(data["labels_path"]),
            synthetic=bool(data["synthetic"]),
            workflow_stand_in=bool(data["workflow_stand_in"]),
            review_mode=str(data["review_mode"]),
            sampling_policy=str(data["sampling_policy"]),
            predecessor_hash=data.get("predecessor_hash"),
        )


@dataclass(frozen=True)
class SealFile:
    """Seal over the canonical manifest bytes."""

    version: str
    manifest_path: str
    manifest_hash: str
    content_hash: str
    sealed_at: str
    sealed_by: str
    second_reviewer_id: str
    previous_content_hash: str | None
    immutable: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of the seal."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SealFile:
        """Rebuild a seal from its on-disk JSON object."""
        return cls(
            version=str(data["version"]),
            manifest_path=str(data["manifest_path"]),
            manifest_hash=str(data["manifest_hash"]),
            content_hash=str(data["content_hash"]),
            sealed_at=str(data["sealed_at"]),
            sealed_by=str(data["sealed_by"]),
            second_reviewer_id=str(data["second_reviewer_id"]),
            previous_content_hash=data.get("previous_content_hash"),
            immutable=bool(data.get("immutable", True)),
        )


@dataclass(frozen=True)
class SealedArtifact:
    """Paths and parsed objects returned by a successful seal."""

    manifest: GoldenManifest
    seal: SealFile
    manifest_path: Path
    labels_path: Path
    seal_path: Path
    audit_path: Path


@dataclass(frozen=True)
class VerificationReport:
    """Result of verifying labels, manifest, seal and audit prefix together."""

    ok: bool
    errors: tuple[str, ...] = field(default_factory=tuple)
    manifest: GoldenManifest | None = None
    seal: SealFile | None = None

    def assert_ok(self) -> None:
        """Raise :class:`SealVerificationError` if the seal is broken."""
        if not self.ok:
            raise SealVerificationError("; ".join(self.errors))


def _artifact_paths(output_dir: str | Path, version: str) -> tuple[Path, Path, Path, Path, Path]:
    """Return base dir, manifest, labels, seal and audit paths for a version."""
    base = resolve_path(output_dir)
    return (
        base,
        base / f"{version}.json",
        base / f"{version}.labels.json",
        base / f"{version}.seal.json",
        base / "audit.jsonl",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Atomically-ish write canonical JSON text with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _read_object(path: Path, kind: str, errors: list[str]) -> dict[str, Any] | None:
    """Read one JSON object, appending an error and returning None on failure."""
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read {kind}: {path} ({exc})")
        return None
    if not isinstance(decoded, dict):
        errors.append(f"{kind} must be a JSON object")
        return None
    return decoded


def compute_content_hash(labels: Sequence[LabelRecord]) -> str:
    """Hash every field of every label in ordered canonical JSON.

    The payload covers ``source_id``, ``call_id``, ``turn_index``,
    ``corrected_transcript``, ``slots``, both reviewer IDs and the agreement
    flag. Order is canonical (sorted by ``source_id``), so input order cannot
    change the hash and no single-field label edit can escape detection.
    """
    ordered = sorted(labels, key=lambda label: label.source_id)
    return sha256_hex(canonical_json({"labels": [label.to_dict() for label in ordered]}))


def label_integrity_errors(labels: Sequence[LabelRecord]) -> tuple[str, ...]:
    """Return structural second-review errors for a set of labels."""
    errors: list[str] = []
    for number, label in enumerate(labels, start=1):
        expected_id = f"{label.call_id}:{label.turn_index}"
        if label.source_id != expected_id:
            errors.append(
                f"label {number}: source_id {label.source_id!r} does not match "
                f"{label.call_id}:{label.turn_index}"
            )
        if not label.first_reviewer_id.strip():
            errors.append(f"label {number}: missing first reviewer")
        if not label.second_reviewer_id.strip():
            errors.append(f"label {number}: missing second reviewer")
        if label.first_reviewer_id == label.second_reviewer_id:
            errors.append(f"label {number}: first and second reviewer must differ")
    return tuple(errors)


def write_labels(labels: Sequence[LabelRecord], path: str | Path, *, version: str) -> Path:
    """Write canonical labels JSON for ``version``, sorted by source ID."""
    ordered = sorted(labels, key=lambda label: label.source_id)
    return _write_json(
        resolve_path(path),
        {"version": version, "labels": [label.to_dict() for label in ordered]},
    )


def load_labels(
    path: str | Path, *, expected_version: str | None = None
) -> tuple[LabelRecord, ...]:
    """Read and validate canonical labels written by :func:`write_labels`.

    Args:
        path: Labels document path.
        expected_version: When supplied, the document version must match.

    Raises:
        ValidationError: If the document shape or version is invalid.

    """
    document = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("labels"), list):
        raise ValidationError("labels file must contain a labels array")
    if expected_version is not None and document.get("version") != expected_version:
        raise ValidationError(
            f"labels document version {document.get('version')!r} "
            f"does not match expected version {expected_version!r}"
        )
    return tuple(LabelRecord.from_dict(item) for item in document["labels"])


def manifest_hash(manifest: GoldenManifest) -> str:
    """Return the SHA-256 hash of the canonical manifest bytes."""
    return sha256_hex(canonical_json(manifest.to_dict()))


def seal_golden_set(
    output_dir: str | Path,
    labels: Sequence[LabelRecord],
    *,
    dataset_id: str,
    actor_id: str,
    reviewer_id: str,
    task_id: str,
    purpose: str,
    audit_log: AccessAuditLog | None = None,
    version: str = "golden.v1",
    require_full_second_review: bool = True,
    synthetic: bool = False,
    workflow_stand_in: bool = False,
    review_mode: str = "human_review",
    sampling_policy: str = DEFAULT_SAMPLING_POLICY,
    predecessor_hash: str | None = None,
    created_at: str | None = None,
) -> SealedArtifact:
    """Write labels, manifest and seal for a new immutable version.

    Existing artifacts are never overwritten. A distinct second reviewer is
    required for every label when ``require_full_second_review`` is set, and
    writes/seals are recorded in the access audit log before the pointer is
    embedded in the manifest.

    Args:
        output_dir: Directory holding the sealed version files.
        labels: Human-labelled records with source IDs.
        dataset_id: Stable identifier for this version.
        actor_id: Identity performing the seal (the writer).
        reviewer_id: Distinct second reviewer approving the seal.
        task_id: Roadmap task authorising the change.
        purpose: Human-readable reason for the access.
        audit_log: Optional audit log; defaults to ``<output_dir>/audit.jsonl``.
        version: Version name, e.g. ``golden.v1``.
        require_full_second_review: Refuse to seal without 100% coverage.
        synthetic: Marks synthetic workflow-stand-in content.
        workflow_stand_in: Explicitly marks the artifact as non-production.
        review_mode: ``human_review`` or ``simulated_synthetic_labels``.
        sampling_policy: Text recorded in the manifest for audit.
        predecessor_hash: Content hash of the previous sealed version.
        created_at: Optional UTC timestamp for deterministic tests.

    Returns:
        Paths and parsed objects for the newly sealed artifact.

    Raises:
        ValidationError: On invalid labels, missing second review or existing files.

    """
    if not labels:
        raise ValidationError("cannot seal an empty golden set")
    structural_errors = label_integrity_errors(labels)
    if structural_errors:
        raise ValidationError("; ".join(structural_errors))
    base, manifest_path, labels_path, seal_path, audit_path = _artifact_paths(output_dir, version)
    existing = [path for path in (manifest_path, labels_path, seal_path) if path.exists()]
    if existing:
        raise ValidationError(
            f"refusing to overwrite sealed artifacts: {', '.join(path.name for path in existing)}"
        )
    if not reviewer_id.strip():
        raise ValidationError("a named second reviewer is required to seal")
    first_reviewers = {label.first_reviewer_id for label in labels}
    if reviewer_id in first_reviewers:
        raise ValidationError("sealing reviewer must differ from every primary reviewer")
    if require_full_second_review:
        missing = [label.source_id for label in labels if label.second_reviewer_id != reviewer_id]
        if missing:
            raise ValidationError(
                f"second review from {reviewer_id!r} missing for: {', '.join(missing)}"
            )

    seconds = {
        label.source_id: SecondReview(
            source_id=label.source_id,
            first_reviewer_id=label.first_reviewer_id,
            second_reviewer_id=label.second_reviewer_id,
            transcript_agreed=label.agreed,
            slots_agreed=label.agreed,
            reviewed_at="",
        )
        for label in labels
        if label.second_reviewer_id
    }
    coverage: ReviewCoverage = compute_coverage(labels, seconds)
    content_hash = compute_content_hash(labels)
    log = audit_log or AccessAuditLog(audit_path)
    log.record(
        actor_id=actor_id,
        action="write",
        purpose=purpose,
        task_id=task_id,
        dataset_id=dataset_id,
        version=version,
        content_hash=content_hash,
    )
    log.record(
        actor_id=actor_id,
        action="seal",
        purpose=purpose,
        task_id=task_id,
        dataset_id=dataset_id,
        version=version,
        content_hash=content_hash,
    )
    pointer = AuditPointer(
        path=rel_posix(log.path, base) if log.path.is_relative_to(base) else log.path.as_posix(),
        entries_at_seal=len(log.entries()),
        head_hash=log.head_hash(),
    )
    manifest = GoldenManifest(
        version=version,
        dataset_id=dataset_id,
        content_hash=content_hash,
        count=len(labels),
        call_ids=tuple(sorted({label.call_id for label in labels})),
        created_at=created_at or utc_timestamp(),
        second_review_count=coverage.second_review_count,
        second_review_total=coverage.total,
        second_review_coverage=coverage.ratio,
        agreement_count=coverage.agreement_count,
        disagreement_count=coverage.disagreement_count,
        audit_pointer=pointer,
        labels_path=rel_posix(labels_path, base),
        synthetic=synthetic,
        workflow_stand_in=workflow_stand_in,
        review_mode=review_mode,
        sampling_policy=sampling_policy,
        predecessor_hash=predecessor_hash,
    )
    write_labels(labels, labels_path, version=version)
    _write_json(manifest_path, manifest.to_dict())
    seal = SealFile(
        version=version,
        manifest_path=rel_posix(manifest_path, base),
        manifest_hash=manifest_hash(manifest),
        content_hash=content_hash,
        sealed_at=utcnow().isoformat(timespec="seconds").replace("+00:00", "Z"),
        sealed_by=actor_id,
        second_reviewer_id=reviewer_id,
        previous_content_hash=predecessor_hash,
    )
    _write_json(seal_path, seal.to_dict())
    artifact = SealedArtifact(manifest, seal, manifest_path, labels_path, seal_path, log.path)
    verify_seal(output_dir, version).assert_ok()
    return artifact


def verify_seal(
    output_dir: str | Path,
    version: str = "golden.v1",
    *,
    actor_id: str | None = None,
    purpose: str | None = None,
    task_id: str | None = None,
) -> VerificationReport:
    """Verify labels, manifest, seal and the sealed audit prefix.

    Appended audit entries after sealing are allowed; altering the sealed
    prefix is not. Every label field is content-addressed, and the structural
    second-review invariants are checked explicitly. Verification is read-only
    unless all of ``actor_id``, ``purpose`` and ``task_id`` are supplied, in
    which case one ``verify`` entry is appended to the audit log.

    Args:
        output_dir: Directory holding the sealed version files.
        version: Version name to verify.
        actor_id: Optional identity requesting an audited verification.
        purpose: Optional reason for an audited verification.
        task_id: Optional roadmap task authorising an audited verification.

    Returns:
        A report with all detected integrity errors.

    Raises:
        ValidationError: If only some audit-context fields are supplied.

    """
    audit_context = (actor_id, purpose, task_id)
    if any(field is not None for field in audit_context) and not all(
        isinstance(field, str) for field in audit_context
    ):
        raise ValidationError(
            "audited verification requires actor_id, purpose and task_id together"
        )

    base, manifest_path, _, seal_path, default_audit_path = _artifact_paths(output_dir, version)
    errors: list[str] = []
    manifest: GoldenManifest | None = None
    seal: SealFile | None = None
    manifest_data = _read_object(manifest_path, "manifest", errors)
    seal_data = _read_object(seal_path, "seal", errors)
    if manifest_data is None or seal_data is None:
        audit_path = default_audit_path
    else:
        try:
            manifest = GoldenManifest.from_dict(manifest_data)
            seal = SealFile.from_dict(seal_data)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"malformed manifest or seal: {exc}")
            manifest, seal = None, None
        audit_path = default_audit_path
        if manifest is not None:
            pointer_path = Path(manifest.audit_pointer.path)
            audit_path = base / pointer_path if not pointer_path.is_absolute() else pointer_path

    if manifest is not None and seal is not None:
        if manifest.version != version or seal.version != version:
            errors.append("version mismatch between path, manifest and seal")
        if seal.manifest_hash != manifest_hash(manifest):
            errors.append("manifest was modified after sealing")
        if seal.content_hash != manifest.content_hash:
            errors.append("seal content hash does not match manifest")
        if not seal.immutable:
            errors.append("seal is marked mutable")

        labels: tuple[LabelRecord, ...] = ()
        try:
            labels = load_labels(base / manifest.labels_path, expected_version=version)
        except (OSError, ValueError, TypeError, ValidationError) as exc:
            errors.append(f"cannot read sealed labels: {exc}")
        if compute_content_hash(labels) != manifest.content_hash:
            errors.append("labels were modified after sealing")
        errors.extend(label_integrity_errors(labels))
        for number, label in enumerate(labels, start=1):
            if label.second_reviewer_id != seal.second_reviewer_id:
                errors.append(
                    f"label {number}: second reviewer {label.second_reviewer_id!r} "
                    f"does not match seal reviewer {seal.second_reviewer_id!r}"
                )
        if len(labels) != manifest.count:
            errors.append(f"label count {len(labels)} does not manifest count {manifest.count}")
        if tuple(sorted({label.call_id for label in labels})) != manifest.call_ids:
            errors.append("source call IDs do not match manifest")
        reviewed = sum(1 for label in labels if label.second_reviewer_id)
        agreed = sum(1 for label in labels if label.second_reviewer_id and label.agreed)
        if reviewed != manifest.second_review_count or agreed != manifest.agreement_count:
            errors.append("second-review coverage does not match manifest")
        if labels and manifest.second_review_coverage != reviewed / len(labels):
            errors.append("second-review coverage ratio does not match manifest")

        pointer = manifest.audit_pointer
        try:
            entries = AccessAuditLog(audit_path).entries()
            errors.extend(AccessAuditLog(audit_path).verify_chain())
        except ValidationError as exc:
            entries = ()
            errors.append(f"invalid audit log: {exc}")
        if len(entries) < pointer.entries_at_seal:
            errors.append("audit log is shorter than the sealed pointer")
        elif not entries or entries[pointer.entries_at_seal - 1].entry_hash != pointer.head_hash:
            errors.append("sealed audit prefix was modified")

    if actor_id is not None and purpose is not None and task_id is not None:
        AccessAuditLog(audit_path).record(
            actor_id=actor_id,
            action="verify",
            purpose=purpose,
            task_id=task_id,
            dataset_id=manifest.dataset_id if manifest else "unknown",
            version=version,
            content_hash=manifest.content_hash if manifest else None,
        )
    return VerificationReport(
        ok=not errors,
        errors=tuple(errors),
        manifest=manifest,
        seal=seal,
    )


def assert_sealed(
    output_dir: str | Path,
    version: str = "golden.v1",
    *,
    actor_id: str | None = None,
    purpose: str | None = None,
    task_id: str | None = None,
) -> GoldenManifest:
    """Return the verified manifest or raise :class:`SealVerificationError`."""
    report = verify_seal(
        output_dir,
        version,
        actor_id=actor_id,
        purpose=purpose,
        task_id=task_id,
    )
    report.assert_ok()
    if report.manifest is None:  # pragma: no cover - guaranteed by a passing report
        raise SealVerificationError("verified report has no manifest")
    return report.manifest


def _next_version(version: str) -> str:
    """Return the next ``golden.vN`` version name."""
    prefix, _, suffix = version.rpartition(".")
    if prefix != "golden" or not suffix.isdigit():
        raise ValidationError(f"cannot derive next version from {version!r}")
    return f"{prefix}.{int(suffix) + 1}"


def amend_sealed_set(
    output_dir: str | Path,
    labels: Sequence[LabelRecord],
    *,
    previous_version: str = "golden.v1",
    new_version: str | None = None,
    **kwargs: Any,
) -> SealedArtifact:
    """Seal an amended version after verifying and auditing its predecessor.

    All keyword arguments are forwarded to :func:`seal_golden_set`. The old
    version remains on disk and stays independently verifiable.

    Args:
        output_dir: Directory holding the sealed versions.
        labels: Newly reviewed label records.
        previous_version: Version being amended.
        new_version: New version name; defaults to the next integer version.
        kwargs: Forwarded sealing arguments, including actor and reviewer IDs.

    Returns:
        The newly sealed artifact.

    Raises:
        SealVerificationError: If the predecessor is broken.
        ValidationError: If the amendment is not fully audited or reviewed.

    """
    previous = assert_sealed(output_dir, previous_version)
    target = new_version or _next_version(previous_version)
    if target == previous_version:
        raise ValidationError("amendment requires a new version name")
    base, manifest_path, _, _, audit_path = _artifact_paths(output_dir, target)
    if manifest_path.exists():
        raise ValidationError(f"version already exists: {target}")
    log = AccessAuditLog(audit_path)
    purpose = str(kwargs.get("purpose", "audited golden-set amendment"))
    task_id = str(kwargs.get("task_id", "P1-06"))
    actor_id = str(kwargs.get("actor_id", "unknown"))
    log.record(
        actor_id=actor_id,
        action="amend",
        purpose=purpose,
        task_id=task_id,
        dataset_id=previous.dataset_id,
        version=target,
        content_hash=previous.content_hash,
    )
    forwarded = {**kwargs, "audit_log": log, "predecessor_hash": previous.content_hash}
    forwarded.setdefault("dataset_id", previous.dataset_id)
    return seal_golden_set(base, labels, version=target, **forwarded)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: ``verify`` checks a sealed version and prints JSON."""
    settings = GoldenSettings()
    parser = argparse.ArgumentParser(description="Verify a sealed golden-set version.")
    sub = parser.add_subparsers(dest="command", required=True)
    verify_parser = sub.add_parser("verify", help="Verify labels, manifest, seal and audit.")
    verify_parser.add_argument("--dir", default=settings.artifact_dir)
    verify_parser.add_argument("--version", default="golden.v1")
    verify_parser.add_argument("--actor", required=True, help="Identity requesting verification.")
    verify_parser.add_argument("--purpose", required=True, help="Reason for verification.")
    verify_parser.add_argument("--task-id", required=True, help="Authorising roadmap task ID.")
    args = parser.parse_args(argv)
    report = verify_seal(
        args.dir,
        args.version,
        actor_id=args.actor,
        purpose=args.purpose,
        task_id=args.task_id,
    )
    print(
        canonical_json(
            {
                "ok": report.ok,
                "errors": list(report.errors),
                "version": args.version,
                "dir": args.dir,
            }
        )
    )
    return 0 if report.ok else 2


if __name__ == "__main__":  # pragma: no cover - exercised via python -m
    raise SystemExit(main())
