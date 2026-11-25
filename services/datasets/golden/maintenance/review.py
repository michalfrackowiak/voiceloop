"""Audited two-reviewer flow for synthetic extension candidates.

Both reviewers are deterministic scripted stand-ins with distinct pseudonymous
actor IDs. Every review carries a reason, an injected UTC timestamp and a
SHA-256 answer hash, and is appended to a hash-chained maintenance audit log.
Verification fails closed on a single reviewer, a missing audit entry, a
disagreement, or any tampering with the audit prefix. This module has no API for
editing the sealed P1-06 artifacts.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from voiceloop_common import ValidationError

from .candidates import CandidateRecord, CandidateSet, candidate_content_hash
from .settings import (
    ARTIFACT_LABEL,
    GoldenMaintenanceSettings,
    assert_allowed_actor,
    assert_safe_id,
    assert_safe_text,
    assert_valid_hash,
    assert_version,
    canonical_json,
    resolve_path,
    sha256_text,
    utc_timestamp,
)

ReviewAction = Literal[
    "candidate_build", "first_review", "second_review", "version_write", "version_seal"
]
Decision = Literal["include"]
_CRITERIA = "fixture-standin-review-criteria-v1"


@dataclass(frozen=True)
class AuditEntry:
    """One append-only maintenance audit record with an integrity hash."""

    actor_id: str
    action: ReviewAction
    reason: str
    task_id: str
    timestamp: str
    version: str
    content_hash: str
    label: str = ARTIFACT_LABEL
    previous_hash: str = ""
    entry_hash: str = ""

    def payload(self) -> dict[str, Any]:
        """Return the fields covered by ``entry_hash``."""
        data = asdict(self)
        data.pop("entry_hash")
        return data

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of the entry."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AuditEntry:
        """Rebuild an entry from an audit JSONL object."""
        label = str(data.get("label", ARTIFACT_LABEL))
        if label != ARTIFACT_LABEL:
            raise ValidationError("audit entry has the wrong synthetic stand-in label")
        try:
            return cls(
                actor_id=str(data["actor_id"]),
                action=data["action"],
                reason=str(data["reason"]),
                task_id=str(data["task_id"]),
                timestamp=str(data["timestamp"]),
                version=str(data["version"]),
                content_hash=str(data["content_hash"]),
                label=label,
                previous_hash=str(data.get("previous_hash", "")),
                entry_hash=str(data.get("entry_hash", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"malformed audit entry: {exc}") from exc


class ReviewAuditLog:
    """Append-only JSONL audit log with a SHA-256 hash chain."""

    def __init__(self, path: str | Path) -> None:
        """Initialise the log at ``path``; the file appears on first write."""
        self.path = resolve_path(path)

    def entries(self) -> tuple[AuditEntry, ...]:
        """Return all entries in file order, or an empty tuple if absent."""
        if not self.path.exists():
            return ()
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValidationError("cannot read the maintenance audit log") from exc
        found: list[AuditEntry] = []
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"audit line {number} is not valid JSON") from exc
            if not isinstance(decoded, dict):
                raise ValidationError(f"audit line {number} must be an object")
            found.append(AuditEntry.from_dict(decoded))
        return tuple(found)

    def head_hash(self) -> str:
        """Return the newest entry hash, or an empty string for an empty log."""
        entries = self.entries()
        return entries[-1].entry_hash if entries else ""

    def record(
        self,
        *,
        actor_id: str,
        action: ReviewAction,
        reason: str,
        task_id: str,
        version: str,
        content_hash: str,
        timestamp: str | None = None,
    ) -> AuditEntry:
        """Append one audited action and return the persisted entry.

        Args:
            actor_id: Pseudonymous scripted actor or operator ID.
            action: ``candidate_build``, ``first_review``, ``second_review``,
                ``version_write`` or ``version_seal``.
            reason: PII-screened reason for the action.
            task_id: Roadmap task authorising the action.
            version: Candidate version being maintained.
            content_hash: SHA-256 digest of the object being acted on.
            timestamp: Explicit submission time to bind a review answer to its
                audit entry; defaults to the injectable current UTC time.

        Returns:
            The entry with its chain and entry hashes.

        Raises:
            ValidationError: If any field fails identity, reason or hash checks.

        """
        assert_safe_id(actor_id, "actor_id")
        assert_safe_text(reason, "reason")
        assert_safe_text(task_id, "task_id")
        assert_version(version)
        assert_valid_hash(content_hash, "content_hash")
        entry_timestamp = timestamp if timestamp is not None else utc_timestamp()
        assert_safe_text(entry_timestamp, "timestamp")
        entry = AuditEntry(
            actor_id=actor_id,
            action=action,
            reason=reason,
            task_id=task_id,
            timestamp=entry_timestamp,
            version=version,
            content_hash=content_hash,
            previous_hash=self.head_hash(),
        )
        entry = replace(entry, entry_hash=sha256_text(canonical_json(entry.payload())))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(entry.to_dict()) + "\n")
        return entry

    def verify_chain(self) -> tuple[str, ...]:
        """Return integrity errors; an empty tuple means the chain is intact."""
        errors: list[str] = []
        previous = ""
        for number, entry in enumerate(self.entries(), start=1):
            if entry.entry_hash != sha256_text(canonical_json(entry.payload())):
                errors.append(f"audit entry {number}: hash mismatch")
            if entry.previous_hash != previous:
                errors.append(f"audit entry {number}: broken chain")
            previous = entry.entry_hash
        return tuple(errors)


@dataclass(frozen=True)
class ReviewSubmission:
    """One scripted reviewer's PII-free answer to one candidate."""

    candidate_id: str
    reviewer_id: str
    decision: Decision
    reason: str
    submitted_at: str
    answer_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of the submission."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReviewSubmission:
        """Rebuild and validate a submission from a maintained document."""
        try:
            submission = cls(
                candidate_id=str(data["candidate_id"]),
                reviewer_id=str(data["reviewer_id"]),
                decision=data["decision"],
                reason=str(data["reason"]),
                submitted_at=str(data["submitted_at"]),
                answer_hash=str(data["answer_hash"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"malformed review submission: {exc}") from exc
        if submission.decision != "include":
            raise ValidationError("scripted review decision must be include")
        assert_safe_id(submission.reviewer_id, "reviewer_id")
        assert_safe_text(submission.reason, "reason")
        assert_valid_hash(submission.answer_hash, "answer_hash")
        if not submission.submitted_at:
            raise ValidationError("review submission has no timestamp")
        if submission.answer_hash != answer_digest(submission.candidate_id, submission.decision):
            raise ValidationError("review answer hash does not match the decision")
        return submission


@dataclass(frozen=True)
class ReviewBundle:
    """Two distinct scripted reviews per candidate, with audited provenance."""

    candidate_hash: str
    first_reviewer_id: str
    second_reviewer_id: str
    pairs: tuple[tuple[ReviewSubmission, ReviewSubmission], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of the review bundle."""
        return {
            "label": ARTIFACT_LABEL,
            "candidate_hash": self.candidate_hash,
            "first_reviewer_id": self.first_reviewer_id,
            "second_reviewer_id": self.second_reviewer_id,
            "pairs": [[first.to_dict(), second.to_dict()] for first, second in self.pairs],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReviewBundle:
        """Rebuild and validate a review bundle from a maintained document."""
        if data.get("label") != ARTIFACT_LABEL:
            raise ValidationError("review bundle has the wrong synthetic stand-in label")
        pairs_data = data.get("pairs")
        if not isinstance(pairs_data, list) or not all(
            isinstance(pair, list) and len(pair) == 2 for pair in pairs_data
        ):
            raise ValidationError("review bundle has malformed pairs")
        try:
            pairs = tuple(
                (ReviewSubmission.from_dict(pair[0]), ReviewSubmission.from_dict(pair[1]))
                for pair in pairs_data
            )
            result = cls(
                candidate_hash=str(data["candidate_hash"]),
                first_reviewer_id=str(data["first_reviewer_id"]),
                second_reviewer_id=str(data["second_reviewer_id"]),
                pairs=pairs,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"malformed review bundle: {exc}") from exc
        assert_valid_hash(result.candidate_hash, "candidate_hash")
        return result


def answer_digest(candidate_id: str, decision: Decision) -> str:
    """Return the deterministic SHA-256 hash of a scripted review answer."""
    return sha256_text(
        canonical_json({"candidate_id": candidate_id, "decision": decision, "criteria": _CRITERIA})
    )


@runtime_checkable
class ScriptedReviewer(Protocol):
    """Boundary for a deterministic, non-human reviewer stand-in."""

    @property
    def reviewer_id(self) -> str:
        """Return the reviewer's pseudonymous actor ID."""
        ...

    def review(self, record: CandidateRecord) -> ReviewSubmission:
        """Return one scripted answer for a candidate."""
        ...


class FixedDecisionReviewer:
    """Scripted reviewer that always returns one decision with a fixed reason."""

    def __init__(self, reviewer_id: str, reason: str, decision: Decision = "include") -> None:
        """Initialise a pseudonymous reviewer with a PII-screened reason."""
        assert_safe_id(reviewer_id, "reviewer_id")
        assert_safe_text(reason, "reason")
        self._reviewer_id = reviewer_id
        self._reason = reason
        self._decision = decision

    @property
    def reviewer_id(self) -> str:
        """Return the reviewer's pseudonymous actor ID."""
        return self._reviewer_id

    def review(self, record: CandidateRecord) -> ReviewSubmission:
        """Return a timestamped, hashed include answer for one candidate."""
        return ReviewSubmission(
            candidate_id=record.candidate_id,
            reviewer_id=self.reviewer_id,
            decision=self._decision,
            reason=self._reason,
            submitted_at=utc_timestamp(),
            answer_hash=answer_digest(record.candidate_id, self._decision),
        )


def record_candidate_build(
    audit_log: ReviewAuditLog,
    candidate_set: CandidateSet,
    settings: GoldenMaintenanceSettings,
    *,
    actor_id: str,
    purpose: str,
) -> AuditEntry:
    """Append the audited build action that precedes any review."""
    assert_allowed_actor(actor_id, settings)
    assert_safe_text(purpose, "purpose")
    return audit_log.record(
        actor_id=actor_id,
        action="candidate_build",
        reason=purpose,
        task_id=settings.task_id,
        version=candidate_set.version,
        content_hash=candidate_set.content_hash,
    )


def run_scripted_review(
    candidate_set: CandidateSet,
    settings: GoldenMaintenanceSettings,
    audit_log: ReviewAuditLog,
) -> ReviewBundle:
    """Run two distinct scripted reviewers and audit both answers per candidate.

    A disagreement, duplicate reviewer or missing identity fails closed before
    any version file is written.

    Args:
        candidate_set: Separate synthetic extension candidate set.
        settings: Typed settings supplying the two reviewer identities.
        audit_log: Maintenance audit log to append to.

    Returns:
        The bundle of paired review submissions.

    Raises:
        ValidationError: If reviewers are not distinct or answers disagree.

    """
    if settings.first_reviewer_id == settings.second_reviewer_id:
        raise ValidationError("two distinct scripted reviewers are required")
    first = FixedDecisionReviewer(settings.first_reviewer_id, settings.review_reason)
    second = FixedDecisionReviewer(settings.second_reviewer_id, settings.review_reason)
    pairs: list[tuple[ReviewSubmission, ReviewSubmission]] = []
    for record in candidate_set.records:
        primary = first.review(record)
        secondary = second.review(record)
        if primary.reviewer_id == secondary.reviewer_id:
            raise ValidationError("the same scripted reviewer cannot review twice")
        if (
            primary.candidate_id != record.candidate_id
            or secondary.candidate_id != record.candidate_id
        ):
            raise ValidationError("review submissions must match one candidate")
        if primary.decision != secondary.decision:
            raise ValidationError("scripted reviewers disagree; no version may be sealed")
        pairs_of_actions: tuple[tuple[ReviewSubmission, ReviewAction], ...] = (
            (primary, "first_review"),
            (secondary, "second_review"),
        )
        for submission, action in pairs_of_actions:
            audit_log.record(
                actor_id=submission.reviewer_id,
                action=action,
                reason=submission.reason,
                task_id=settings.task_id,
                version=candidate_set.version,
                content_hash=submission.answer_hash,
                timestamp=submission.submitted_at,
            )
        pairs.append((primary, secondary))
    return ReviewBundle(
        candidate_hash=candidate_set.content_hash,
        first_reviewer_id=first.reviewer_id,
        second_reviewer_id=second.reviewer_id,
        pairs=tuple(pairs),
    )


def verify_review_bundle(
    bundle: ReviewBundle,
    records: Sequence[CandidateRecord],
    audit_log: ReviewAuditLog,
    settings: GoldenMaintenanceSettings,
) -> tuple[str, ...]:
    """Return errors if two-reviewer, agreement and audit requirements are unmet.

    Args:
        bundle: Paired scripted submissions from the candidate document.
        records: Candidates in the same version.
        audit_log: Maintenance audit log whose prefix was sealed.
        settings: Typed settings with expected reviewers and task ID.

    Returns:
        Integrity errors; an empty tuple means the review is approved.

    """
    errors: list[str] = list(audit_log.verify_chain())
    expected_ids = (settings.first_reviewer_id, settings.second_reviewer_id)
    if settings.first_reviewer_id == settings.second_reviewer_id:
        errors.append("scripted reviewers are not distinct")
    if bundle.first_reviewer_id != expected_ids[0] or bundle.second_reviewer_id != expected_ids[1]:
        errors.append("review bundle reviewer IDs do not match the configured stand-ins")
    if bundle.first_reviewer_id == bundle.second_reviewer_id:
        errors.append("review bundle has a single reviewer")
    by_id = {record.candidate_id: record for record in records}
    primary_ids = [primary.candidate_id for primary, _ in bundle.pairs]
    if len(bundle.pairs) != len(records) or not records:
        errors.append("review bundle does not cover every candidate exactly once")
    if len(set(primary_ids)) != len(primary_ids):
        errors.append("review bundle contains duplicate candidate pairs")
    if set(primary_ids) != set(by_id):
        errors.append("review bundle candidate IDs do not match the recorded candidate set")
    if bundle.candidate_hash != candidate_content_hash(records):
        errors.append("review bundle candidate hash does not match the candidate set")

    builds = [
        entry
        for entry in audit_log.entries()
        if entry.action == "candidate_build"
        and entry.content_hash == bundle.candidate_hash
        and entry.version == settings.candidate_version
    ]
    if not builds:
        errors.append("candidate build is missing from the audit log")

    for number, pair in enumerate(bundle.pairs, start=1):
        primary, secondary = pair
        if primary.candidate_id not in by_id:
            errors.append(f"review pair {number}: unknown candidate")
            continue
        if secondary.candidate_id != primary.candidate_id:
            errors.append(f"review pair {number}: reviewers answered different candidates")
        if primary.reviewer_id != expected_ids[0] or secondary.reviewer_id != expected_ids[1]:
            errors.append(f"review pair {number}: reviewer IDs do not match the stand-ins")
        if primary.reviewer_id == secondary.reviewer_id:
            errors.append(f"review pair {number}: single reviewer")
        if primary.decision != secondary.decision:
            errors.append(f"review pair {number}: reviewers disagree")
        if primary.reason != settings.review_reason or secondary.reason != settings.review_reason:
            errors.append(f"review pair {number}: review reason is not the scripted reason")
        pairs_of_actions: tuple[tuple[ReviewSubmission, ReviewAction], ...] = (
            (primary, "first_review"),
            (secondary, "second_review"),
        )
        for submission, action in pairs_of_actions:
            matches = [
                entry
                for entry in audit_log.entries()
                if entry.action == action
                and entry.actor_id == submission.reviewer_id
                and entry.content_hash == submission.answer_hash
                and entry.timestamp == submission.submitted_at
                and entry.reason == submission.reason
                and entry.version == settings.candidate_version
                and entry.task_id == settings.task_id
            ]
            if not matches:
                errors.append(
                    f"review pair {number}: audited {action} missing for {submission.reviewer_id}"
                )
    return tuple(errors)


def assert_review_approved(
    bundle: ReviewBundle,
    records: Sequence[CandidateRecord],
    audit_log: ReviewAuditLog,
    settings: GoldenMaintenanceSettings,
) -> None:
    """Raise if the audited second-review flow is incomplete or tampered with."""
    errors = verify_review_bundle(bundle, records, audit_log, settings)
    if errors:
        raise ValidationError("review not approved: " + "; ".join(errors))
