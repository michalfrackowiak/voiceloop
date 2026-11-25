"""Typed settings and safe helpers for P3-09 golden-set maintenance.

Every configuration field reads an environment variable prefixed with
``VOICELOOP_GOLDEN_MAINT_``. This package never writes to the sealed P1-06
artifact directory: all candidate versions, review audits and drift reports go
to a separately guarded output path. Only synthetic fixture stand-ins are
supported; there are no real reviewers, live traffic or network seams here.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Final

from pydantic import Field, field_validator, model_validator

from voiceloop_common import BaseSettings, ValidationError, settings_config, utcnow

SEALED_CONTENT_HASH: Final = "662cdaf43ee2d5fde03faf5e69fde196114d4f312406d94c00bf53714d03da8e"
"""Committed P1-06 sealed synthetic golden stand-in content hash (read-only)."""

FIXTURE_SHA256: Final = "aca23a303954d035245a97f78319df3694d63611b717a2ff2131bc2e0a4453fd"
"""Committed SHA-256 of ``tests/fixtures/turns.jsonl`` (read-only)."""

ARTIFACT_LABEL: Final = "SYNTHETIC/FIXTURE STAND-IN - NOT A HUMAN-REVIEWED GOLDEN SET"
"""Mandatory label on every artifact, report and audit entry produced here."""

REPO_ROOT: Final = Path(__file__).resolve().parents[4]

_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_VERSION_RE = re.compile(r"^candidate\.v[1-9][0-9]*$")
_SEALED_VERSION_RE = re.compile(r"^golden\.v[1-9][0-9]*$")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_TEXT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.:;/()-]*$")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<!\w)(?:\+\d[\d .()\-]{6,}\d|\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4})(?!\w)")
_CARD_RE = re.compile(r"(?<!\w)(?:\d[ -]?){12,}\d(?!\w)")


def canonical_json(payload: Any) -> str:
    """Return canonical UTF-8 JSON text with sorted keys and no optional spaces."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    """Return the lowercase SHA-256 hex digest of UTF-8 encoded text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 hex digest of a file's raw bytes."""
    digest = hashlib.sha256()
    with resolve_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_timestamp() -> str:
    """Return the injectable current UTC time as an RFC 3339 ``Z`` string."""
    return utcnow().isoformat(timespec="seconds").replace("+00:00", "Z")


def resolve_path(path: str | Path) -> Path:
    """Resolve a repo-relative path against the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def pseudonymous_id(prefix: str, value: str) -> str:
    """Return a stable pseudonym built only from a prefix and a SHA-256 digest."""
    assert_safe_id(prefix, "prefix")
    return f"{prefix}-{sha256_text(value)}"


def contains_pii_pattern(text: str) -> bool:
    """Return whether text looks like an email, phone number or card-like run."""
    return any(pattern.search(text) is not None for pattern in (_EMAIL_RE, _PHONE_RE, _CARD_RE))


def assert_safe_text(value: str, field: str) -> None:
    """Reject free text that could carry PII, credentials or control characters.

    Args:
        value: Text to screen before it reaches an artifact, audit entry or log.
        field: Field name used in a generic error message (the value is not echoed).

    Raises:
        ValidationError: If the value is blank, outside the allowlisted alphabet
            or matches an email/phone/card pattern.

    """
    if not value or not _SAFE_TEXT_RE.fullmatch(value):
        raise ValidationError(f"{field} contains disallowed characters")
    if contains_pii_pattern(value):
        raise ValidationError(f"{field} failed PII screening")


def assert_safe_id(value: str, field: str) -> None:
    """Reject non-pseudonymous or malformed actor/reviewer identifiers."""
    if not _ID_RE.fullmatch(value):
        raise ValidationError(f"{field} must be a short pseudonymous identifier")


def assert_valid_hash(value: str, field: str) -> None:
    """Require an exact lowercase SHA-256 hex digest."""
    if not _HASH_RE.fullmatch(value):
        raise ValidationError(f"{field} must be a lowercase SHA-256 hex digest")


def assert_version(value: str, field: str = "version") -> None:
    """Require a ``candidate.vN`` version name with a positive integer N."""
    if not _VERSION_RE.fullmatch(value):
        raise ValidationError(f"{field} must match candidate.vN with N >= 1")


def version_number(value: str) -> int:
    """Return the integer N from a validated ``candidate.vN`` version."""
    assert_version(value)
    return int(value.rsplit(".v", 1)[1])


def next_version(value: str) -> str:
    """Return the next ``candidate.vN`` version after a validated version."""
    return f"candidate.v{version_number(value) + 1}"


def assert_outside_sealed(path: str | Path, settings: GoldenMaintenanceSettings) -> None:
    """Refuse any path inside or equal to the sealed golden artifact directory.

    Args:
        path: Candidate output, audit or report destination.
        settings: Typed maintenance settings supplying the sealed directory.

    Raises:
        ValidationError: If the destination could overwrite sealed artifacts.

    """
    destination = resolve_path(path).resolve()
    sealed = resolve_path(settings.sealed_artifact_dir).resolve()
    if destination == sealed or sealed in destination.parents:
        raise ValidationError("maintenance output must stay outside the sealed golden artifacts")


class GoldenMaintenanceSettings(BaseSettings):
    """Fixture-only configuration for audited golden-set extension rehearsals.

    All fields read ``VOICELOOP_GOLDEN_MAINT_*`` environment variables. Defaults
    point at read-only synthetic fixtures and a maintenance-owned output folder.
    """

    model_config = settings_config("VOICELOOP_GOLDEN_MAINT_")

    fixtures_path: str = "tests/fixtures/turns.jsonl"
    sealed_artifact_dir: str = "services/datasets/golden/artifacts"
    sealed_version: str = "golden.v1"
    expected_sealed_content_hash: str = SEALED_CONTENT_HASH
    expected_fixture_sha256: str = FIXTURE_SHA256
    output_dir: str = "services/datasets/golden/maintenance/artifacts"
    candidate_version: str = "candidate.v1"
    candidate_calls: int = Field(default=2, ge=1)
    candidate_seed: int = 11
    drift_calls: int = Field(default=2, ge=1)
    drift_seed: int = 23
    first_reviewer_id: str = "synthetic-reviewer-a"
    second_reviewer_id: str = "synthetic-reviewer-b"
    allowed_actor_ids: tuple[str, ...] = ("data-steward",)
    review_reason: str = "scripted synthetic review; not a human reviewer"
    task_id: str = "P3-09"
    artifact_label: str = ARTIFACT_LABEL

    @field_validator("expected_sealed_content_hash", "expected_fixture_sha256")
    @classmethod
    def _hashes_are_sha256(cls, value: str) -> str:
        """Require pinned source hashes to be exact lowercase SHA-256 digests."""
        assert_valid_hash(value, "expected hash")
        return value

    @field_validator("first_reviewer_id", "second_reviewer_id")
    @classmethod
    def _reviewers_are_pseudonyms(cls, value: str) -> str:
        """Require short pseudonymous scripted reviewer IDs."""
        assert_safe_id(value, "reviewer_id")
        return value

    @field_validator("allowed_actor_ids", mode="before")
    @classmethod
    def _parse_actor_ids(cls, value: Any) -> tuple[str, ...]:
        """Accept a comma-separated env string or an explicit tuple/list."""
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        return tuple(str(item) for item in value)

    @field_validator("allowed_actor_ids")
    @classmethod
    def _actor_ids_are_pseudonyms(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require a non-empty, unique allowlist of short pseudonymous actor IDs."""
        if not value:
            raise ValidationError("allowed_actor_ids must contain at least one actor")
        if len(set(value)) != len(value):
            raise ValidationError("allowed_actor_ids must not contain duplicates")
        for actor_id in value:
            assert_safe_id(actor_id, "allowed_actor_ids")
        return value

    @field_validator("review_reason", "task_id", "artifact_label")
    @classmethod
    def _free_text_is_safe(cls, value: str) -> str:
        """Screen fixed reason/label text before it can reach an audit entry."""
        assert_safe_text(value, "free text")
        return value

    @field_validator("candidate_version")
    @classmethod
    def _version_is_valid(cls, value: str) -> str:
        """Require a versioned candidate name, never an in-place amendment."""
        assert_version(value)
        return value

    @field_validator("sealed_version")
    @classmethod
    def _sealed_version_is_valid(cls, value: str) -> str:
        """Require a ``golden.vN`` read-only predecessor version."""
        if not _SEALED_VERSION_RE.fullmatch(value):
            raise ValidationError("sealed_version must match golden.vN with N >= 1")
        return value

    @model_validator(mode="after")
    def _settings_are_consistent(self) -> GoldenMaintenanceSettings:
        """Enforce distinct reviewers, the mandatory label and path separation."""
        if self.first_reviewer_id == self.second_reviewer_id:
            raise ValidationError("scripted reviewers must have distinct identifiers")
        if self.artifact_label != ARTIFACT_LABEL:
            raise ValidationError("artifact_label must be the mandatory synthetic stand-in label")
        reviewers = {self.first_reviewer_id, self.second_reviewer_id}
        if reviewers.intersection(self.allowed_actor_ids):
            raise ValidationError("proposal actors must not be scripted reviewers")
        assert_outside_sealed(self.output_dir, self)
        return self


def assert_allowed_actor(actor_id: str, settings: GoldenMaintenanceSettings) -> None:
    """Require a pseudonymous proposal actor from the trusted, non-reviewer allowlist.

    Args:
        actor_id: Actor supplied by the CLI or a public API caller.
        settings: Typed settings with the trusted actor and reviewer pins.

    Raises:
        ValidationError: If the actor is malformed, unlisted or also a reviewer.

    """
    assert_safe_id(actor_id, "actor_id")
    if actor_id in (settings.first_reviewer_id, settings.second_reviewer_id):
        raise ValidationError("proposal actor must not be a scripted reviewer")
    if actor_id not in settings.allowed_actor_ids:
        raise ValidationError("actor_id is not in the configured allowlist")
