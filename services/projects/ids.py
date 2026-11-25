"""Validated pseudonymous project identifiers for the isolated P4-05 seam.

SYNTHETIC/FIXTURE STAND-IN - NOT A MULTI-PROJECT PRODUCTION GUARANTEE.
Internal artefact label only; no cross-service contract version bump is implied
or performed. Error messages are fixed-code strings and never echo the rejected
value or a caller-supplied field name, so PII cannot leak into logs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PROJECT_ID_MIN_LENGTH = 2
PROJECT_ID_MAX_LENGTH = 64

_SLUG_RE = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+")
_PHONE_RE = re.compile(r"(?:\+\d|\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4})")
_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,}\d(?!\d)")
_LONG_DIGIT_RUN_RE = re.compile(r"\d{8,}")

_DEFAULT_FIELD = "project_id"
_FIELD_SLUG_RE = re.compile(r"[a-z][a-z0-9_]*")
_FIELD_MAX_LENGTH = 64
# Only field names used by this package may reach an error message. Anything
# else (including a caller-supplied email or card number) falls back to the
# fixed default, so refusal codes stay stable without echoing input text.
_SAFE_FIELDS: frozenset[str] = frozenset(
    {
        "alias",
        "alpha_project_id",
        "artifact_name",
        "beta_project_id",
        "call_id",
        "dataset_name",
        "default_project_id",
        "eval_set_name",
        "extension",
        "model_name",
        "project_id",
        "report_name",
        "turn_index",
        "version",
    }
)


class InvalidProjectIdError(ValueError):
    """Raised when a project identifier is missing or fails pseudonym validation."""


def _safe_field(field: object) -> str:
    """Return a known-safe field slug or the fixed default field name."""
    if not isinstance(field, str) or len(field) > _FIELD_MAX_LENGTH:
        return _DEFAULT_FIELD
    if field not in _SAFE_FIELDS or _FIELD_SLUG_RE.fullmatch(field) is None:
        return _DEFAULT_FIELD
    return field


def validate_project_id(value: object, *, field: str = "project_id") -> str:
    """Return a validated lowercase pseudonymous slug.

    Args:
        value: Candidate identifier. Anything that is not a string is rejected.
        field: Candidate field name for generic error messages. Only an
            internal allowlisted slug is used; anything else falls back to
            ``"project_id"``, so caller-supplied text is never echoed.

    Returns:
        The validated slug.

    Raises:
        InvalidProjectIdError: If the value is missing, malformed, or looks like
            a name/email/phone/card rather than a pseudonym.

    """
    field = _safe_field(field)
    if not isinstance(value, str) or not value:
        raise InvalidProjectIdError(f"{field} is required")
    if not (PROJECT_ID_MIN_LENGTH <= len(value) <= PROJECT_ID_MAX_LENGTH):
        raise InvalidProjectIdError(f"{field} length is outside the allowed range")
    if _EMAIL_RE.search(value):
        raise InvalidProjectIdError(f"{field} must not contain email-like text")
    if _CARD_RE.search(value) or _PHONE_RE.search(value) or _LONG_DIGIT_RUN_RE.search(value):
        raise InvalidProjectIdError(f"{field} must not contain phone/card-like digit runs")
    if _SLUG_RE.fullmatch(value) is None:
        raise InvalidProjectIdError(f"{field} must be a lowercase pseudonymous slug")
    return value


@dataclass(frozen=True, slots=True, order=True)
class ProjectId:
    """Immutable validated pseudonymous project slug."""

    value: str

    def __post_init__(self) -> None:
        """Validate the slug at construction time, so no invalid ID can exist."""
        object.__setattr__(self, "value", validate_project_id(self.value))

    def __str__(self) -> str:
        """Return the bare pseudonymous slug."""
        return self.value

    @classmethod
    def parse(cls, value: str | ProjectId | None, *, field: str = "project_id") -> ProjectId:
        """Coerce a candidate into a validated project ID, failing closed.

        Args:
            value: A validated ID, a candidate string, or ``None``.
            field: Fixed field name for a generic error message.

        Returns:
            The validated project ID.

        Raises:
            InvalidProjectIdError: If the candidate is missing or invalid.

        """
        if isinstance(value, cls):
            # Re-validate even an existing instance: a tampered ID must not
            # become a pass-through that bypasses the fail-closed guard.
            return cls(validate_project_id(value.value, field=field))
        return cls(validate_project_id(value, field=field))
