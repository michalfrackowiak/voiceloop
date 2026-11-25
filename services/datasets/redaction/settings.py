"""Typed settings and injectable infrastructure seams for redaction (P3-10).

Everything in this package is a **synthetic / fake stand-in**. No audio library,
object store, network or speech recognition is used: audio is a deterministic
fake waveform with an explicit sample rate, and all storage/audit/clock
interactions go through the small protocols defined here.

Retention numbers and the pseudonym salt are **engineering placeholders**, not
legal or compliance decisions. ``docs/compliance.md`` remains
``OPEN — counsel review required``; no numeric retention period has been
approved by a human.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import Field, field_validator

from voiceloop_common import BaseSettings, settings_config, utcnow

# ENGINEERING DEFAULTS ONLY — NOT LEGAL OR COMPLIANCE DECISIONS.
# The human data protection policy owner (docs/compliance.md section 5/6) must
# set real values before any production use. Status stays OPEN pending counsel.
ENGINEERING_DEFAULT_PADDING_MS = 50
ENGINEERING_DEFAULT_RETENTION_DAYS = 30
ENGINEERING_DEFAULT_PSEUDONYM_PREFIX = "PERSON"
ENGINEERING_DEFAULT_PSEUDONYM_SALT = "voiceloop-engineering-placeholder-salt"
ENGINEERING_DEFAULT_FIXTURES_PATH = "tests/fixtures/turns.jsonl"
ENGINEERING_DEFAULT_OUTPUT_DIR = "services/datasets/redaction/artifacts"

POLICY_STATUS = "ENGINEERING_PLACEHOLDER_NOT_A_LEGAL_DECISION"
COMPLIANCE_STATUS = "OPEN - counsel review required"
SYNTHETIC_LABEL = "SYNTHETIC/FAKE STAND-IN - no real audio, store, or deletion"


class RedactionSettings(BaseSettings):
    """Typed environment-backed settings with prefix ``VOICELOOP_REDACTION_``.

    All defaults are engineering placeholders for local/synthetic rehearsals.
    The salt is not a secret in the repo: operators must override it for any
    non-synthetic deployment, but doing so does not constitute a compliance
    decision.
    """

    model_config = settings_config("VOICELOOP_REDACTION_")

    padding_ms: int = Field(
        default=ENGINEERING_DEFAULT_PADDING_MS,
        ge=0,
        description="Conservative padding added on both sides of every PII audio span.",
    )
    retention_days: int = Field(
        default=ENGINEERING_DEFAULT_RETENTION_DAYS,
        gt=0,
        description="Engineering placeholder maximum age of RAW audio before fake deletion.",
    )
    pseudonym_prefix: str = Field(
        default=ENGINEERING_DEFAULT_PSEUDONYM_PREFIX,
        min_length=1,
        description="Uppercase tag used inside salted person-name pseudonyms.",
    )
    pseudonym_salt: str = Field(
        default=ENGINEERING_DEFAULT_PSEUDONYM_SALT,
        min_length=1,
        description="Per-call salt namespace; override outside synthetic rehearsals.",
    )
    fixtures_path: str = Field(
        default=ENGINEERING_DEFAULT_FIXTURES_PATH,
        description="Read-only synthetic labels manifest used by the CLI.",
    )
    output_dir: str = Field(
        default=ENGINEERING_DEFAULT_OUTPUT_DIR,
        description="Directory for CLI JSON/JSONL reports (never the read-only fixtures).",
    )

    @field_validator("pseudonym_prefix")
    @classmethod
    def _validate_prefix(cls, value: str) -> str:
        """Keep pseudonym tokens free of punctuation and lower-case text."""
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", value):
            raise ValueError("pseudonym_prefix must match [A-Z][A-Z0-9_]*")
        return value


class Clock(Protocol):
    """Injectable UTC clock for deterministic retention reports."""

    def now(self) -> datetime:
        """Return the current timezone-aware UTC time."""
        ...


class SystemClock:
    """Clock backed by the shared injectable ``voiceloop_common.utcnow()``."""

    def now(self) -> datetime:
        """Return the current UTC time."""
        return utcnow()


@dataclass(frozen=True, slots=True)
class FixedClock:
    """Deterministic fake clock that always reports one instant."""

    value: datetime

    def now(self) -> datetime:
        """Return the fixed instant, normalised to UTC."""
        if self.value.tzinfo is None:
            return self.value.replace(tzinfo=UTC)
        return self.value.astimezone(UTC)


AuditDetailValue = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """PII-free audit entry: event type, time and scalar details only.

    Details must never contain transcript text, matched PII values, original
    names, or audio bytes. Only URIs, offsets, counts, categories and policy
    labels are permitted.
    """

    event_type: str
    occurred_at: datetime
    details: Mapping[str, AuditDetailValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe, sorted representation for reports/logs."""
        return {
            "event_type": self.event_type,
            "occurred_at": self.occurred_at.isoformat(),
            "details": {key: self.details[key] for key in sorted(self.details)},
        }


class AuditSink(Protocol):
    """Injectable audit sink; implementations are in-memory fakes by default."""

    def emit(self, event: AuditEvent) -> None:
        """Record one PII-free audit event."""
        ...


@dataclass(slots=True)
class InMemoryAuditSink:
    """Fake in-process audit sink for tests and synthetic CLI runs."""

    events: list[AuditEvent] = field(default_factory=list)

    def emit(self, event: AuditEvent) -> None:
        """Append an audit event to the in-memory list."""
        self.events.append(event)

    def to_dicts(self) -> list[dict[str, Any]]:
        """Return all audit events as sorted JSON-safe dictionaries."""
        return [event.to_dict() for event in self.events]
