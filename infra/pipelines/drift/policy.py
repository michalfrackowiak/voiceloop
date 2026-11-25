"""P4-02.2 trigger policy: data volume AND (drift OR audited manual override).

A scheduled/calendar tick is recorded for audit but is never a trigger. The
policy deliberately differs from the broader P2-10 OR policy; see the drift
README/handoff for the scope decision. No training or MLflow call is made here.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Final, Protocol, runtime_checkable

from voiceloop_common import Clock, ValidationError, utcnow

from .detectors import assert_no_sensitive_text
from .settings import SYNTHETIC_LABEL, DriftSettings

_ACTOR_PATTERN = re.compile(r"^(?:operator|reviewer|admin|lead)-[0-9a-f]{4,8}$")

MANUAL_AUDIT_SINK_REQUIRED: Final = "manual_override requires an explicit audit_sink"

MANUAL_REASON_DESCRIPTIONS: Mapping[str, str] = {
    "incident_replay": "Operator requested an incident-replay retraining rehearsal.",
    "review_backlog": "Operator requested retraining after a verified review backlog.",
    "lead_authorized": "Lead-authorized retraining override.",
}


class TriggerReason(StrEnum):
    """Allowed reasons for starting the retraining chain."""

    MANUAL_OVERRIDE = "manual_override"
    DRIFT = "drift"
    DATA_VOLUME = "data_volume"

    @classmethod
    def allows_start(cls, value: str) -> bool:
        """Return whether a recorded reason is allowed to start a pipeline."""
        return value in {reason.value for reason in cls}


_REASON_PRIORITY = (TriggerReason.MANUAL_OVERRIDE, TriggerReason.DRIFT, TriggerReason.DATA_VOLUME)


def _iso_utc(value: datetime) -> str:
    """Return an RFC 3339 UTC timestamp from a checked datetime."""
    if not isinstance(value, datetime):
        raise ValidationError("drift clock must return a datetime")
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return aware.astimezone(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class ManualOverride:
    """Auditable operator override with a pseudonymous actor and coded reason."""

    actor_id: str
    reason_code: str

    def __post_init__(self) -> None:
        """Validate the actor and reason; free text is never accepted."""
        if not isinstance(self.actor_id, str) or not _ACTOR_PATTERN.fullmatch(self.actor_id):
            raise ValidationError("manual actor_id must be a pseudonymous role-id token")
        if self.reason_code not in MANUAL_REASON_DESCRIPTIONS:
            raise ValidationError("manual reason_code is not allowlisted")

    @property
    def reason(self) -> str:
        """Return the fixed, PII-free description of the coded reason."""
        return MANUAL_REASON_DESCRIPTIONS[self.reason_code]

    def to_dict(self) -> dict[str, str]:
        """Return the actor/reason audit fields."""
        payload = {
            "actor_id": self.actor_id,
            "reason_code": self.reason_code,
            "reason": self.reason,
        }
        assert_no_sensitive_text(payload)
        return payload


@runtime_checkable
class AuditSink(Protocol):
    """Write-ahead audit seam for manual overrides."""

    def record(self, entry: Mapping[str, Any]) -> None:
        """Persist one audit entry before the pipeline is invoked."""


@dataclass
class InMemoryAuditSink:
    """Deterministic fake audit sink supplied explicitly by callers and tests."""

    entries: list[dict[str, Any]] = field(default_factory=list)

    def record(self, entry: Mapping[str, Any]) -> None:
        """Copy and validate one audit entry."""
        payload = dict(entry)
        assert_no_sensitive_text(payload)
        self.entries.append(payload)


def require_explicit_audit_sink(
    manual_override: ManualOverride | None, audit_sink: AuditSink | None
) -> None:
    """Refuse a manual override that has no caller-supplied audit sink.

    Args:
        manual_override: Parsed override, or None on the routine path.
        audit_sink: Explicit sink from the caller; no in-package fallback exists.

    Raises:
        ValidationError: If an override is present without a sink.

    """
    if manual_override is not None and audit_sink is None:
        raise ValidationError(MANUAL_AUDIT_SINK_REQUIRED)


def manual_override_from_settings(settings: DriftSettings | None = None) -> ManualOverride | None:
    """Build an override only from an explicit flag plus actor and reason."""
    config = settings or DriftSettings()
    if not config.manual_override:
        if config.manual_actor_id or config.manual_reason_code:
            raise ValidationError(
                "manual actor/reason require VOICELOOP_DRIFT_MANUAL_OVERRIDE=true"
            )
        return None
    return ManualOverride(actor_id=config.manual_actor_id, reason_code=config.manual_reason_code)


@dataclass(frozen=True, slots=True)
class TriggerDecision:
    """Result of the P4-02 gate, safe to embed in a run record."""

    triggered: bool
    reasons: tuple[TriggerReason, ...]
    new_verified_count: int
    min_new_verified: int
    drifted: bool
    manual_override: ManualOverride | None
    checked_at: str
    last_run_at: str | None = None
    scheduled_tick: bool = False
    blocked_reasons: tuple[str, ...] = ()
    audit_recorded: bool = False

    @property
    def reason(self) -> TriggerReason | None:
        """Return the highest-priority satisfied reason, or None."""
        return self.reasons[0] if self.reasons else None

    def to_dict(self) -> dict[str, Any]:
        """Return a labelled, JSON-serialisable decision report."""
        payload: dict[str, Any] = {
            "label": SYNTHETIC_LABEL,
            "triggered": self.triggered,
            "reason": self.reason.value if self.reason is not None else None,
            "reasons": [reason.value for reason in self.reasons],
            "data_volume_ok": self.new_verified_count >= self.min_new_verified,
            "drifted": self.drifted,
            "manual_override": (
                self.manual_override.to_dict() if self.manual_override is not None else None
            ),
            "new_verified_count": self.new_verified_count,
            "min_new_verified": self.min_new_verified,
            "checked_at": self.checked_at,
            "last_run_at": self.last_run_at,
            "scheduled_tick": self.scheduled_tick,
            "calendar_tick_is_trigger": False,
            "blocked_reasons": list(self.blocked_reasons),
            "audit_recorded": self.audit_recorded,
        }
        assert_no_sensitive_text(payload)
        return payload


def decide_trigger(
    *,
    new_verified_count: int,
    drifted: bool,
    settings: DriftSettings | None = None,
    manual_override: ManualOverride | None = None,
    audit_sink: AuditSink | None = None,
    clock: Clock = utcnow,
    scheduled_tick: bool = False,
    last_run_at: datetime | None = None,
) -> TriggerDecision:
    """Apply the data-volume AND (drift OR manual) gate.

    Args:
        new_verified_count: Newly verified examples available for retraining.
        drifted: Combined detector alarm from replayed windows.
        settings: Typed settings; ``min_new_verified`` is the volume floor.
        manual_override: Explicit actor/reason override, or None.
        audit_sink: Required caller-supplied sink for a manual override.
        clock: Injected UTC clock for deterministic audit timestamps.
        scheduled_tick: Audit-only marker for calendar-driven invocations.
        last_run_at: Audit-only timestamp of the previous run.

    Returns:
        A decision whose reasons never include a calendar tick.

    Raises:
        ValidationError: For an invalid count, override, or missing audit sink.

    """
    require_explicit_audit_sink(manual_override, audit_sink)
    if isinstance(new_verified_count, bool) or not isinstance(new_verified_count, int):
        raise ValidationError("new_verified_count must be an integer")
    if new_verified_count < 0:
        raise ValidationError("new_verified_count must be non-negative")
    if not isinstance(drifted, bool):
        raise ValidationError("drifted must be a boolean")

    config = settings or DriftSettings()
    if manual_override is not None and not isinstance(manual_override, ManualOverride):
        raise ValidationError("manual_override must be a ManualOverride or None")

    volume_ok = new_verified_count >= config.min_new_verified
    gate_ok = bool(drifted) or manual_override is not None
    triggered = volume_ok and gate_ok

    reasons = tuple(
        reason
        for reason in _REASON_PRIORITY
        if (reason is TriggerReason.MANUAL_OVERRIDE and manual_override is not None)
        or (reason is TriggerReason.DRIFT and drifted)
        or (reason is TriggerReason.DATA_VOLUME and volume_ok)
    )
    blocked: list[str] = []
    if not volume_ok:
        blocked.append("insufficient_new_verified_data")
    if not gate_ok:
        blocked.append("no_drift_or_manual_override")

    checked_at = _iso_utc(clock())
    audit_recorded = False
    if manual_override is not None:
        assert audit_sink is not None  # guaranteed by require_explicit_audit_sink()
        entry: dict[str, Any] = {
            "label": SYNTHETIC_LABEL,
            "event": "manual_override",
            **manual_override.to_dict(),
            "checked_at": checked_at,
            "new_verified_count": new_verified_count,
            "min_new_verified": config.min_new_verified,
            "drifted": bool(drifted),
            "data_volume_ok": volume_ok,
            "triggered": triggered,
            "scheduled_tick": bool(scheduled_tick),
        }
        sink = audit_sink if audit_sink is not None else InMemoryAuditSink()
        sink.record(entry)
        audit_recorded = True

    return TriggerDecision(
        triggered=triggered,
        reasons=reasons if triggered else (),
        new_verified_count=new_verified_count,
        min_new_verified=config.min_new_verified,
        drifted=bool(drifted),
        manual_override=manual_override,
        checked_at=checked_at,
        last_run_at=_iso_utc(last_run_at) if last_run_at is not None else None,
        scheduled_tick=bool(scheduled_tick),
        blocked_reasons=tuple(blocked),
        audit_recorded=audit_recorded,
    )
