"""Trigger policy for retraining: verified data volume, drift, or manual override.

A calendar tick is deliberately *not* a trigger. Time arguments are accepted so
callers can record when the check happened, but they never affect the decision
(roadmap P2-10.2 and P4-02.2; context.md section 9 keeps the gate data-driven).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from voiceloop_common import ValidationError, utcnow

from .settings import PipelineSettings


class TriggerReason(StrEnum):
    """Reasons that are allowed to start a chain."""

    MANUAL = "manual_override"
    DRIFT = "drift_alarm"
    NEW_DATA = "new_verified_data"


_REASON_PRIORITY = (TriggerReason.MANUAL, TriggerReason.DRIFT, TriggerReason.NEW_DATA)


def _iso(value: datetime | None) -> str | None:
    """Return an RFC 3339 UTC string for a datetime, or None."""
    if value is None:
        return None
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return aware.astimezone(UTC).isoformat()


def count_new_verified(verified_ids: Iterable[str], seen_ids: Iterable[str]) -> int:
    """Count verified example IDs not present in a previous run record.

    Args:
        verified_ids: IDs currently available for training.
        seen_ids: IDs already consumed by the last recorded run.

    Returns:
        The number of genuinely new IDs.

    """
    return len(set(verified_ids) - set(seen_ids))


@dataclass(frozen=True, slots=True)
class TriggerDecision:
    """Result of evaluating the trigger policy, safe to embed in a run record."""

    triggered: bool
    reasons: tuple[TriggerReason, ...]
    new_verified_count: int
    threshold: int
    drift_alarm: bool
    manual: bool
    last_run_at: datetime | None = None
    checked_at: datetime | None = None

    @property
    def reason(self) -> TriggerReason | None:
        """Return the highest-priority satisfied reason, or None if no trigger."""
        return self.reasons[0] if self.reasons else None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation for CLI output and records."""
        return {
            "triggered": self.triggered,
            "reason": self.reason.value if self.reason is not None else None,
            "reasons": [reason.value for reason in self.reasons],
            "new_verified_count": self.new_verified_count,
            "threshold": self.threshold,
            "drift_alarm": self.drift_alarm,
            "manual": self.manual,
            "last_run_at": _iso(self.last_run_at),
            "checked_at": _iso(self.checked_at),
        }


def decide_trigger(
    *,
    new_verified_count: int,
    settings: PipelineSettings | None = None,
    drift_alarm: bool | None = None,
    manual: bool | None = None,
    last_run_at: datetime | None = None,
    checked_at: datetime | None = None,
) -> TriggerDecision:
    """Evaluate the three allowed trigger conditions.

    Args:
        new_verified_count: Verified examples added since the last recorded run.
        settings: Typed settings supplying the volume threshold and defaults.
        drift_alarm: Overrides ``settings.drift_alarm`` when provided.
        manual: Overrides ``settings.manual`` when provided.
        last_run_at: Informational timestamp of the previous run; never a trigger.
        checked_at: Informational time of this check; never a trigger.

    Returns:
        A decision with every satisfied reason and a deterministic primary reason.

    Raises:
        ValidationError: If ``new_verified_count`` is negative or not an integer.

    """
    if isinstance(new_verified_count, bool) or not isinstance(new_verified_count, int):
        raise ValidationError("new_verified_count must be an integer")
    if new_verified_count < 0:
        raise ValidationError("new_verified_count must be non-negative")

    config = settings or PipelineSettings()
    drift = config.drift_alarm if drift_alarm is None else bool(drift_alarm)
    manual_override = config.manual if manual is None else bool(manual)

    satisfied: set[TriggerReason] = set()
    if manual_override:
        satisfied.add(TriggerReason.MANUAL)
    if drift:
        satisfied.add(TriggerReason.DRIFT)
    if new_verified_count >= config.min_new_verified:
        satisfied.add(TriggerReason.NEW_DATA)
    reasons = tuple(reason for reason in _REASON_PRIORITY if reason in satisfied)

    return TriggerDecision(
        triggered=bool(reasons),
        reasons=reasons,
        new_verified_count=new_verified_count,
        threshold=config.min_new_verified,
        drift_alarm=drift,
        manual=manual_override,
        last_run_at=last_run_at,
        checked_at=checked_at or utcnow(),
    )


def should_trigger(
    *,
    new_verified_count: int,
    settings: PipelineSettings | None = None,
    drift_alarm: bool | None = None,
    manual: bool | None = None,
    last_run_at: datetime | None = None,
    checked_at: datetime | None = None,
) -> bool:
    """Return True only for enough new data, a drift alarm, or a manual override.

    Time passing (``last_run_at`` / ``checked_at``) is intentionally ignored.
    """
    return decide_trigger(
        new_verified_count=new_verified_count,
        settings=settings,
        drift_alarm=drift_alarm,
        manual=manual,
        last_run_at=last_run_at,
        checked_at=checked_at,
    ).triggered
