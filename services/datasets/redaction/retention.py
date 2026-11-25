"""Injected retention job for fake raw-audio deletion (P3-10.3).

The retention window is an **engineering placeholder**, not a legal or
compliance decision. ``docs/compliance.md`` stays ``OPEN — counsel review
required``; this job deletes nothing in the real world. It calls an injected
metadata-only store seam and emits an auditable, PII-free report with deleted
and kept counts and explicit reasons. Redacted and training-derived assets are
never eligible for deletion here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol

from voiceloop_common import StorageError

from .settings import (
    COMPLIANCE_STATUS,
    POLICY_STATUS,
    SYNTHETIC_LABEL,
    AuditEvent,
    AuditSink,
    Clock,
    InMemoryAuditSink,
    RedactionSettings,
    SystemClock,
)
from .spans import RedactionError

REASON_EXPIRED_RAW = "expired_raw_audio"
REASON_WITHIN_WINDOW = "within_retention_window"
REASON_NOT_RAW = "not_raw_audio"
REASON_FUTURE_CAPTURE = "future_capture_timestamp"

NO_REAL_DELETION_CLAIM = (
    "FAKE STAND-IN: deletion applies only to the injected in-memory store; "
    "no real object store or legal deletion is claimed"
)


class AssetKind(StrEnum):
    """Metadata asset classes; only RAW_AUDIO can be selected for deletion."""

    RAW_AUDIO = "raw_audio"
    REDACTED_AUDIO = "redacted_audio"
    TRAINING_DERIVED = "training_derived"


@dataclass(frozen=True, slots=True)
class RetainedAsset:
    """Metadata-only asset record; never holds audio bytes or guest text."""

    uri: str
    kind: AssetKind
    captured_at: datetime
    source_call_id: str = ""


class RetentionStore(Protocol):
    """Injectable metadata store seam; the default fake never touches a real store."""

    def list_assets(self) -> tuple[RetainedAsset, ...]:
        """Return all retained asset metadata records."""
        ...

    def delete_raw(self, uri: str) -> None:
        """Remove one raw-audio record from the fake store."""
        ...


@dataclass(slots=True)
class InMemoryRetentionStore:
    """Fake retention store: metadata only, deterministic iteration."""

    assets: dict[str, RetainedAsset] = field(default_factory=dict)
    deleted_uris: list[str] = field(default_factory=list)

    def add(self, asset: RetainedAsset) -> None:
        """Store one metadata-only asset record."""
        self.assets[asset.uri] = asset

    def list_assets(self) -> tuple[RetainedAsset, ...]:
        """Return asset metadata sorted by URI and kind."""
        return tuple(sorted(self.assets.values(), key=lambda item: (item.uri, str(item.kind))))

    def delete_raw(self, uri: str) -> None:
        """Delete a raw record, failing closed on unknown or non-raw assets."""
        asset = self.assets.get(uri)
        if asset is None:
            raise StorageError(f"unknown synthetic asset URI: {uri}")
        if asset.kind is not AssetKind.RAW_AUDIO:
            raise StorageError(f"asset is not raw audio: {uri}")
        del self.assets[uri]
        self.deleted_uris.append(uri)


@dataclass(frozen=True, slots=True)
class DeletionDecision:
    """One auditable keep/delete decision with a stable reason."""

    uri: str
    kind: AssetKind
    deleted: bool
    reason: str
    age_days: float

    def to_dict(self) -> dict[str, str | bool | float]:
        """Return a PII-free decision record."""
        return {
            "uri": self.uri,
            "kind": str(self.kind),
            "deleted": self.deleted,
            "reason": self.reason,
            "age_days": self.age_days,
        }


@dataclass(frozen=True, slots=True)
class RetentionReport:
    """Auditable summary of one idempotent fake retention run."""

    now: datetime
    window_days: int
    decisions: tuple[DeletionDecision, ...]
    policy_status: str = POLICY_STATUS
    compliance_status: str = COMPLIANCE_STATUS
    deletion_claim: str = NO_REAL_DELETION_CLAIM

    @property
    def deleted_count(self) -> int:
        """Return the number of raw assets deleted in this run."""
        return sum(1 for decision in self.decisions if decision.deleted)

    @property
    def kept_count(self) -> int:
        """Return the number of assets kept in this run."""
        return sum(1 for decision in self.decisions if not decision.deleted)

    def reason_counts(self) -> dict[str, int]:
        """Return counts grouped by stable reason, sorted by reason name."""
        counts: dict[str, int] = {}
        for decision in self.decisions:
            counts[decision.reason] = counts.get(decision.reason, 0) + 1
        return {key: counts[key] for key in sorted(counts)}

    def to_dict(self) -> dict[str, object]:
        """Return the full JSON-safe audit report; no real deletion is claimed."""
        return {
            "now": self.now.isoformat(),
            "window_days": self.window_days,
            "scanned": len(self.decisions),
            "deleted": self.deleted_count,
            "kept": self.kept_count,
            "reasons": self.reason_counts(),
            "decisions": [decision.to_dict() for decision in self.decisions],
            "policy_status": self.policy_status,
            "compliance_status": self.compliance_status,
            "deletion_claim": self.deletion_claim,
            "synthetic": True,
            "label": SYNTHETIC_LABEL,
        }


def _as_utc(value: datetime) -> datetime:
    """Normalise a datetime to UTC, interpreting naive values as UTC."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def run_retention(
    store: RetentionStore,
    *,
    settings: RedactionSettings | None = None,
    clock: Clock | None = None,
    audit: AuditSink | None = None,
) -> RetentionReport:
    """Delete only expired raw audio from the injected fake store.

    Args:
        store: Injected metadata-only store seam.
        settings: Typed ``VOICELOOP_REDACTION_`` settings; the retention window
            is an engineering placeholder pending counsel review.
        clock: Injected UTC clock for deterministic boundary tests.
        audit: Injected PII-free audit sink; defaults to an in-memory fake.

    Returns:
        Report with deleted/kept counts, reasons and no real deletion claim.
        Running twice on the same store deletes nothing the second time.

    """
    config = settings or RedactionSettings()
    now = _as_utc((clock or SystemClock()).now())
    audit_sink = audit or InMemoryAuditSink()
    window = timedelta(days=config.retention_days)

    decisions: list[DeletionDecision] = []
    for asset in sorted(store.list_assets(), key=lambda item: (item.uri, str(item.kind))):
        captured = _as_utc(asset.captured_at)
        age = now - captured
        age_days = age.total_seconds() / 86400.0
        if asset.kind is not AssetKind.RAW_AUDIO:
            deleted, reason = False, REASON_NOT_RAW
        elif age < timedelta(0):
            deleted, reason = False, REASON_FUTURE_CAPTURE
        elif age >= window:
            store.delete_raw(asset.uri)
            deleted, reason = True, REASON_EXPIRED_RAW
            audit_sink.emit(
                AuditEvent(
                    event_type="raw_audio_fake_deleted",
                    occurred_at=now,
                    details={
                        "uri": asset.uri,
                        "reason": reason,
                        "age_days": round(age_days, 3),
                        "window_days": config.retention_days,
                        "policy_status": POLICY_STATUS,
                        "label": SYNTHETIC_LABEL,
                    },
                )
            )
        else:
            deleted, reason = False, REASON_WITHIN_WINDOW
        decisions.append(
            DeletionDecision(
                uri=asset.uri,
                kind=asset.kind,
                deleted=deleted,
                reason=reason,
                age_days=round(age_days, 3),
            )
        )

    report = RetentionReport(
        now=now,
        window_days=config.retention_days,
        decisions=tuple(decisions),
    )
    audit_sink.emit(
        AuditEvent(
            event_type="retention_run",
            occurred_at=now,
            details={
                "scanned": len(decisions),
                "deleted": report.deleted_count,
                "kept": report.kept_count,
                "window_days": config.retention_days,
                "policy_status": POLICY_STATUS,
                "compliance_status": COMPLIANCE_STATUS,
                "label": SYNTHETIC_LABEL,
            },
        )
    )
    return report


def asset_from_dict(data: Mapping[str, Any]) -> RetainedAsset:
    """Parse one CLI asset record with strict, PII-free validation.

    Args:
        data: JSON-like mapping with ``uri``, ``kind``, ``captured_at`` and an
            optional ``source_call_id``.

    Returns:
        Validated metadata-only asset record.

    Raises:
        RedactionError: A field is missing, malformed or has an unknown kind.

    """
    uri = data.get("uri")
    raw_kind = data.get("kind")
    captured = data.get("captured_at")
    if not isinstance(uri, str) or not uri:
        raise RedactionError("invalid_asset_record", "asset uri must be a non-empty string")
    if not isinstance(raw_kind, str):
        raise RedactionError("invalid_asset_record", "asset kind must be a string")
    try:
        kind = AssetKind(raw_kind)
    except ValueError as exc:
        raise RedactionError(
            "invalid_asset_record",
            "unknown asset kind",
            {"allowed_kinds": len(AssetKind)},
        ) from exc
    if not isinstance(captured, str):
        raise RedactionError("invalid_asset_record", "asset captured_at must be a string")
    try:
        captured_at = datetime.fromisoformat(captured)
    except ValueError as exc:
        raise RedactionError("invalid_asset_record", "asset captured_at is not ISO-8601") from exc
    call_id = data.get("source_call_id", "")
    if not isinstance(call_id, str):
        raise RedactionError("invalid_asset_record", "asset source_call_id must be a string")
    return RetainedAsset(
        uri=uri,
        kind=kind,
        captured_at=captured_at,
        source_call_id=call_id,
    )


__all__ = [
    "REASON_EXPIRED_RAW",
    "REASON_FUTURE_CAPTURE",
    "REASON_NOT_RAW",
    "REASON_WITHIN_WINDOW",
    "AssetKind",
    "DeletionDecision",
    "InMemoryRetentionStore",
    "RetainedAsset",
    "RetentionReport",
    "RetentionStore",
    "asset_from_dict",
    "run_retention",
]
