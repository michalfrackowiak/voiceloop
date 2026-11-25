"""Bounded, idempotent, audited one-step rollback for the canary rehearsal.

The only mutable registry operation in this package is a single ``set`` that
restores the champion version recorded in an injected P2-09 promotion record.
There is no promote path, no alias creation and no threshold edit
(``context.md`` §2 invariants 4/10 and §14). The current alias is cross-checked
against the P3-06 read-only registry protocol before any write.

Rollback is fail closed: missing/unexpected alias state, a stale version, an
outside-window observation or an audit failure all refuse the flip (or restore
the pre-flip alias) and return a typed result instead of raising. A successful
rollback writes exactly one alias flip and exactly one hash-chained audit
record; repeating the call is a no-op.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol

from services.evalgate.gate.aliases import CHAMPION_ALIAS
from services.evalgate.gate.audit import PromotionRecord
from services.release.sync.registry import ReadOnlyAliasStore
from voiceloop_common import ValidationError

from .metrics import RegressionVerdict
from .settings import CanarySettings

REPO_ROOT = Path(__file__).resolve().parents[3]
RollbackStatus = Literal[
    "not_needed", "applied", "idempotent", "failed", "outside_window", "stale_alias_version"
]


def canonical_json(payload: Any) -> str:
    """Return canonical UTF-8 JSON with sorted keys and no whitespace."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    """Return the SHA-256 hex digest of UTF-8 encoded text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def timestamp(value: datetime) -> str:
    """Format a fake-clock instant as RFC 3339 UTC."""
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


class AliasStore(Protocol):
    """P2-09-compatible alias seam; one ``set`` is one alias flip."""

    def get(self, alias: str) -> str | None:
        """Return the version bound to ``alias``, or None when unbound."""
        ...

    def set(self, alias: str, version: str) -> None:
        """Bind ``alias`` to ``version`` in one atomic flip."""
        ...


@dataclass(frozen=True)
class CanaryDeployment:
    """Immutable start state of one canary, derived from a gate promotion."""

    alias: str
    previous_version: str
    canary_version: str
    started_at: datetime
    evidence_label: str

    @classmethod
    def from_promotion(
        cls,
        promotion: PromotionRecord,
        started_at: datetime,
        evidence_label: str,
        alias: str = CHAMPION_ALIAS,
    ) -> CanaryDeployment:
        """Build the rollback target from a read-only P2-09 promote record.

        Args:
            promotion: Gate audit record whose ``before_version`` is the sole
                permitted rollback target.
            started_at: Fake-clock instant when canary monitoring began.
            evidence_label: Mandatory synthetic/fixture provenance label.
            alias: Alias that was flipped by the gate (defaults to champion).

        Returns:
            The immutable deployment context.

        Raises:
            voiceloop_common.ValidationError: If the record is not a promote
                or its versions are blank.

        """
        if promotion.action != "promote":
            raise ValidationError("canary deployment requires a promote record")
        for name, value in (
            ("before_version", promotion.before_version),
            ("after_version", promotion.after_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"{name} must be a non-blank string")
        if promotion.before_version == promotion.after_version:
            raise ValidationError("promote record has no alias change to roll back")
        return cls(
            alias=alias,
            previous_version=promotion.before_version,
            canary_version=promotion.after_version,
            started_at=started_at,
            evidence_label=evidence_label,
        )

    def deadline(self, window_seconds: float) -> datetime:
        """Return the latest instant at which a rollback is permitted."""
        return self.started_at + timedelta(seconds=window_seconds)


@dataclass(frozen=True)
class RollbackRecord:
    """One hash-chained, PII-safe rollback audit record."""

    task_id: str
    actor_id: str
    alias: str
    previous_version: str
    canary_version: str
    reason: str
    observed_at: str
    evidence_label: str
    previous_hash: str = ""
    entry_hash: str = ""

    def payload(self) -> dict[str, Any]:
        """Return the fields covered by ``entry_hash``."""
        data = asdict(self)
        data.pop("entry_hash")
        return data

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of the record."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RollbackRecord:
        """Rebuild a record from its on-disk JSON object."""
        return cls(
            task_id=str(data["task_id"]),
            actor_id=str(data["actor_id"]),
            alias=str(data["alias"]),
            previous_version=str(data["previous_version"]),
            canary_version=str(data["canary_version"]),
            reason=str(data["reason"]),
            observed_at=str(data["observed_at"]),
            evidence_label=str(data["evidence_label"]),
            previous_hash=str(data.get("previous_hash", "")),
            entry_hash=str(data.get("entry_hash", "")),
        )


class RollbackAudit(Protocol):
    """Append-only audit seam used by the rollback controller."""

    def head_hash(self) -> str:
        """Return the newest entry hash, or an empty string when empty."""
        ...

    def record(self, record: RollbackRecord) -> RollbackRecord:
        """Append one record and return it with chain hashes filled in."""
        ...

    def entries(self) -> tuple[RollbackRecord, ...]:
        """Return all persisted records in order."""
        ...


def _chain(record: RollbackRecord, head: str) -> RollbackRecord:
    """Attach the previous hash and recompute the entry hash."""
    linked = replace(record, previous_hash=head)
    return replace(linked, entry_hash=sha256_hex(canonical_json(linked.payload())))


class InMemoryRollbackAudit:
    """Deterministic in-memory FAKE append-only audit sink."""

    def __init__(self) -> None:
        """Initialise an empty record list."""
        self._records: list[RollbackRecord] = []

    def entries(self) -> tuple[RollbackRecord, ...]:
        """Return all records in append order."""
        return tuple(self._records)

    def head_hash(self) -> str:
        """Return the last record's hash, or an empty string."""
        return self._records[-1].entry_hash if self._records else ""

    def record(self, record: RollbackRecord) -> RollbackRecord:
        """Append one chained record."""
        linked = _chain(record, self.head_hash())
        self._records.append(linked)
        return linked

    def verify_chain(self) -> tuple[str, ...]:
        """Return integrity errors; an empty tuple means the chain is intact."""
        errors: list[str] = []
        previous = ""
        for number, record in enumerate(self._records, start=1):
            if record.previous_hash != previous:
                errors.append(f"audit record {number}: broken chain")
            if record.entry_hash != sha256_hex(canonical_json(record.payload())):
                errors.append(f"audit record {number}: hash mismatch")
            previous = record.entry_hash
        return tuple(errors)


class CollectingRollbackAudit(InMemoryRollbackAudit):
    """Alias for the in-memory FAKE rollback audit used in tests and demos."""


class RollbackAuditLog(InMemoryRollbackAudit):
    """Append-only JSONL audit log with SHA-256 chain, resolved from repo root."""

    def __init__(self, path: str | Path) -> None:
        """Initialise the log path; the file is created on first write."""
        super().__init__()
        candidate = Path(path)
        self.path = candidate if candidate.is_absolute() else REPO_ROOT / candidate

    def entries(self) -> tuple[RollbackRecord, ...]:
        """Read all records from disk, or return an empty tuple if absent."""
        if not self.path.exists():
            return ()
        found: list[RollbackRecord] = []
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"Audit log line {number} is not valid JSON: {exc}") from exc
            if not isinstance(decoded, dict):
                raise ValidationError(f"Audit log line {number} must be an object")
            found.append(RollbackRecord.from_dict(decoded))
        return tuple(found)

    def head_hash(self) -> str:
        """Return the newest on-disk record hash, or an empty string."""
        entries = self.entries()
        return entries[-1].entry_hash if entries else ""

    def record(self, record: RollbackRecord) -> RollbackRecord:
        """Chain, append and return one audit record."""
        linked = _chain(record, self.head_hash())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(linked.to_dict()) + "\n")
        return linked


@dataclass(frozen=True)
class RollbackResult:
    """Typed outcome of one bounded rollback attempt."""

    status: RollbackStatus
    reason: str
    observed_at: str
    alias: str
    current_version: str | None
    target_version: str | None
    writes: int
    error_type: str | None = None
    record: RollbackRecord | None = None
    evidence_label: str = ""

    def to_event(self, task_id: str) -> dict[str, Any]:
        """Return an allowlisted structured-log payload."""
        return {
            "event": "canary.rollback",
            "task_id": task_id,
            "status": self.status,
            "reason": self.reason,
            "observed_at": self.observed_at,
            "alias": self.alias,
            "current_version": self.current_version,
            "target_version": self.target_version,
            "writes": self.writes,
            "error_type": self.error_type,
            "evidence_label": self.evidence_label,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a PII-safe report fragment."""
        return {
            "status": self.status,
            "reason": self.reason,
            "observed_at": self.observed_at,
            "alias": self.alias,
            "current_version": self.current_version,
            "target_version": self.target_version,
            "alias_writes": self.writes,
            "error_type": self.error_type,
            "audit_record": self.record.to_dict() if self.record is not None else None,
            "evidence_label": self.evidence_label,
        }


class RollbackController:
    """One-shot, idempotent rollback guarded by the configured window."""

    def __init__(
        self,
        settings: CanarySettings,
        store: AliasStore,
        audit: RollbackAudit,
        registry: ReadOnlyAliasStore | None = None,
    ) -> None:
        """Inject typed settings, the writable alias seam, audit and registry."""
        self._settings = settings
        self._store = store
        self._audit = audit
        self._registry: ReadOnlyAliasStore = registry if registry is not None else store
        self._applied: RollbackRecord | None = None

    @property
    def applied(self) -> RollbackRecord | None:
        """Return the applied rollback record, or None when not yet applied."""
        return self._applied

    def rollback(
        self,
        verdict: RegressionVerdict,
        deployment: CanaryDeployment,
        observed_at: datetime,
    ) -> RollbackResult:
        """Restore the recorded previous version in one flip, if permitted.

        Args:
            verdict: Descriptive, fail-closed decision from canary metrics.
            deployment: Immutable start state from a P2-09 promote record.
            observed_at: Fake-clock instant, compared with the window deadline.

        Returns:
            A typed result; all expected failures are reported, not raised.

        """
        at = timestamp(observed_at)
        target = deployment.previous_version
        if self._applied is not None:
            return RollbackResult(
                status="idempotent",
                reason="already_rolled_back",
                observed_at=at,
                alias=deployment.alias,
                current_version=self._safe_get(deployment.alias),
                target_version=target,
                writes=0,
                record=self._applied,
                evidence_label=deployment.evidence_label,
            )
        if not verdict.rollback:
            return RollbackResult(
                status="not_needed",
                reason=verdict.reason,
                observed_at=at,
                alias=deployment.alias,
                current_version=self._safe_get(deployment.alias),
                target_version=target,
                writes=0,
                evidence_label=deployment.evidence_label,
            )
        if observed_at > deployment.deadline(self._settings.window_seconds):
            return RollbackResult(
                status="outside_window",
                reason="rollback_deadline_exceeded",
                observed_at=at,
                alias=deployment.alias,
                current_version=self._safe_get(deployment.alias),
                target_version=target,
                writes=0,
                evidence_label=deployment.evidence_label,
            )

        try:
            current = self._store.get(deployment.alias)
            observed = self._registry.get(deployment.alias)
        except Exception as exc:
            return RollbackResult(
                status="failed",
                reason="alias_read_failed",
                observed_at=at,
                alias=deployment.alias,
                current_version=None,
                target_version=target,
                writes=0,
                error_type=type(exc).__name__,
                evidence_label=deployment.evidence_label,
            )
        if current is None or not str(current).strip():
            return RollbackResult(
                status="failed",
                reason="alias_unbound",
                observed_at=at,
                alias=deployment.alias,
                current_version=current,
                target_version=target,
                writes=0,
                evidence_label=deployment.evidence_label,
            )
        if observed != current:
            return RollbackResult(
                status="failed",
                reason="registry_alias_mismatch",
                observed_at=at,
                alias=deployment.alias,
                current_version=current,
                target_version=target,
                writes=0,
                evidence_label=deployment.evidence_label,
            )
        if current == target:
            return RollbackResult(
                status="idempotent",
                reason="alias_already_at_previous_version",
                observed_at=at,
                alias=deployment.alias,
                current_version=current,
                target_version=target,
                writes=0,
                evidence_label=deployment.evidence_label,
            )
        if current != deployment.canary_version:
            return RollbackResult(
                status="stale_alias_version",
                reason="unexpected_current_version",
                observed_at=at,
                alias=deployment.alias,
                current_version=current,
                target_version=target,
                writes=0,
                evidence_label=deployment.evidence_label,
            )

        record = RollbackRecord(
            task_id=self._settings.task_id,
            actor_id=self._settings.rollback_actor_id,
            alias=deployment.alias,
            previous_version=target,
            canary_version=current,
            reason=verdict.reason,
            observed_at=at,
            evidence_label=deployment.evidence_label,
        )
        try:
            self._store.set(deployment.alias, target)
        except Exception as exc:
            return RollbackResult(
                status="failed",
                reason="alias_write_failed",
                observed_at=at,
                alias=deployment.alias,
                current_version=current,
                target_version=target,
                writes=0,
                error_type=type(exc).__name__,
                evidence_label=deployment.evidence_label,
            )
        try:
            saved = self._audit.record(record)
        except Exception as exc:
            # Compensating restore: net zero alias change, no audit record.
            self._store.set(deployment.alias, current)
            return RollbackResult(
                status="failed",
                reason="audit_write_failed",
                observed_at=at,
                alias=deployment.alias,
                current_version=current,
                target_version=target,
                writes=2,
                error_type=type(exc).__name__,
                evidence_label=deployment.evidence_label,
            )

        self._applied = saved
        return RollbackResult(
            status="applied",
            reason=verdict.reason,
            observed_at=at,
            alias=deployment.alias,
            current_version=current,
            target_version=target,
            writes=1,
            record=saved,
            evidence_label=deployment.evidence_label,
        )

    def _safe_get(self, alias: str) -> str | None:
        """Read the alias for reporting without propagating source errors."""
        try:
            value = self._store.get(alias)
        except Exception:
            return None
        return value if isinstance(value, str) else None
