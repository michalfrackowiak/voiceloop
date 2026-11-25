"""Append-only, hash-chained audit records for alias promotions and rollbacks.

Every record stores who acted, when (``voiceloop_common.utcnow()``), why, and
the before/after champion versions. Records are frozen dataclasses written as
canonical JSON Lines; the SHA-256 chain makes unaudited edits detectable. This
is tamper evidence, not a cryptographic signature.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol

from voiceloop_common import ValidationError, utcnow

PromotionAction = Literal["promote", "rollback"]
AUDIT_TASK_ID = "P2-09"
REPO_ROOT = Path(__file__).resolve().parents[3]


def canonical_json(payload: Any) -> str:
    """Return canonical UTF-8 JSON: sorted keys and no optional whitespace."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    """Return the SHA-256 hex digest of UTF-8 encoded text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_timestamp() -> str:
    """Return the current UTC time in RFC 3339 form with a ``Z`` suffix."""
    return utcnow().isoformat(timespec="seconds").replace("+00:00", "Z")


def resolve_path(path: str | Path) -> Path:
    """Resolve a repo-relative path against the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


@dataclass(frozen=True)
class PromotionRecord:
    """One immutable alias promotion or rollback record."""

    actor_id: str
    action: PromotionAction
    reason: str
    before_version: str
    after_version: str
    timestamp: str = field(default_factory=utc_timestamp)
    task_id: str = AUDIT_TASK_ID
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
    def from_dict(cls, data: Mapping[str, Any]) -> PromotionRecord:
        """Rebuild a record from its on-disk JSON object."""
        return cls(
            actor_id=str(data["actor_id"]),
            action=data["action"],
            reason=str(data["reason"]),
            before_version=str(data["before_version"]),
            after_version=str(data["after_version"]),
            timestamp=str(data["timestamp"]),
            task_id=str(data.get("task_id", AUDIT_TASK_ID)),
            previous_hash=str(data.get("previous_hash", "")),
            entry_hash=str(data.get("entry_hash", "")),
        )


class AuditLog(Protocol):
    """Minimal audit interface so alias operations can use in-memory fakes."""

    def record(
        self,
        *,
        actor_id: str,
        action: PromotionAction,
        reason: str,
        before_version: str,
        after_version: str,
        task_id: str = AUDIT_TASK_ID,
    ) -> PromotionRecord:
        """Append one record and return it."""
        ...


class PromotionAuditLog:
    """Append-only JSONL audit log with a SHA-256 entry chain."""

    def __init__(self, path: str | Path) -> None:
        """Initialise the log at ``path``; the file is created on first write."""
        self.path = resolve_path(path)

    def entries(self) -> tuple[PromotionRecord, ...]:
        """Return all records in file order, or an empty tuple if absent."""
        if not self.path.exists():
            return ()
        found: list[PromotionRecord] = []
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"Audit log line {number} is not valid JSON: {exc}") from exc
            if not isinstance(decoded, dict):
                raise ValidationError(f"Audit log line {number} must be an object")
            found.append(PromotionRecord.from_dict(decoded))
        return tuple(found)

    def head_hash(self) -> str:
        """Return the newest record hash, or an empty string for an empty log."""
        entries = self.entries()
        return entries[-1].entry_hash if entries else ""

    def record(
        self,
        *,
        actor_id: str,
        action: PromotionAction,
        reason: str,
        before_version: str,
        after_version: str,
        task_id: str = AUDIT_TASK_ID,
    ) -> PromotionRecord:
        """Append one record and return it with its chain and entry hashes.

        Args:
            actor_id: Human or service identity performing the alias flip.
            action: ``promote`` or ``rollback``.
            reason: Human-readable justification for the change.
            before_version: Champion version before the flip.
            after_version: Champion version after the flip.
            task_id: Roadmap task authorising the change.

        Returns:
            The persisted, immutable record.

        Raises:
            voiceloop_common.ValidationError: If any required text is blank.

        """
        for name, value in (
            ("actor_id", actor_id),
            ("reason", reason),
            ("before_version", before_version),
            ("after_version", after_version),
            ("task_id", task_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"Audit record requires a non-blank {name}")
        if action not in ("promote", "rollback"):
            raise ValidationError(f"Unknown promotion action: {action!r}")

        record = PromotionRecord(
            actor_id=actor_id,
            action=action,
            reason=reason,
            before_version=before_version,
            after_version=after_version,
            task_id=task_id,
            previous_hash=self.head_hash(),
        )
        record = replace(record, entry_hash=sha256_hex(canonical_json(record.payload())))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(record.to_dict()) + "\n")
        return record

    def verify_chain(self) -> tuple[str, ...]:
        """Return integrity errors; an empty tuple means the chain is intact."""
        errors: list[str] = []
        previous = ""
        for number, record in enumerate(self.entries(), start=1):
            expected = sha256_hex(canonical_json(record.payload()))
            if record.entry_hash != expected:
                errors.append(f"audit record {number}: hash mismatch")
            if record.previous_hash != previous:
                errors.append(f"audit record {number}: broken chain")
            previous = record.entry_hash
        return tuple(errors)
