"""Append-only access audit log for the sealed golden set.

Every entry records who accessed the set, when (``voiceloop_common.utcnow()``),
why and under which task ID. Entries are hash-chained so unaudited edits are
detectable. This is tamper evidence, not a cryptographic signature.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from voiceloop_common import ValidationError, utcnow

AuditAction = Literal["read", "write", "seal", "verify", "amend"]
REPO_ROOT = Path(__file__).resolve().parents[3]


def canonical_json(payload: Any) -> str:
    """Return canonical UTF-8 JSON text: sorted keys, no optional whitespace."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    """Return the SHA-256 hex digest of UTF-8 encoded text."""
    return hashlib.sha256(text.encode()).hexdigest()


def utc_timestamp() -> str:
    """Return the current UTC time in RFC 3339 form with a ``Z`` suffix."""
    return utcnow().isoformat(timespec="seconds").replace("+00:00", "Z")


def resolve_path(path: str | Path) -> Path:
    """Resolve a repo-relative path against the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def rel_posix(path: Path, base: Path) -> str:
    """Return ``path`` relative to ``base`` as a POSIX-style string."""
    return path.resolve().relative_to(base.resolve()).as_posix()


@dataclass(frozen=True)
class AuditEntry:
    """One access record. ``entry_hash`` covers every other field."""

    actor_id: str
    action: AuditAction
    purpose: str
    task_id: str
    dataset_id: str
    version: str
    timestamp: str = field(default_factory=utc_timestamp)
    content_hash: str | None = None
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
    def from_dict(cls, data: dict[str, Any]) -> AuditEntry:
        """Rebuild an entry from its on-disk JSON object."""
        return cls(
            actor_id=str(data["actor_id"]),
            action=data["action"],
            purpose=str(data["purpose"]),
            task_id=str(data["task_id"]),
            dataset_id=str(data["dataset_id"]),
            version=str(data["version"]),
            timestamp=str(data["timestamp"]),
            content_hash=data.get("content_hash"),
            previous_hash=str(data.get("previous_hash", "")),
            entry_hash=str(data.get("entry_hash", "")),
        )


@dataclass(frozen=True)
class AuditPointer:
    """Immutable pointer from a sealed manifest to an audit-log prefix."""

    path: str
    entries_at_seal: int
    head_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of the pointer."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditPointer:
        """Rebuild a pointer from its on-disk JSON object."""
        return cls(
            path=str(data["path"]),
            entries_at_seal=int(data["entries_at_seal"]),
            head_hash=str(data["head_hash"]),
        )


class AccessAuditLog:
    """Append-only JSONL audit log with a SHA-256 entry chain."""

    def __init__(self, path: str | Path) -> None:
        """Initialise the log at ``path``; the file is created on first write."""
        self.path = resolve_path(path)

    def entries(self) -> tuple[AuditEntry, ...]:
        """Return all entries in file order, or an empty tuple if absent."""
        if not self.path.exists():
            return ()
        found: list[AuditEntry] = []
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"Audit log line {number} is not valid JSON: {exc}") from exc
            if not isinstance(decoded, dict):
                raise ValidationError(f"Audit log line {number} must be an object")
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
        action: AuditAction,
        purpose: str,
        task_id: str,
        dataset_id: str,
        version: str,
        content_hash: str | None = None,
    ) -> AuditEntry:
        """Append one entry and return it.

        Args:
            actor_id: Human or service identity performing the access.
            action: ``read``, ``write``, ``seal``, ``verify`` or ``amend``.
            purpose: Human-readable reason for the access.
            task_id: Roadmap task ID authorising the access.
            dataset_id: Golden-set dataset identifier.
            version: Golden-set version string, e.g. ``golden.v1``.
            content_hash: Content hash being accessed, when known.

        Returns:
            The persisted entry, including its chain and entry hashes.

        Raises:
            ValidationError: If an identity, purpose or task ID is blank.

        """
        for name, value in (
            ("actor_id", actor_id),
            ("purpose", purpose),
            ("task_id", task_id),
            ("dataset_id", dataset_id),
            ("version", version),
        ):
            if not value.strip():
                raise ValidationError(f"Audit entry requires a non-blank {name}")
        previous = self.head_hash()
        entry = AuditEntry(
            actor_id=actor_id,
            action=action,
            purpose=purpose,
            task_id=task_id,
            dataset_id=dataset_id,
            version=version,
            content_hash=content_hash,
            previous_hash=previous,
        )
        entry = replace(entry, entry_hash=sha256_hex(canonical_json(entry.payload())))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(entry.to_dict()) + "\n")
        return entry

    def verify_chain(self) -> tuple[str, ...]:
        """Return a list of integrity errors; an empty tuple means intact."""
        errors: list[str] = []
        previous = ""
        for number, entry in enumerate(self.entries(), start=1):
            expected = sha256_hex(canonical_json(entry.payload()))
            if entry.entry_hash != expected:
                errors.append(f"audit entry {number}: hash mismatch")
            if entry.previous_hash != previous:
                errors.append(f"audit entry {number}: broken chain")
            previous = entry.entry_hash
        return tuple(errors)
