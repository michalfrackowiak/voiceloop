"""Injectable seams and deterministic in-memory fakes for P2-12.

Nothing here talks to Postgres, S3, MLflow, an identity provider or the
network. Every real integration is a :class:`typing.Protocol` with a clearly
labelled ``Fake...`` implementation, so the whole governance workflow can be
exercised deterministically on synthetic fixtures.

Audit events use fixed reason/action codes and carry a pseudonymous call
reference instead of a raw ``call_id``. Tombstones keep the exact call ID only
inside the injected sink because future dataset builds must match it exactly.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

from voiceloop_common import ValidationError, utcnow, validate


class Clock(Protocol):
    """Injectable clock; production can use :class:`SystemClock`."""

    def now(self) -> datetime:
        """Return the current UTC datetime."""
        ...


class SystemClock:
    """Real-time clock backed by ``voiceloop_common.utcnow()``."""

    def now(self) -> datetime:
        """Return the current UTC datetime."""
        return utcnow()


@dataclass(frozen=True, slots=True)
class FixedClock:
    """Deterministic fake clock returning one fixed datetime."""

    value: datetime

    def now(self) -> datetime:
        """Return the fixed datetime, interpreted as UTC when naive."""
        return self.value.replace(tzinfo=UTC) if self.value.tzinfo is None else self.value


def utc_text(value: datetime) -> str:
    """Format an injectable clock value as RFC 3339 UTC text."""
    return FixedClock(value).now().isoformat(timespec="seconds").replace("+00:00", "Z")


class ManifestSource(Protocol):
    """Read-only source of committed P2-05 dataset manifests."""

    def manifests(self) -> tuple[Mapping[str, Any], ...]:
        """Return every committed dataset manifest."""
        ...


class FakeManifestSource:
    """In-memory fake of a manifest store; validates on ingest."""

    def __init__(self, manifests: Iterable[Mapping[str, Any]] = ()) -> None:
        """Initialise the fake with optional schema-valid manifests."""
        self._items: list[dict[str, Any]] = []
        for manifest in manifests:
            self.add(manifest)

    def add(self, manifest: Mapping[str, Any]) -> None:
        """Append a deep copy after schema validation.

        Args:
            manifest: A P2-05 manifest conforming to ``dataset_manifest`` 1.0.0.

        Raises:
            ValidationError: If the manifest does not satisfy the schema.

        """
        snapshot = deepcopy(dict(manifest))
        validate(snapshot, "dataset_manifest")
        self._items.append(snapshot)

    def manifests(self) -> tuple[Mapping[str, Any], ...]:
        """Return deep copies so callers cannot mutate committed manifests."""
        return tuple(deepcopy(item) for item in self._items)


class TrainerLineageSource(Protocol):
    """Read-only source of P2-06 trainer contract output payloads."""

    def results(self) -> tuple[Mapping[str, Any], ...]:
        """Return every contract-valid trainer result."""
        ...

    def model_id_for_run(self, run_id: str) -> str | None:
        """Return the registry model ID for a run, when explicitly mapped."""
        ...


class FakeTrainerLineageSource:
    """In-memory fake of MLflow/registry lineage; validates on ingest."""

    def __init__(
        self,
        results: Iterable[Mapping[str, Any]] = (),
        model_ids: Mapping[str, str] | None = None,
    ) -> None:
        """Initialise the fake with results and optional run-to-model mapping."""
        self._items: list[dict[str, Any]] = []
        self._model_ids = dict(model_ids or {})
        for result in results:
            self.add(result)

    def add(self, result: Mapping[str, Any], *, model_id: str | None = None) -> None:
        """Append a contract-valid trainer output payload.

        The trainer schema makes ``lineage`` optional, but lineage queries must
        fail closed without it, so missing or incomplete lineage is rejected
        here rather than guessed.

        Args:
            result: A P2-06 ``TrainResponse`` payload (contract 1.0.0).
            model_id: Optional registry model ID; ``run_id`` is used otherwise.

        Raises:
            ValidationError: If the payload is not an output result with
                complete dataset lineage.

        """
        snapshot = deepcopy(dict(result))
        validate(snapshot, "trainer_contract")
        if snapshot.get("contract_part") != "output":
            raise ValidationError("trainer lineage source requires contract_part=output")
        lineage = snapshot.get("lineage")
        if not isinstance(lineage, Mapping) or not str(lineage.get("dataset_hash", "")).strip():
            raise ValidationError("trainer result has no dataset_hash lineage")
        run_id = str(snapshot.get("run_id", ""))
        if not run_id.strip() or not str(snapshot.get("artifact_uri", "")).strip():
            raise ValidationError("trainer result needs a run_id and artifact_uri")
        self._items.append(snapshot)
        if model_id:
            if not model_id.strip():
                raise ValidationError("model_id must be non-blank when supplied")
            self._model_ids[run_id] = model_id

    def results(self) -> tuple[Mapping[str, Any], ...]:
        """Return deep copies of all trainer output payloads."""
        return tuple(deepcopy(item) for item in self._items)

    def model_id_for_run(self, run_id: str) -> str | None:
        """Return an explicit registry model ID for ``run_id``, if any."""
        return self._model_ids.get(run_id)


@dataclass(frozen=True, slots=True)
class TombstoneRecord:
    """Fail-closed exclusion record for one deleted call.

    The raw ``call_id`` stays in the injected sink so builds can match it
    exactly; audit events derived from this record use ``call_ref`` instead.
    """

    call_id: str
    reason: str
    actor_id: str
    role: str
    timestamp: str
    call_ref: str
    dataset_ids: tuple[str, ...]
    content_hashes: tuple[str, ...]
    model_ids: tuple[str, ...]


class TombstoneSink(Protocol):
    """Durable (injectable) deny-list of calls excluded from future datasets."""

    def deny(self, record: TombstoneRecord) -> None:
        """Add a call to the deny-list; repeating a denial is a no-op."""
        ...

    def is_denied(self, call_id: str) -> bool:
        """Return whether the exact call ID is denied."""
        ...

    def records(self) -> tuple[TombstoneRecord, ...]:
        """Return all tombstone records in deterministic order."""
        ...


class ExclusionSource(Protocol):
    """Call IDs a dataset build must exclude."""

    def excluded_call_ids(self) -> frozenset[str]:
        """Return all tombstoned call IDs."""
        ...


class InMemoryTombstoneSink:
    """Fake append-only tombstone store and exclusion source."""

    def __init__(self) -> None:
        """Initialise an empty deny-list."""
        self._records: dict[str, TombstoneRecord] = {}

    def deny(self, record: TombstoneRecord) -> None:
        """Record a tombstone once; a repeated denial cannot weaken it."""
        if not isinstance(record.call_id, str) or not record.call_id:
            raise ValidationError("tombstone requires a non-empty call_id")
        self._records.setdefault(record.call_id, record)

    def is_denied(self, call_id: str) -> bool:
        """Return whether ``call_id`` is in the deny-list."""
        return call_id in self._records

    def records(self) -> tuple[TombstoneRecord, ...]:
        """Return records sorted by call ID."""
        return tuple(self._records[key] for key in sorted(self._records))

    def excluded_call_ids(self) -> frozenset[str]:
        """Return the exact call IDs a build must exclude."""
        return frozenset(self._records)

    def record_for(self, call_id: str) -> TombstoneRecord | None:
        """Return the stored tombstone for ``call_id``, if any."""
        return self._records.get(call_id)


class ModelFlag(StrEnum):
    """Non-destructive registry flags applied to affected models."""

    TAINTED = "tainted"
    NEEDS_RETRAIN = "needs_retrain"


@dataclass(frozen=True, slots=True)
class ModelFlagRecord:
    """One flag applied to one model; no alias or promotion fields exist."""

    model_id: str
    flag: ModelFlag
    reason: str
    actor_id: str
    timestamp: str
    call_ref: str
    dataset_ids: tuple[str, ...]


class ModelFlagSink(Protocol):
    """Injectable model-registry flag sink; never flips aliases."""

    def flag(self, record: ModelFlagRecord) -> None:
        """Apply a flag to a model, deduplicating exact repeats."""
        ...

    def records(self) -> tuple[ModelFlagRecord, ...]:
        """Return all flag records in deterministic order."""
        ...


class InMemoryModelFlagSink:
    """Fake append-only model flag store."""

    def __init__(self) -> None:
        """Initialise an empty flag store."""
        self._records: list[ModelFlagRecord] = []

    def flag(self, record: ModelFlagRecord) -> None:
        """Append a flag unless the exact flag record already exists."""
        if record not in self._records:
            self._records.append(record)

    def records(self) -> tuple[ModelFlagRecord, ...]:
        """Return records sorted by model, flag and reason."""
        return tuple(
            sorted(
                self._records,
                key=lambda item: (item.model_id, item.flag.value, item.reason, item.call_ref),
            )
        )

    def flagged_model_ids(self) -> tuple[str, ...]:
        """Return every flagged model ID in deterministic order."""
        return tuple(sorted({record.model_id for record in self._records}))


class AuditEventType(StrEnum):
    """Fixed audit event kinds used by this package."""

    AUTHORIZATION = "authorization"
    DELETION = "deletion"
    DELETION_DUPLICATE = "deletion_duplicate"
    DELETION_FAILED = "deletion_failed"


class AuditDecision(StrEnum):
    """Fixed decision codes; no free-form outcome text is accepted."""

    ALLOW = "allow"
    DENY = "deny"
    COMPLETED = "completed"
    DUPLICATE = "duplicate"
    FAILED = "failed"


class AuditAction(StrEnum):
    """Closed action codes stored in audit events.

    The first eight values mirror ``roles.Capability``; the two sentinels keep
    invalid inputs and workflow side effects out of the free-form string space.
    """

    REVIEW_READ = "review_read"
    REVIEW_WRITE = "review_write"
    DATASET_BUILD = "dataset_build"
    TRAIN = "train"
    LINEAGE_READ = "lineage_read"
    DELETION_AUTHORIZE = "deletion_authorize"
    CONSENT_CHANGE = "consent_change"
    AUDIT_READ = "audit_read"
    DELETION_EXECUTE = "deletion_execute"
    UNKNOWN_ACTION = "unknown_action"


class AuditReason(StrEnum):
    """Closed reason codes; no free-form request or error text is stored."""

    CAPABILITY_GRANTED = "capability_granted"
    CAPABILITY_MISSING = "capability_missing"
    UNKNOWN_ACTION = "unknown_action"
    UNKNOWN_RESOURCE = "unknown_resource"
    UNKNOWN_PURPOSE = "unknown_purpose"
    ERASURE_REQUEST = "erasure_request"
    CONSENT_WITHDRAWN = "consent_withdrawn"
    COMPLIANCE_REVIEW = "compliance_review"
    UNKNOWN_CALL = "unknown_call"
    SINK_FAILURE = "sink_failure"


class Resource(StrEnum):
    """Fixed resource codes passed to authorisation checks."""

    DELETION = "deletion"
    LINEAGE = "lineage"
    DATASET_BUILD = "dataset_build"
    TRAINING = "training"
    REVIEW = "review"
    CONSENT = "consent"
    AUDIT = "audit"
    UNKNOWN_RESOURCE = "unknown_resource"


class AuditPurpose(StrEnum):
    """Fixed purpose codes; never free-form request text."""

    GOVERNANCE = "governance"
    OPERATION = "operation"
    QUERY = "query"
    UNKNOWN_PURPOSE = "unknown_purpose"


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """One PII-free, hash-chained audit event.

    ``call_ref`` is a one-way pseudonym; ``actor_id`` must be a short
    pseudonymous identifier. Model and dataset IDs are exact, non-guest IDs.
    """

    event_type: AuditEventType
    timestamp: str
    actor_id: str
    role: str
    action: AuditAction
    decision: AuditDecision
    resource: Resource
    purpose: AuditPurpose
    reason: AuditReason
    call_ref: str = ""
    model_ids: tuple[str, ...] = ()
    dataset_ids: tuple[str, ...] = ()
    previous_hash: str = ""
    entry_hash: str = ""

    def payload(self) -> dict[str, Any]:
        """Return the fields covered by the entry hash."""
        data = asdict(self)
        data.pop("entry_hash")
        return data

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of the event."""
        return asdict(self)


class AuditSink(Protocol):
    """Append-only audit sink; the real adapter may write to a secure store."""

    def record(self, event: AuditEvent) -> AuditEvent:
        """Append an event and return it with chain hashes filled in."""
        ...

    def events(self) -> tuple[AuditEvent, ...]:
        """Return all events in append order."""
        ...


def canonical_json(payload: Any) -> str:
    """Return stable compact JSON for hashing audit events."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    """Return the SHA-256 hex digest of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pseudonymize_call_id(call_id: str, *, domain: str = "voiceloop-lineage-v1") -> str:
    """Return a deterministic one-way reference for a call ID.

    The domain is a public domain separator, not a secret. Call IDs may still
    be used verbatim in query results, but never in audit events or logs.

    Args:
        call_id: Exact call ID to pseudonymise.
        domain: Stable domain string from ``LineageSettings``.

    Returns:
        A ``call_sha256:<32 hex>`` reference that cannot be reversed.

    """
    digest = sha256_hex(f"{domain}:{call_id}")
    return f"call_sha256:{digest[:32]}"


class InMemoryAuditSink:
    """Fake append-only, hash-chained audit sink for deterministic tests."""

    def __init__(self) -> None:
        """Initialise an empty audit chain."""
        self._events: tuple[AuditEvent, ...] = ()

    def record(self, event: AuditEvent) -> AuditEvent:
        """Append one event with tamper-evidence hashes.

        Raises:
            ValidationError: If required, PII-free fields are blank or malformed.

        """
        if not event.actor_id.strip() or not event.role.strip():
            raise ValidationError("audit event needs a non-blank actor_id and role")
        if not isinstance(event.event_type, AuditEventType) or not isinstance(
            event.decision, AuditDecision
        ):
            raise ValidationError("audit event needs fixed event_type and decision codes")
        if not isinstance(event.action, AuditAction):
            raise ValidationError("audit event needs a fixed action code")
        if not isinstance(event.resource, Resource):
            raise ValidationError("audit event needs a fixed resource code")
        if not isinstance(event.purpose, AuditPurpose):
            raise ValidationError("audit event needs a fixed purpose code")
        if not isinstance(event.reason, AuditReason):
            raise ValidationError("audit event needs a fixed reason code")
        chained = replace(
            event,
            previous_hash=self.head_hash(),
        )
        chained = replace(chained, entry_hash=sha256_hex(canonical_json(chained.payload())))
        self._events += (chained,)
        return chained

    def events(self) -> tuple[AuditEvent, ...]:
        """Return all audit events in append order."""
        return self._events

    def head_hash(self) -> str:
        """Return the newest entry hash, or an empty string for an empty log."""
        return self._events[-1].entry_hash if self._events else ""

    def verify_chain(self) -> tuple[str, ...]:
        """Return integrity errors; an empty tuple means the chain is intact."""
        errors: list[str] = []
        previous = ""
        for index, event in enumerate(self._events, start=1):
            if event.entry_hash != sha256_hex(canonical_json(event.payload())):
                errors.append(f"audit event {index}: hash mismatch")
            if event.previous_hash != previous:
                errors.append(f"audit event {index}: broken chain")
            previous = event.entry_hash
        return tuple(errors)


__all__ = [
    "AuditAction",
    "AuditDecision",
    "AuditEvent",
    "AuditEventType",
    "AuditPurpose",
    "AuditReason",
    "AuditSink",
    "Clock",
    "ExclusionSource",
    "FakeManifestSource",
    "FakeTrainerLineageSource",
    "FixedClock",
    "InMemoryAuditSink",
    "InMemoryModelFlagSink",
    "InMemoryTombstoneSink",
    "ManifestSource",
    "ModelFlag",
    "ModelFlagRecord",
    "ModelFlagSink",
    "Resource",
    "SystemClock",
    "TombstoneRecord",
    "TombstoneSink",
    "TrainerLineageSource",
    "canonical_json",
    "pseudonymize_call_id",
    "sha256_hex",
    "utc_text",
]
