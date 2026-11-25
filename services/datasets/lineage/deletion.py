"""Fail-closed deletion workflow and future-build exclusion guard (P2-12.2).

Deletion here is **not physical deletion**. It writes an injectable tombstone /
deny-list entry, flags affected models as ``tainted`` and ``needs_retrain``
without touching MLflow aliases or promotion state, and emits an auditable
impact report. A future dataset build must consult the same exclusion source;
:func:`guard_manifest` refuses any manifest that still contains a tombstoned
call. Unknown calls, missing manifests and denied authorisation raise rather
than returning a silent success.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from voiceloop_common import ValidationError, VoiceLoopError, validate

from .graph import DatasetRef, LineageGraph, ModelRef, UnknownCallError
from .roles import Capability, Principal
from .settings import LineageSettings
from .store import (
    AuditAction,
    AuditDecision,
    AuditEvent,
    AuditEventType,
    AuditPurpose,
    AuditReason,
    AuditSink,
    Clock,
    ExclusionSource,
    ModelFlag,
    ModelFlagRecord,
    ModelFlagSink,
    Resource,
    TombstoneRecord,
    TombstoneSink,
    pseudonymize_call_id,
    utc_text,
)

if TYPE_CHECKING:
    from ..auth.policy import Authorizer


class DeletionReason(StrEnum):
    """Closed set of reason codes; no free-form text reaches audit events."""

    ERASURE_REQUEST = "erasure_request"
    CONSENT_WITHDRAWN = "consent_withdrawn"
    COMPLIANCE_REVIEW = "compliance_review"


class DeletedCallInBuildError(VoiceLoopError):
    """A manifest contains calls that are tombstoned for deletion."""


class DeletionError(VoiceLoopError):
    """A workflow side effect failed after authorisation succeeded."""


@dataclass(frozen=True, slots=True)
class DeletionReport:
    """PII-free impact report for one deletion decision.

    The raw call ID is replaced by a one-way ``call_ref``. Datasets and models
    are reported with their exact IDs so a human can act on the impact.
    """

    call_ref: str
    status: str
    reason: str
    timestamp: str
    actor_id: str
    role: str
    datasets: tuple[DatasetRef, ...]
    models: tuple[ModelRef, ...]
    model_flags: tuple[tuple[str, ModelFlag], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of the report."""
        return {
            "call_ref": self.call_ref,
            "status": self.status,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "actor_id": self.actor_id,
            "role": self.role,
            # Omit source_call_ids: the report is distributed to humans and
            # must not re-expose the deleted call identifier.
            "datasets": [
                {
                    "dataset_id": dataset.dataset_id,
                    "content_hash": dataset.content_hash,
                    "created_at": dataset.created_at,
                }
                for dataset in self.datasets
            ],
            "models": [asdict(model) for model in self.models],
            "model_flags": [
                {"model_id": model_id, "flag": flag.value} for model_id, flag in self.model_flags
            ],
        }


@dataclass(frozen=True, slots=True)
class ManifestGuardResult:
    """Result of a successful future-build exclusion check."""

    dataset_id: str
    content_hash: str
    source_call_count: int
    blocked_count: int = 0


def guard_manifest(
    manifest: Any,
    exclusion_source: ExclusionSource,
    *,
    domain: str = "voiceloop-lineage-v1",
) -> ManifestGuardResult:
    """Fail a dataset build whose manifest still contains tombstoned calls.

    Args:
        manifest: Manifest from a P2-05 build, validated against schema 1.0.0.
        exclusion_source: Source of exact call IDs excluded by deletion.
        domain: Pseudonym domain used only in the error message.

    Returns:
        A guard result with zero blocked calls.

    Raises:
        ValidationError: If the manifest is not schema-valid.
        DeletedCallInBuildError: If any tombstoned call is present.

    """
    validate(manifest, "dataset_manifest")
    calls = tuple(str(call_id) for call_id in manifest["source_call_ids"])
    blocked = sorted(set(calls) & set(exclusion_source.excluded_call_ids()))
    if blocked:
        refs = ", ".join(pseudonymize_call_id(call_id, domain=domain) for call_id in blocked)
        raise DeletedCallInBuildError(f"tombstoned calls remain in dataset build: {refs}")
    return ManifestGuardResult(
        dataset_id=str(manifest["dataset_id"]),
        content_hash=str(manifest["content_hash"]),
        source_call_count=len(calls),
    )


class DeletionWorkflow:
    """Authorise, tombstone, flag and report a call deletion request."""

    def __init__(
        self,
        *,
        graph: LineageGraph,
        authorizer: Authorizer,
        tombstones: TombstoneSink,
        model_flags: ModelFlagSink,
        audit_sink: AuditSink,
        clock: Clock,
        settings: LineageSettings | None = None,
    ) -> None:
        """Wire the workflow to injected read-only sources and sinks."""
        self._graph = graph
        self._authorizer = authorizer
        self._tombstones = tombstones
        self._model_flags = model_flags
        self._audit_sink = audit_sink
        self._clock = clock
        self._settings = settings or LineageSettings()

    def _audit(
        self,
        *,
        event_type: AuditEventType,
        decision: AuditDecision,
        reason: AuditReason,
        principal: Principal,
        call_ref: str,
        dataset_ids: tuple[str, ...],
        model_ids: tuple[str, ...],
    ) -> AuditEvent:
        """Append a PII-free deletion audit event with fixed action/reason codes."""
        return self._audit_sink.record(
            AuditEvent(
                event_type=event_type,
                timestamp=utc_text(self._clock.now()),
                actor_id=principal.actor_id,
                role=principal.role.value,
                action=AuditAction.DELETION_EXECUTE,
                decision=decision,
                resource=Resource.DELETION,
                purpose=AuditPurpose.GOVERNANCE,
                reason=reason,
                call_ref=call_ref,
                model_ids=model_ids,
                dataset_ids=dataset_ids,
            )
        )

    def delete(self, call_id: str, principal: Principal, reason: DeletionReason) -> DeletionReport:
        """Authorise and process one deletion request.

        Args:
            call_id: Exact source call ID to tombstone.
            principal: Principal with the ``deletion_authorize`` capability.
            reason: Closed reason code recorded in the audit event.

        Returns:
            An impact report; repeated requests return ``status="duplicate"``.

        Raises:
            AuthorizationError: If the principal lacks deletion authority.
            UnknownCallError: If no committed manifest contains the call.
            DeletionError: If a tombstone or flag sink fails after approval.

        """
        if not isinstance(reason, DeletionReason):
            raise ValidationError("deletion reason must be a DeletionReason code")
        call_ref = pseudonymize_call_id(call_id, domain=self._settings.pseudonym_domain)
        self._authorizer.require(
            principal,
            Capability.DELETION_AUTHORIZE,
            Resource.DELETION,
            purpose=AuditPurpose.GOVERNANCE,
            call_ref=call_ref,
        )

        try:
            datasets = self._graph.datasets_for_call(call_id)
            models = self._graph.models_for_call(call_id)
        except UnknownCallError:
            self._audit(
                event_type=AuditEventType.DELETION_FAILED,
                decision=AuditDecision.FAILED,
                reason=AuditReason.UNKNOWN_CALL,
                principal=principal,
                call_ref=call_ref,
                dataset_ids=(),
                model_ids=(),
            )
            raise

        dataset_ids = tuple(dataset.dataset_id for dataset in datasets)
        content_hashes = tuple(dict.fromkeys(dataset.content_hash for dataset in datasets))
        model_ids = tuple(dict.fromkeys(model.model_id for model in models))
        timestamp = utc_text(self._clock.now())

        if self._tombstones.is_denied(call_id):
            self._audit(
                event_type=AuditEventType.DELETION_DUPLICATE,
                decision=AuditDecision.DUPLICATE,
                reason=AuditReason(reason.value),
                principal=principal,
                call_ref=call_ref,
                dataset_ids=dataset_ids,
                model_ids=model_ids,
            )
            return DeletionReport(
                call_ref=call_ref,
                status="duplicate",
                reason=reason.value,
                timestamp=timestamp,
                actor_id=principal.actor_id,
                role=principal.role.value,
                datasets=datasets,
                models=models,
                model_flags=tuple(),
            )

        record = TombstoneRecord(
            call_id=call_id,
            reason=reason.value,
            actor_id=principal.actor_id,
            role=principal.role.value,
            timestamp=timestamp,
            call_ref=call_ref,
            dataset_ids=dataset_ids,
            content_hashes=content_hashes,
            model_ids=model_ids,
        )
        try:
            # Tombstone first: even if flagging fails, the call stays excluded.
            self._tombstones.deny(record)
            applied = self._flag_models(models, principal, reason, call_ref, dataset_ids, timestamp)
        except Exception as exc:
            self._audit(
                event_type=AuditEventType.DELETION_FAILED,
                decision=AuditDecision.FAILED,
                reason=AuditReason.SINK_FAILURE,
                principal=principal,
                call_ref=call_ref,
                dataset_ids=dataset_ids,
                model_ids=model_ids,
            )
            raise DeletionError("deletion sink failed; call remains excluded") from exc

        self._audit(
            event_type=AuditEventType.DELETION,
            decision=AuditDecision.COMPLETED,
            reason=AuditReason(reason.value),
            principal=principal,
            call_ref=call_ref,
            dataset_ids=dataset_ids,
            model_ids=model_ids,
        )
        return DeletionReport(
            call_ref=call_ref,
            status="completed",
            reason=reason.value,
            timestamp=timestamp,
            actor_id=principal.actor_id,
            role=principal.role.value,
            datasets=datasets,
            models=models,
            model_flags=applied,
        )

    def _flag_models(
        self,
        models: tuple[ModelRef, ...],
        principal: Principal,
        reason: DeletionReason,
        call_ref: str,
        dataset_ids: tuple[str, ...],
        timestamp: str,
    ) -> tuple[tuple[str, ModelFlag], ...]:
        """Apply both governance flags to every affected model."""
        applied: list[tuple[str, ModelFlag]] = []
        for model in models:
            for flag in (ModelFlag.TAINTED, ModelFlag.NEEDS_RETRAIN):
                self._model_flags.flag(
                    ModelFlagRecord(
                        model_id=model.model_id,
                        flag=flag,
                        reason=reason.value,
                        actor_id=principal.actor_id,
                        timestamp=timestamp,
                        call_ref=call_ref,
                        dataset_ids=dataset_ids,
                    )
                )
                applied.append((model.model_id, flag))
        return tuple(applied)


__all__ = [
    "DeletionError",
    "DeletionReason",
    "DeletionReport",
    "DeletionWorkflow",
    "DeletedCallInBuildError",
    "ManifestGuardResult",
    "guard_manifest",
]
