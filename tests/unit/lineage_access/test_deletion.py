"""P2-12.2: fail-closed deletion, exclusion guard and model flags."""

from __future__ import annotations

import json
from typing import Any

import pytest

from services.datasets import build_dataset_from_fixtures, build_manifest
from services.datasets.auth import AuthorizationError, Authorizer
from services.datasets.lineage import (
    AuditAction,
    AuditDecision,
    AuditEventType,
    DeletedCallInBuildError,
    DeletionError,
    DeletionReason,
    DeletionWorkflow,
    FixedClock,
    InMemoryAuditSink,
    InMemoryModelFlagSink,
    InMemoryTombstoneSink,
    LineageGraph,
    ModelFlag,
    ModelFlagRecord,
    ModelFlagSink,
    Principal,
    UnknownCallError,
    guard_manifest,
    pseudonymize_call_id,
)
from tests.unit.lineage_access.conftest import CALL_A, MANIFEST_TIME, UNKNOWN_CALL
from voiceloop_common import StorageError

RUN_PARSER = "run-parser-v1"
RUN_STT = "run-stt-v1"
MODEL_PARSER = "model-parser-v1"


class FailingFlagSink:
    """Injected sink that fails on the first flag write."""

    def flag(self, record: ModelFlagRecord) -> None:
        """Always fail."""
        raise StorageError("registry unavailable")

    def records(self) -> tuple[ModelFlagRecord, ...]:
        """Return no records."""
        return ()


def test_admin_deletion_audits_impacts_and_flags_models(
    workflow: DeletionWorkflow,
    tombstones: InMemoryTombstoneSink,
    model_flags: InMemoryModelFlagSink,
    audit_sink: InMemoryAuditSink,
    admin: Principal,
) -> None:
    report = workflow.delete(CALL_A, admin, DeletionReason.ERASURE_REQUEST)
    payload = json.dumps(report.to_dict())

    assert report.status == "completed"
    assert report.reason == "erasure_request"
    assert report.call_ref == pseudonymize_call_id(CALL_A)
    assert report.actor_id == admin.actor_id
    assert report.role == "admin"
    assert report.timestamp == "2026-01-02T03:04:05Z"
    assert CALL_A not in payload
    assert len(report.datasets) == 1
    assert {model.model_id for model in report.models} == {MODEL_PARSER, RUN_STT}
    assert set(report.model_flags) == {
        (MODEL_PARSER, ModelFlag.TAINTED),
        (MODEL_PARSER, ModelFlag.NEEDS_RETRAIN),
        (RUN_STT, ModelFlag.TAINTED),
        (RUN_STT, ModelFlag.NEEDS_RETRAIN),
    }
    assert tombstones.is_denied(CALL_A)

    events = audit_sink.events()
    assert [event.event_type for event in events] == [
        AuditEventType.AUTHORIZATION,
        AuditEventType.DELETION,
    ]
    deletion = events[1]
    assert deletion.action is AuditAction.DELETION_EXECUTE
    assert deletion.decision is AuditDecision.COMPLETED
    assert deletion.actor_id == admin.actor_id
    assert deletion.role == "admin"
    assert deletion.reason == "erasure_request"
    assert deletion.timestamp == "2026-01-02T03:04:05Z"
    assert deletion.call_ref == report.call_ref
    assert set(deletion.model_ids) == {MODEL_PARSER, RUN_STT}
    assert audit_sink.verify_chain() == ()
    assert CALL_A not in json.dumps([event.to_dict() for event in events])

    flags = model_flags.records()
    assert len(flags) == 4
    assert all(flag.call_ref == report.call_ref for flag in flags)
    assert all(not hasattr(flag, "alias") for flag in flags)


@pytest.mark.parametrize(
    ("principal_name", "role_name"),
    [("reviewer", "reviewer"), ("ml_engineer", "ml_engineer")],
)
def test_non_admin_deletion_is_denied_without_side_effects(
    workflow: DeletionWorkflow,
    tombstones: InMemoryTombstoneSink,
    model_flags: InMemoryModelFlagSink,
    audit_sink: InMemoryAuditSink,
    request: pytest.FixtureRequest,
    principal_name: str,
    role_name: str,
) -> None:
    principal = request.getfixturevalue(principal_name)

    with pytest.raises(AuthorizationError, match="lacks capability deletion_authorize"):
        workflow.delete(CALL_A, principal, DeletionReason.ERASURE_REQUEST)

    assert not tombstones.is_denied(CALL_A)
    assert model_flags.records() == ()
    events = audit_sink.events()
    assert len(events) == 1
    assert events[0].event_type is AuditEventType.AUTHORIZATION
    assert events[0].decision is AuditDecision.DENY
    assert events[0].role == role_name


def test_unknown_call_fails_closed(
    workflow: DeletionWorkflow,
    tombstones: InMemoryTombstoneSink,
    model_flags: InMemoryModelFlagSink,
    audit_sink: InMemoryAuditSink,
    admin: Principal,
) -> None:
    with pytest.raises(UnknownCallError):
        workflow.delete(UNKNOWN_CALL, admin, DeletionReason.ERASURE_REQUEST)

    assert tombstones.excluded_call_ids() == frozenset()
    assert model_flags.records() == ()
    events = audit_sink.events()
    assert [event.event_type for event in events] == [
        AuditEventType.AUTHORIZATION,
        AuditEventType.DELETION_FAILED,
    ]
    assert events[1].reason == "unknown_call"
    assert UNKNOWN_CALL not in json.dumps([event.to_dict() for event in events])


def test_repeated_deletion_is_duplicate_and_does_not_duplicate_flags(
    workflow: DeletionWorkflow,
    model_flags: InMemoryModelFlagSink,
    audit_sink: InMemoryAuditSink,
    admin: Principal,
) -> None:
    first = workflow.delete(CALL_A, admin, DeletionReason.ERASURE_REQUEST)
    second = workflow.delete(CALL_A, admin, DeletionReason.COMPLIANCE_REVIEW)

    assert first.status == "completed"
    assert second.status == "duplicate"
    assert len(model_flags.records()) == 4
    assert [event.event_type for event in audit_sink.events()] == [
        AuditEventType.AUTHORIZATION,
        AuditEventType.DELETION,
        AuditEventType.AUTHORIZATION,
        AuditEventType.DELETION_DUPLICATE,
    ]


def test_deleted_call_is_excluded_from_future_dataset_builds(
    workflow: DeletionWorkflow,
    tombstones: InMemoryTombstoneSink,
    manifest: dict[str, Any],
    admin: Principal,
) -> None:
    guard_manifest(manifest, tombstones)  # no tombstones yet

    workflow.delete(CALL_A, admin, DeletionReason.CONSENT_WITHDRAWN)

    future_build = build_dataset_from_fixtures(regression_call_ids=tombstones.excluded_call_ids())
    future_manifest = build_manifest(future_build, created_at=MANIFEST_TIME)
    assert CALL_A not in future_manifest["source_call_ids"]
    result = guard_manifest(future_manifest, tombstones)
    assert result.blocked_count == 0
    assert result.content_hash == future_manifest["content_hash"]

    with pytest.raises(DeletedCallInBuildError) as excinfo:
        guard_manifest(manifest, tombstones)
    assert CALL_A not in str(excinfo.value)
    assert pseudonymize_call_id(CALL_A) in str(excinfo.value)


def test_flag_sink_failure_keeps_tombstone_and_audits_failure(
    graph: LineageGraph,
    authorizer: Authorizer,
    tombstones: InMemoryTombstoneSink,
    audit_sink: InMemoryAuditSink,
    clock: FixedClock,
    admin: Principal,
) -> None:
    workflow = DeletionWorkflow(
        graph=graph,
        authorizer=authorizer,
        tombstones=tombstones,
        model_flags=FailingFlagSink(),
        audit_sink=audit_sink,
        clock=clock,
    )

    with pytest.raises(DeletionError, match="call remains excluded"):
        workflow.delete(CALL_A, admin, DeletionReason.ERASURE_REQUEST)

    assert tombstones.is_denied(CALL_A)
    assert [event.event_type for event in audit_sink.events()] == [
        AuditEventType.AUTHORIZATION,
        AuditEventType.DELETION_FAILED,
    ]
    assert audit_sink.events()[1].reason == "sink_failure"


def test_model_flag_protocol_accepts_deterministic_fake() -> None:
    memory_sink = InMemoryModelFlagSink()
    sink: ModelFlagSink = memory_sink
    record = ModelFlagRecord(
        model_id="model-x",
        flag=ModelFlag.TAINTED,
        reason="erasure_request",
        actor_id="admin-1",
        timestamp="2026-01-02T03:04:05Z",
        call_ref=pseudonymize_call_id(CALL_A),
        dataset_ids=("dataset-x",),
    )
    sink.flag(record)
    sink.flag(record)

    assert sink.records() == (record,)
    assert memory_sink.flagged_model_ids() == ("model-x",)
