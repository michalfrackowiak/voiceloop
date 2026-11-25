"""P2-12.3: default-deny roles, non-escalation and audited decisions."""

from __future__ import annotations

import json
from collections.abc import MutableMapping
from dataclasses import replace
from typing import cast

import pytest

import services.datasets.lineage as lineage_package
from services.datasets.auth import (
    AuthorizationError,
    Authorizer,
    assert_pseudonymous_call_ref,
)
from services.datasets.lineage import (
    AuditAction,
    AuditDecision,
    AuditEvent,
    AuditEventType,
    AuditPurpose,
    AuditReason,
    Capability,
    DeletionReason,
    DeletionWorkflow,
    FixedClock,
    InMemoryAuditSink,
    InMemoryModelFlagSink,
    InMemoryTombstoneSink,
    Principal,
    Resource,
    Role,
    capabilities_for,
    pseudonymize_call_id,
)
from services.datasets.lineage.roles import ROLE_CAPABILITIES
from tests.unit.lineage_access.conftest import CALL_A, FIXED_NOW
from voiceloop_common import ValidationError

ROLE_MATRIX = [
    (Role.REVIEWER, Capability.REVIEW_READ, True),
    (Role.REVIEWER, Capability.REVIEW_WRITE, True),
    (Role.REVIEWER, Capability.DATASET_BUILD, False),
    (Role.REVIEWER, Capability.TRAIN, False),
    (Role.REVIEWER, Capability.LINEAGE_READ, False),
    (Role.REVIEWER, Capability.DELETION_AUTHORIZE, False),
    (Role.REVIEWER, Capability.CONSENT_CHANGE, False),
    (Role.REVIEWER, Capability.AUDIT_READ, False),
    (Role.ML_ENGINEER, Capability.DATASET_BUILD, True),
    (Role.ML_ENGINEER, Capability.TRAIN, True),
    (Role.ML_ENGINEER, Capability.LINEAGE_READ, True),
    (Role.ML_ENGINEER, Capability.REVIEW_READ, False),
    (Role.ML_ENGINEER, Capability.REVIEW_WRITE, False),
    (Role.ML_ENGINEER, Capability.DELETION_AUTHORIZE, False),
    (Role.ML_ENGINEER, Capability.CONSENT_CHANGE, False),
    (Role.ML_ENGINEER, Capability.AUDIT_READ, False),
    (Role.ADMIN, Capability.LINEAGE_READ, True),
    (Role.ADMIN, Capability.DELETION_AUTHORIZE, True),
    (Role.ADMIN, Capability.CONSENT_CHANGE, True),
    (Role.ADMIN, Capability.AUDIT_READ, True),
    (Role.ADMIN, Capability.REVIEW_WRITE, False),
    (Role.ADMIN, Capability.DATASET_BUILD, False),
    (Role.ADMIN, Capability.TRAIN, False),
]


@pytest.mark.parametrize(("role", "capability", "allowed"), ROLE_MATRIX)
def test_capability_matrix_is_default_deny_and_audited(
    role: Role, capability: Capability, allowed: bool
) -> None:
    sink = InMemoryAuditSink()
    authorizer = Authorizer(sink, FixedClock(FIXED_NOW))
    principal = Principal("actor-1", role)

    decision = authorizer.check(
        principal,
        capability,
        Resource.LINEAGE,
        purpose=AuditPurpose.QUERY,
    )

    expected_reason = AuditReason.CAPABILITY_GRANTED if allowed else AuditReason.CAPABILITY_MISSING
    assert decision.allowed is allowed
    assert decision.reason is expected_reason
    assert len(sink.events()) == 1
    event = sink.events()[0]
    assert event.event_type is AuditEventType.AUTHORIZATION
    assert event.decision is (AuditDecision.ALLOW if allowed else AuditDecision.DENY)
    assert event.role == role.value
    assert event.action is AuditAction(capability.value)
    assert event.reason is expected_reason
    assert event.actor_id == "actor-1"
    assert event.timestamp == "2026-01-02T03:04:05Z"
    assert sink.verify_chain() == ()


def test_require_raises_and_still_audits_denial() -> None:
    sink = InMemoryAuditSink()
    authorizer = Authorizer(sink, FixedClock(FIXED_NOW))

    with pytest.raises(AuthorizationError, match="reviewer lacks capability deletion_authorize"):
        authorizer.require(
            Principal("reviewer-1", Role.REVIEWER),
            Capability.DELETION_AUTHORIZE,
            Resource.DELETION,
            purpose=AuditPurpose.GOVERNANCE,
        )
    assert [event.decision for event in sink.events()] == [AuditDecision.DENY]


def test_role_matrix_is_immutable_and_reviewer_cannot_escalate(
    workflow: DeletionWorkflow,
    authorizer: Authorizer,
    reviewer: Principal,
    tombstones: InMemoryTombstoneSink,
    model_flags: InMemoryModelFlagSink,
    audit_sink: InMemoryAuditSink,
) -> None:
    mutable_view = cast(
        MutableMapping[Role, frozenset[Capability]],
        ROLE_CAPABILITIES,
    )
    with pytest.raises(TypeError):
        mutable_view[Role.REVIEWER] = frozenset(
            {Capability.REVIEW_READ, Capability.REVIEW_WRITE, Capability.DELETION_AUTHORIZE}
        )
    with pytest.raises(AttributeError):
        cast(set[Capability], ROLE_CAPABILITIES[Role.REVIEWER]).add(Capability.CONSENT_CHANGE)
    assert not hasattr(lineage_package, "ROLE_CAPABILITIES")
    assert capabilities_for(Role.REVIEWER) == frozenset(
        {Capability.REVIEW_READ, Capability.REVIEW_WRITE}
    )

    with pytest.raises(AuthorizationError, match="lacks capability deletion_authorize"):
        workflow.delete(CALL_A, reviewer, DeletionReason.ERASURE_REQUEST)
    with pytest.raises(AuthorizationError, match="lacks capability consent_change"):
        authorizer.require(
            reviewer,
            Capability.CONSENT_CHANGE,
            Resource.CONSENT,
            purpose=AuditPurpose.GOVERNANCE,
        )

    assert not tombstones.is_denied(CALL_A)
    assert model_flags.records() == ()
    assert [event.decision for event in audit_sink.events()] == [
        AuditDecision.DENY,
        AuditDecision.DENY,
    ]


def test_roles_cannot_be_escalated_from_strings_or_spoofed_claims() -> None:
    with pytest.raises(ValidationError):
        Principal("actor-1", "admin")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        Principal("admin@example.com", Role.ADMIN)
    with pytest.raises(ValidationError):
        capabilities_for("admin")  # type: ignore[arg-type]
    assert Capability.DELETION_AUTHORIZE not in capabilities_for(Role.REVIEWER)
    assert Capability.DELETION_AUTHORIZE not in capabilities_for(Role.ML_ENGINEER)
    assert Capability.CONSENT_CHANGE not in capabilities_for(Role.ML_ENGINEER)


@pytest.mark.parametrize(
    (
        "capability",
        "resource",
        "purpose",
        "expected_action",
        "expected_resource",
        "expected_purpose",
        "expected_reason",
    ),
    [
        (
            "delete_all",
            Resource.DELETION,
            AuditPurpose.GOVERNANCE,
            AuditAction.UNKNOWN_ACTION,
            Resource.DELETION,
            AuditPurpose.GOVERNANCE,
            AuditReason.UNKNOWN_ACTION,
        ),
        (
            Capability.LINEAGE_READ,
            "custom_resource",
            AuditPurpose.QUERY,
            AuditAction.LINEAGE_READ,
            Resource.UNKNOWN_RESOURCE,
            AuditPurpose.QUERY,
            AuditReason.UNKNOWN_RESOURCE,
        ),
        (
            Capability.LINEAGE_READ,
            Resource.LINEAGE,
            "custom_purpose",
            AuditAction.LINEAGE_READ,
            Resource.LINEAGE,
            AuditPurpose.UNKNOWN_PURPOSE,
            AuditReason.UNKNOWN_PURPOSE,
        ),
    ],
)
def test_unknown_authorization_inputs_are_denied_and_audited(
    capability: object,
    resource: object,
    purpose: object,
    expected_action: AuditAction,
    expected_resource: Resource,
    expected_purpose: AuditPurpose,
    expected_reason: AuditReason,
) -> None:
    sink = InMemoryAuditSink()
    authorizer = Authorizer(sink, FixedClock(FIXED_NOW))
    admin = Principal("admin-1", Role.ADMIN)

    with pytest.raises(AuthorizationError, match="invalid authorization input"):
        authorizer.check(
            admin,
            cast(Capability, capability),
            cast(Resource, resource),
            purpose=cast(AuditPurpose, purpose),
        )

    assert len(sink.events()) == 1
    event = sink.events()[0]
    assert event.event_type is AuditEventType.AUTHORIZATION
    assert event.decision is AuditDecision.DENY
    assert event.action is expected_action
    assert event.resource is expected_resource
    assert event.purpose is expected_purpose
    assert event.reason is expected_reason
    serialized = json.dumps(event.to_dict())
    for raw_value in ("delete_all", "custom_resource", "custom_purpose"):
        assert raw_value not in serialized
    assert sink.verify_chain() == ()


def test_require_with_unknown_action_audits_denial_once() -> None:
    sink = InMemoryAuditSink()
    authorizer = Authorizer(sink, FixedClock(FIXED_NOW))

    with pytest.raises(AuthorizationError, match="invalid authorization input: unknown_action"):
        authorizer.require(
            Principal("admin-1", Role.ADMIN),
            cast(Capability, "delete_all"),
            Resource.DELETION,
        )
    assert len(sink.events()) == 1
    assert sink.events()[0].reason is AuditReason.UNKNOWN_ACTION


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("action", "delete_all", "fixed action code"),
        ("resource", "custom_resource", "fixed resource code"),
        ("purpose", "custom_purpose", "fixed purpose code"),
        ("reason", "custom_reason", "fixed reason code"),
    ],
)
def test_audit_sink_rejects_non_fixed_codes(field: str, bad_value: str, message: str) -> None:
    sink = InMemoryAuditSink()
    base = AuditEvent(
        event_type=AuditEventType.AUTHORIZATION,
        timestamp="2026-01-02T03:04:05Z",
        actor_id="admin-1",
        role="admin",
        action=AuditAction.LINEAGE_READ,
        decision=AuditDecision.DENY,
        resource=Resource.LINEAGE,
        purpose=AuditPurpose.QUERY,
        reason=AuditReason.CAPABILITY_MISSING,
    )

    with pytest.raises(ValidationError, match=message):
        sink.record(replace(base, **{field: bad_value}))  # type: ignore[arg-type]
    assert sink.events() == ()


def test_audit_codes_cover_capabilities_and_deletion_reasons() -> None:
    assert {capability.value for capability in Capability} <= {
        action.value for action in AuditAction
    }
    assert {reason.value for reason in DeletionReason} <= {reason.value for reason in AuditReason}


def test_authorizer_rejects_raw_call_ids_in_audit_metadata() -> None:
    sink = InMemoryAuditSink()
    authorizer = Authorizer(sink, FixedClock(FIXED_NOW))

    with pytest.raises(ValidationError, match="pseudonym"):
        authorizer.check(
            Principal("admin-1", Role.ADMIN),
            Capability.LINEAGE_READ,
            Resource.LINEAGE,
            call_ref="call-0001",
        )
    assert sink.events() == ()
    assert assert_pseudonymous_call_ref("call_sha256:abc") == "call_sha256:abc"


def test_pseudonymisation_is_stable_and_one_way() -> None:
    first = pseudonymize_call_id("call-0001")
    assert first == pseudonymize_call_id("call-0001")
    assert first != pseudonymize_call_id("call-0002")
    assert "call-0001" not in first
    assert first != pseudonymize_call_id("call-0001", domain="other-domain")


def test_audit_chain_detects_tampering() -> None:
    sink = InMemoryAuditSink()
    authorizer = Authorizer(sink, FixedClock(FIXED_NOW))
    authorizer.check(
        Principal("admin-1", Role.ADMIN),
        Capability.LINEAGE_READ,
        Resource.LINEAGE,
    )
    original = sink.events()[0]
    sink._events = (  # noqa: SLF001 - deliberate tamper
        replace(original, reason=cast(AuditReason, "tampered")),
    )
    errors = sink.verify_chain()

    assert any("hash mismatch" in error for error in errors)
