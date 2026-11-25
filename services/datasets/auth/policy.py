"""Default-deny authorisation with an audit event for every decision.

This package is deliberately free of a real identity provider: a production
adapter must verify credentials out of band and construct a
``lineage.roles.Principal`` with a verified ``Role`` enum member. There is no
role-assignment endpoint here, so a reviewer or ML engineer cannot escalate.
"""

from __future__ import annotations

from dataclasses import dataclass

from voiceloop_common import ValidationError, VoiceLoopError

from ..lineage.roles import Capability, Principal, capabilities_for
from ..lineage.store import (
    AuditAction,
    AuditDecision,
    AuditEvent,
    AuditEventType,
    AuditPurpose,
    AuditReason,
    AuditSink,
    Clock,
    Resource,
)


class AuthorizationError(VoiceLoopError):
    """Raised when a principal lacks a required capability."""


@dataclass(frozen=True, slots=True)
class AuthorizationDecision:
    """Result of one default-deny capability check."""

    actor_id: str
    role: str
    capability: Capability
    resource: Resource
    allowed: bool
    reason: AuditReason


class Authorizer:
    """Central policy decision point; every check is appended to the audit sink."""

    def __init__(self, audit_sink: AuditSink, clock: Clock) -> None:
        """Wire the authorizer to the injected audit sink and clock.

        Args:
            audit_sink: Append-only sink receiving allow and deny decisions.
            clock: Injectable UTC clock for deterministic audit timestamps.

        """
        self._audit_sink = audit_sink
        self._clock = clock

    def _record_decision(
        self,
        *,
        principal: Principal,
        action: AuditAction,
        decision: AuditDecision,
        resource: Resource,
        purpose: AuditPurpose,
        reason: AuditReason,
        call_ref: str,
        model_ids: tuple[str, ...],
        dataset_ids: tuple[str, ...],
    ) -> AuditEvent:
        """Append one PII-free audit event using only fixed enum codes."""
        return self._audit_sink.record(
            AuditEvent(
                event_type=AuditEventType.AUTHORIZATION,
                timestamp=self._clock.now().isoformat(timespec="seconds").replace("+00:00", "Z"),
                actor_id=principal.actor_id,
                role=principal.role.value,
                action=action,
                decision=decision,
                resource=resource,
                purpose=purpose,
                reason=reason,
                call_ref=call_ref,
                model_ids=model_ids,
                dataset_ids=dataset_ids,
            )
        )

    def check(
        self,
        principal: Principal,
        capability: Capability,
        resource: Resource,
        *,
        purpose: AuditPurpose = AuditPurpose.OPERATION,
        call_ref: str = "",
        model_ids: tuple[str, ...] = (),
        dataset_ids: tuple[str, ...] = (),
    ) -> AuthorizationDecision:
        """Evaluate a capability and audit the decision either way.

        Invalid action, resource or purpose values never reach the audit sink
        verbatim: a DENY event with ``unknown_action``, ``unknown_resource`` or
        ``unknown_purpose`` codes is recorded first, then an
        :class:`AuthorizationError` is raised.

        Args:
            principal: Authenticated principal with a verified role.
            capability: Operation being requested; must be a ``Capability``.
            resource: Fixed resource code from :class:`Resource`.
            purpose: Fixed purpose code from :class:`AuditPurpose`.
            call_ref: Pseudonymous call reference, when the decision concerns
                one call. Raw call IDs are never accepted here.
            model_ids: Exact model IDs affected by the decision, if any.
            dataset_ids: Exact dataset IDs affected by the decision, if any.

        Returns:
            The decision, including a fixed reason code.

        Raises:
            AuthorizationError: If a capability is denied or an operation input
                is not one of the approved enum values.
            ValidationError: If a raw call ID is supplied as ``call_ref``.

        """
        if call_ref:
            assert_pseudonymous_call_ref(call_ref)

        invalid_action = not isinstance(capability, Capability)
        invalid_resource = not isinstance(resource, Resource)
        invalid_purpose = not isinstance(purpose, AuditPurpose)
        if invalid_action or invalid_resource or invalid_purpose:
            action = AuditAction.UNKNOWN_ACTION if invalid_action else AuditAction(capability.value)
            resource_code = Resource.UNKNOWN_RESOURCE if invalid_resource else resource
            purpose_code = AuditPurpose.UNKNOWN_PURPOSE if invalid_purpose else purpose
            if invalid_action:
                reason = AuditReason.UNKNOWN_ACTION
            elif invalid_resource:
                reason = AuditReason.UNKNOWN_RESOURCE
            else:
                reason = AuditReason.UNKNOWN_PURPOSE
            self._record_decision(
                principal=principal,
                action=action,
                decision=AuditDecision.DENY,
                resource=resource_code,
                purpose=purpose_code,
                reason=reason,
                call_ref=call_ref,
                model_ids=model_ids,
                dataset_ids=dataset_ids,
            )
            raise AuthorizationError(f"invalid authorization input: {reason.value}")

        action = AuditAction(capability.value)
        allowed = capability in capabilities_for(principal.role)
        reason = AuditReason.CAPABILITY_GRANTED if allowed else AuditReason.CAPABILITY_MISSING
        decision = AuthorizationDecision(
            actor_id=principal.actor_id,
            role=principal.role.value,
            capability=capability,
            resource=resource,
            allowed=allowed,
            reason=reason,
        )
        self._record_decision(
            principal=principal,
            action=action,
            decision=AuditDecision.ALLOW if allowed else AuditDecision.DENY,
            resource=resource,
            purpose=purpose,
            reason=reason,
            call_ref=call_ref,
            model_ids=model_ids,
            dataset_ids=dataset_ids,
        )
        return decision

    def require(
        self,
        principal: Principal,
        capability: Capability,
        resource: Resource,
        *,
        purpose: AuditPurpose = AuditPurpose.OPERATION,
        call_ref: str = "",
        model_ids: tuple[str, ...] = (),
        dataset_ids: tuple[str, ...] = (),
    ) -> AuthorizationDecision:
        """Audit a check and raise when the capability is not granted.

        Args and return value match :meth:`check`; denial raises instead of
        returning a decision.

        Raises:
            AuthorizationError: If the capability is denied.

        """
        decision = self.check(
            principal,
            capability,
            resource,
            purpose=purpose,
            call_ref=call_ref,
            model_ids=model_ids,
            dataset_ids=dataset_ids,
        )
        if not decision.allowed:
            raise AuthorizationError(
                f"role {principal.role.value} lacks capability {capability.value}"
            )
        return decision


def assert_pseudonymous_call_ref(call_ref: str) -> str:
    """Reject a raw call ID accidentally passed where a pseudonym is required.

    Args:
        call_ref: Value to check.

    Returns:
        The same string when it is a ``call_sha256:`` reference.

    Raises:
        ValidationError: If the value looks like a raw call ID.

    """
    if not isinstance(call_ref, str) or not call_ref.startswith("call_sha256:"):
        raise ValidationError("audit call_ref must be a pseudonym, not a raw call_id")
    return call_ref
