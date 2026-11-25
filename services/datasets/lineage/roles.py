"""Role and capability definitions for VoiceLoop governance (P2-12.3).

The three roles match ``context.md`` section 12. Authorisation decisions are
taken in ``services.datasets.auth.policy``; this module owns the role matrix,
which is exposed only as a read-only ``MappingProxyType`` over copied
``frozenset`` values. There is deliberately no API here to change or escalate a
role: a real identity-provider adapter must map a verified principal to one of
these enum values before calling any workflow.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from voiceloop_common import ValidationError


class Role(StrEnum):
    """Verified, non-escalatable governance roles."""

    REVIEWER = "reviewer"
    ML_ENGINEER = "ml_engineer"
    ADMIN = "admin"


class Capability(StrEnum):
    """Operations that an authorisation decision can grant or deny."""

    REVIEW_READ = "review_read"
    REVIEW_WRITE = "review_write"
    DATASET_BUILD = "dataset_build"
    TRAIN = "train"
    LINEAGE_READ = "lineage_read"
    DELETION_AUTHORIZE = "deletion_authorize"
    CONSENT_CHANGE = "consent_change"
    AUDIT_READ = "audit_read"


# Read-only view of the role matrix. The dict and frozensets are constructed
# inside MappingProxyType, so no mutable reference escapes this module.
# ``services.datasets.lineage`` does not re-export this name: callers should use
# :func:`capabilities_for` instead.
ROLE_CAPABILITIES: Mapping[Role, frozenset[Capability]] = MappingProxyType(
    {
        # Review is read/write only: reviewers do not build, train or authorise.
        Role.REVIEWER: frozenset({Capability.REVIEW_READ, Capability.REVIEW_WRITE}),
        # Dataset/training/lineage access only: no deletion, consent or review.
        Role.ML_ENGINEER: frozenset(
            {
                Capability.DATASET_BUILD,
                Capability.TRAIN,
                Capability.LINEAGE_READ,
            }
        ),
        # Governance admin authorises deletion and consent changes and can
        # inspect lineage/audit records; it cannot review-write or train.
        Role.ADMIN: frozenset(
            {
                Capability.LINEAGE_READ,
                Capability.DELETION_AUTHORIZE,
                Capability.CONSENT_CHANGE,
                Capability.AUDIT_READ,
            }
        ),
    }
)

__all__ = ["Capability", "Principal", "Role", "capabilities_for"]


def capabilities_for(role: Role) -> frozenset[Capability]:
    """Return the immutable capability set for a verified role.

    Args:
        role: A :class:`Role` enum member. Strings are rejected so a caller
            cannot smuggle an arbitrary role name past the default-deny check.

    Returns:
        The capabilities granted to that role; an empty set for unknown values.

    Raises:
        ValidationError: If ``role`` is not a :class:`Role` member.

    """
    if not isinstance(role, Role):
        raise ValidationError("role must be a verified Role enum member")
    return ROLE_CAPABILITIES.get(role, frozenset())


@dataclass(frozen=True, slots=True)
class Principal:
    """An authenticated actor with exactly one verified governance role.

    ``actor_id`` must be a pseudonymous service or staff identifier: no email
    address, whitespace or free-form text is accepted, so it cannot carry PII
    into audit events.
    """

    actor_id: str
    role: Role

    def __post_init__(self) -> None:
        """Reject malformed identities and non-enum roles."""
        if not isinstance(self.role, Role):
            raise ValidationError("principal role must be a verified Role enum member")
        if (
            not isinstance(self.actor_id, str)
            or not self.actor_id
            or len(self.actor_id) > 128
            or any(char.isspace() for char in self.actor_id)
            or "@" in self.actor_id
        ):
            raise ValidationError("principal actor_id must be a short pseudonymous identifier")

    @property
    def is_admin(self) -> bool:
        """Return whether this principal has the admin role."""
        return self.role is Role.ADMIN
