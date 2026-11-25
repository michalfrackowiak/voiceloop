"""Service-local access-control package for the dataset service (P2-12.3).

Authority is default-deny and externalised: construct a verified
``services.datasets.lineage.Principal`` from your identity provider, then use
:class:`Authorizer` so every allow and deny decision is audited.
"""

from __future__ import annotations

from .policy import (
    AuthorizationDecision,
    AuthorizationError,
    Authorizer,
    assert_pseudonymous_call_ref,
)

__all__ = [
    "AuthorizationDecision",
    "AuthorizationError",
    "Authorizer",
    "assert_pseudonymous_call_ref",
]
