"""MLflow alias flip primitives with audit and one-step rollback.

Promotion binds ``@champion`` to the version currently held by ``@challenger``;
rollback binds it back to the recorded previous version. Both operations are a
single alias write (``context.md`` section 2 invariant 10) followed by an audit
record. An injectable :class:`AliasStore` keeps MLflow and HTTP out of the unit
test path; P3-06 supplies the real alias sync backend.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from voiceloop_common import ValidationError

from .audit import AUDIT_TASK_ID, AuditLog, PromotionRecord
from .rules import GateError

CHAMPION_ALIAS = "@champion"
CHALLENGER_ALIAS = "@challenger"


class AliasStore(Protocol):
    """Minimal alias-store interface; one ``set`` call is one alias flip."""

    def get(self, alias: str) -> str | None:
        """Return the version bound to ``alias``, or None when unbound."""
        ...

    def set(self, alias: str, version: str) -> None:
        """Bind ``alias`` to ``version`` in one atomic flip."""
        ...


class InMemoryAliasStore:
    """Fake alias store for tests and local runs; no MLflow dependency."""

    def __init__(self, aliases: Mapping[str, str] | None = None) -> None:
        """Initialise from an optional alias-to-version mapping."""
        self._aliases: dict[str, str] = dict(aliases or {})

    def get(self, alias: str) -> str | None:
        """Return the version bound to ``alias``, or None when unbound."""
        return self._aliases.get(alias)

    def set(self, alias: str, version: str) -> None:
        """Bind ``alias`` to ``version`` after validating both strings."""
        _require_text(alias, "alias")
        _require_text(version, "version")
        self._aliases[alias] = version

    def snapshot(self) -> dict[str, str]:
        """Return a copy of the current alias bindings."""
        return dict(self._aliases)


def promote(
    store: AliasStore,
    challenger_version: str,
    *,
    actor_id: str,
    reason: str,
    audit_log: AuditLog,
    task_id: str = AUDIT_TASK_ID,
) -> PromotionRecord:
    """Flip ``@champion`` to the evaluated ``@challenger`` version and audit it.

    Args:
        store: Injected alias store.
        challenger_version: MLflow version whose converted artifact was evaluated.
        actor_id: Human or service identity authorising the promotion.
        reason: Human-readable justification.
        audit_log: Append-only audit sink.
        task_id: Roadmap task ID recorded in the audit entry.

    Returns:
        The persisted promotion record, including the previous champion version.

    Raises:
        GateError: If either alias is unbound or ``@challenger`` no longer points
            at ``challenger_version`` (a stale promotion is refused).

    """
    _require_text(challenger_version, "challenger_version")
    previous = store.get(CHAMPION_ALIAS)
    current_challenger = store.get(CHALLENGER_ALIAS)
    if not previous:
        raise GateError("no current @champion version to replace")
    if current_challenger != challenger_version:
        raise GateError(
            f"@challenger is {current_challenger!r}, not evaluated version {challenger_version!r}"
        )

    try:
        store.set(CHAMPION_ALIAS, challenger_version)
        return audit_log.record(
            actor_id=actor_id,
            action="promote",
            reason=reason,
            before_version=previous,
            after_version=challenger_version,
            task_id=task_id,
        )
    except Exception:
        # Keep the alias and audit consistent if the write fails.
        store.set(CHAMPION_ALIAS, previous)
        raise


def rollback(
    store: AliasStore,
    *,
    previous_version: str,
    actor_id: str,
    reason: str,
    audit_log: AuditLog,
    expected_current_version: str | None = None,
    task_id: str = AUDIT_TASK_ID,
) -> PromotionRecord:
    """Restore the previous champion alias in one step and audit it.

    Args:
        store: Injected alias store.
        previous_version: Champion version to restore.
        actor_id: Human or service identity authorising the rollback.
        reason: Human-readable justification.
        audit_log: Append-only audit sink.
        expected_current_version: When given, refuse to roll back unless
            ``@champion`` still points at this version.
        task_id: Roadmap task ID recorded in the audit entry.

    Returns:
        The persisted rollback record.

    Raises:
        GateError: If the alias is unbound, unchanged, or not at the expected version.

    """
    _require_text(previous_version, "previous_version")
    current = store.get(CHAMPION_ALIAS)
    if not current:
        raise GateError("no current @champion version to roll back")
    if expected_current_version is not None and current != expected_current_version:
        raise GateError(
            f"@champion is {current!r}, not expected version {expected_current_version!r}"
        )
    if current == previous_version:
        raise GateError("nothing to roll back: @champion already points at previous_version")

    try:
        store.set(CHAMPION_ALIAS, previous_version)
        return audit_log.record(
            actor_id=actor_id,
            action="rollback",
            reason=reason,
            before_version=current,
            after_version=previous_version,
            task_id=task_id,
        )
    except Exception:
        store.set(CHAMPION_ALIAS, current)
        raise


def rollback_promotion(
    store: AliasStore,
    promotion: PromotionRecord,
    *,
    actor_id: str,
    reason: str,
    audit_log: AuditLog,
    task_id: str = AUDIT_TASK_ID,
) -> PromotionRecord:
    """Undo one recorded promotion, restoring its ``before_version``.

    Raises:
        GateError: If ``promotion`` is not a promote record.

    """
    if promotion.action != "promote":
        raise GateError("rollback_promotion requires a promote record")
    return rollback(
        store,
        previous_version=promotion.before_version,
        actor_id=actor_id,
        reason=reason,
        audit_log=audit_log,
        expected_current_version=promotion.after_version,
        task_id=task_id,
    )


def _require_text(value: str, label: str) -> None:
    """Reject blank or non-string identifiers and versions."""
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label} must be a non-blank string")
