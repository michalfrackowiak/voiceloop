"""P3-06 alias sync loop: poll, verify, swap atomically, auto-revert on failure.

The agent reads the champion alias but never writes it: promotion and rollback
remain gate-owned one-step alias flips (`context.md` section 2 invariant 10).
Every failure is reported as a :class:`SyncResult` and a structured log event;
``tick()`` does not raise into its caller. This is a background poller and must
never be invoked from a live call path (invariant 1).

> [!IMPORTANT]
> **FAKE ONLY by default.** All injected protocols have deterministic fakes in
> this package. Real MLflow, CTranslate2/ONNX and vLLM adapters are seams, not
> imports; no network, GPU or serving stack is required to run or test it.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import structlog

from voiceloop_common import utcnow

from .artifacts import ArtifactStore, verify_artifact
from .health import HealthProbe, HealthResult, HealthStatus
from .registry import AliasState, ChampionRegistry, ChampionSnapshot
from .serving import ServingStack, ServingState, ServingTarget, SwapPlan
from .settings import ReleaseSyncSettings

TICK_EVENT = "release_sync.tick"


class SyncOutcome(StrEnum):
    """Result of one sync tick."""

    UNCHANGED = "unchanged"
    SKIPPED = "skipped"
    SWAPPED = "swapped"
    REVERTED = "reverted"
    FAILED = "failed"
    ROLLBACK_FAILED = "rollback_failed"


class StructuredLogger(Protocol):
    """Small logging seam needed for structured sync events."""

    def info(self, event: str, **fields: object) -> None:
        """Emit an informational structured event."""
        ...

    def error(self, event: str, **fields: object) -> None:
        """Emit an error structured event."""
        ...

    def warning(self, event: str, **fields: object) -> None:
        """Emit a warning structured event."""
        ...


class StructlogEventLogger:
    """Adapter from ``structlog`` to :class:`StructuredLogger`."""

    def __init__(self, name: str) -> None:
        """Bind a structlog logger to the given service/component name."""
        self._logger: Any = structlog.get_logger(name)

    def info(self, event: str, **fields: object) -> None:
        """Emit an informational structured event."""
        self._logger.info(event, **fields)

    def error(self, event: str, **fields: object) -> None:
        """Emit an error structured event."""
        self._logger.error(event, **fields)

    def warning(self, event: str, **fields: object) -> None:
        """Emit a warning structured event."""
        self._logger.warning(event, **fields)


@dataclass(frozen=True)
class LogEvent:
    """One captured structured event (used by the deterministic fake logger)."""

    level: str
    event: str
    fields: dict[str, object]


class CollectingLogger:
    """Deterministic FAKE logger that stores structured events in memory."""

    def __init__(self) -> None:
        """Initialise an empty event list."""
        self.events: list[LogEvent] = []

    def info(self, event: str, **fields: object) -> None:
        """Capture an informational event."""
        self.events.append(LogEvent("info", event, dict(fields)))

    def error(self, event: str, **fields: object) -> None:
        """Capture an error event."""
        self.events.append(LogEvent("error", event, dict(fields)))

    def warning(self, event: str, **fields: object) -> None:
        """Capture a warning event."""
        self.events.append(LogEvent("warning", event, dict(fields)))

    def events_named(self, event: str) -> list[LogEvent]:
        """Return captured events with the given name, in order."""
        return [entry for entry in self.events if entry.event == event]


@dataclass(frozen=True)
class SyncResult:
    """Report of one tick, safe to hand to metrics or an operator dashboard."""

    outcome: SyncOutcome
    task_id: str
    alias: str
    alias_state: AliasState
    alias_version: str | None
    artifact_hash: str | None
    previous_version: Mapping[ServingTarget, str | None]
    current_version: Mapping[ServingTarget, str | None]
    changed: bool
    reason: str | None = None
    error_type: str | None = None
    rollback_errors: tuple[str, ...] = ()

    def to_event(self) -> dict[str, object]:
        """Return structured log fields including versions, hash and outcome."""
        return {
            "task_id": self.task_id,
            "outcome": self.outcome.value,
            "alias": self.alias,
            "alias_state": self.alias_state.value,
            "alias_version": self.alias_version,
            "artifact_hash": self.artifact_hash,
            "previous_version": _version_map(self.previous_version),
            "current_version": _version_map(self.current_version),
            "changed": self.changed,
            "reason": self.reason,
            "error_type": self.error_type,
            "rollback_errors": list(self.rollback_errors),
        }


class AliasSyncAgent:
    """Mirror the read-only champion alias into a transactional serving stack."""

    def __init__(
        self,
        *,
        settings: ReleaseSyncSettings,
        registry: ChampionRegistry,
        artifacts: ArtifactStore,
        serving: ServingStack,
        probe: HealthProbe,
        logger: StructuredLogger | None = None,
    ) -> None:
        """Inject every external seam; defaults are deterministic fakes only."""
        self._settings = settings
        self._registry = registry
        self._artifacts = artifacts
        self._serving = serving
        self._probe = probe
        self._logger: StructuredLogger = logger or StructlogEventLogger("release_sync")
        self._last_identity: tuple[str, str | None, AliasState] | None = None
        self._last_result: SyncResult | None = None

    @property
    def last_result(self) -> SyncResult | None:
        """Return the most recent tick result, if any."""
        return self._last_result

    @property
    def observed_identity(self) -> tuple[str, str | None, AliasState] | None:
        """Return the last processed alias/version/state identity."""
        return self._last_identity

    def reset(self) -> None:
        """Forget the observed identity so an operator can retry the same alias."""
        self._last_identity = None
        self._last_result = None

    def tick(self) -> SyncResult:
        """Poll once and, only on a real change, swap and health-check serving.

        Returns:
            A structured result. No expected failure is raised to the caller.

        """
        try:
            result = self._sync()
        except Exception as exc:
            states = self._safe_states()
            result = self._result(
                SyncOutcome.FAILED,
                alias=self._settings.alias,
                state=AliasState.UNKNOWN,
                version=None,
                artifact_hash=None,
                before=states,
                current=states,
                changed=False,
                reason="unexpected sync error",
                error_type=type(exc).__name__,
            )
        self._last_result = result
        with suppress(Exception):  # logging must never break the call path
            self._log(result)
        return result

    def _sync(self) -> SyncResult:
        """Run one poll and transactional swap attempt."""
        try:
            snapshot = self._registry.poll()
        except Exception as exc:
            states = self._safe_states()
            return self._result(
                SyncOutcome.FAILED,
                alias=self._settings.alias,
                state=AliasState.UNKNOWN,
                version=None,
                artifact_hash=None,
                before=states,
                current=states,
                changed=False,
                reason="registry poll failed",
                error_type=type(exc).__name__,
            )

        before = self._safe_states()
        identity = snapshot.identity()
        if identity == self._last_identity:
            return self._result(
                SyncOutcome.UNCHANGED,
                alias=snapshot.alias,
                state=snapshot.state,
                version=snapshot.version,
                artifact_hash=None,
                before=before,
                current=before,
                changed=False,
            )
        # Remember the identity before any work: a failed release must not be
        # retried on every poll while the alias still points at it.
        self._last_identity = identity

        if snapshot.state is not AliasState.BOUND or snapshot.version is None:
            return self._result(
                SyncOutcome.SKIPPED,
                alias=snapshot.alias,
                state=snapshot.state,
                version=snapshot.version,
                artifact_hash=None,
                before=before,
                current=before,
                changed=True,
                reason="alias is unbound; serving left unchanged",
            )
        if snapshot.artifact is None:
            return self._result(
                SyncOutcome.FAILED,
                alias=snapshot.alias,
                state=snapshot.state,
                version=snapshot.version,
                artifact_hash=None,
                before=before,
                current=before,
                changed=True,
                reason="bound champion has no artifact metadata",
            )

        try:
            return self._swap(
                snapshot, snapshot.version, snapshot.artifact.uri, snapshot.artifact.sha256, before
            )
        except Exception as exc:
            states = self._safe_states()
            return self._result(
                SyncOutcome.FAILED,
                alias=snapshot.alias,
                state=snapshot.state,
                version=snapshot.version,
                artifact_hash=None,
                before=before,
                current=states,
                changed=True,
                reason="artifact staging failed",
                error_type=type(exc).__name__,
            )

    def _swap(
        self,
        snapshot: ChampionSnapshot,
        version: str,
        artifact_uri: str,
        expected_hash: str,
        before: tuple[ServingState, ...],
    ) -> SyncResult:
        """Download, verify, prepare, commit and health-check one release."""
        with tempfile.TemporaryDirectory(
            prefix="release-sync-", dir=self._settings.temp_root
        ) as tmp:
            try:
                downloaded = self._artifacts.download(artifact_uri, Path(tmp))
            except Exception as exc:
                return self._result(
                    SyncOutcome.FAILED,
                    alias=snapshot.alias,
                    state=snapshot.state,
                    version=version,
                    artifact_hash=None,
                    before=before,
                    current=self._safe_states(),
                    changed=True,
                    reason="artifact download failed",
                    error_type=type(exc).__name__,
                )
            try:
                artifact_hash = verify_artifact(downloaded, expected_hash)
            except Exception as exc:
                return self._result(
                    SyncOutcome.FAILED,
                    alias=snapshot.alias,
                    state=snapshot.state,
                    version=version,
                    artifact_hash=None,
                    before=before,
                    current=self._safe_states(),
                    changed=True,
                    reason="artifact hash verification failed",
                    error_type=type(exc).__name__,
                )

            plan = SwapPlan(version=version, artifact_hash=artifact_hash, before=before)
            try:
                prepared = self._serving.prepare(plan, downloaded)
            except Exception as exc:
                return self._result(
                    SyncOutcome.FAILED,
                    alias=snapshot.alias,
                    state=snapshot.state,
                    version=version,
                    artifact_hash=artifact_hash,
                    before=before,
                    current=self._safe_states(),
                    changed=True,
                    reason="serving prepare failed",
                    error_type=type(exc).__name__,
                )
            try:
                committed = self._serving.commit(prepared)
            except Exception as exc:
                return self._result(
                    SyncOutcome.FAILED,
                    alias=snapshot.alias,
                    state=snapshot.state,
                    version=version,
                    artifact_hash=artifact_hash,
                    before=before,
                    current=self._safe_states(),
                    changed=True,
                    reason="serving commit failed",
                    error_type=type(exc).__name__,
                )

            try:
                health: HealthResult = self._probe.check(expected_version=version)
            except Exception as exc:
                errors = self._serving.rollback(committed)
                return self._result(
                    SyncOutcome.ROLLBACK_FAILED if errors else SyncOutcome.FAILED,
                    alias=snapshot.alias,
                    state=snapshot.state,
                    version=version,
                    artifact_hash=artifact_hash,
                    before=before,
                    current=self._safe_states(),
                    changed=True,
                    reason="health probe error",
                    error_type=type(exc).__name__,
                    rollback_errors=errors,
                )

            if health.status is HealthStatus.HEALTHY:
                return self._result(
                    SyncOutcome.SWAPPED,
                    alias=snapshot.alias,
                    state=snapshot.state,
                    version=version,
                    artifact_hash=artifact_hash,
                    before=before,
                    current=self._safe_states(),
                    changed=True,
                )

            errors = self._serving.rollback(committed)
            reason = "health check failed"
            if health.failed_checks:
                reason += ": " + ",".join(health.failed_checks)
            return self._result(
                SyncOutcome.ROLLBACK_FAILED if errors else SyncOutcome.REVERTED,
                alias=snapshot.alias,
                state=snapshot.state,
                version=version,
                artifact_hash=artifact_hash,
                before=before,
                current=self._safe_states(),
                changed=True,
                reason=reason,
                rollback_errors=errors,
            )

    def _result(
        self,
        outcome: SyncOutcome,
        *,
        alias: str,
        state: AliasState,
        version: str | None,
        artifact_hash: str | None,
        before: tuple[ServingState, ...],
        current: tuple[ServingState, ...],
        changed: bool,
        reason: str | None = None,
        error_type: str | None = None,
        rollback_errors: tuple[str, ...] = (),
    ) -> SyncResult:
        """Build a report from a tick branch without raising."""
        return SyncResult(
            outcome=outcome,
            task_id=self._settings.task_id,
            alias=alias,
            alias_state=state,
            alias_version=version,
            artifact_hash=artifact_hash,
            previous_version=_states_to_versions(before),
            current_version=_states_to_versions(current),
            changed=changed,
            reason=reason,
            error_type=error_type,
            rollback_errors=rollback_errors,
        )

    def _safe_states(self) -> tuple[ServingState, ...]:
        """Read serving state without allowing a probe failure to escape."""
        try:
            return self._serving.states()
        except Exception:
            return ()

    def _log(self, result: SyncResult) -> None:
        """Emit exactly one structured event for the tick."""
        fields = result.to_event()
        fields["observed_at"] = utcnow().isoformat(timespec="seconds").replace("+00:00", "Z")
        level = self._level_for(result.outcome)
        getattr(self._logger, level)(TICK_EVENT, **fields)

    @staticmethod
    def _level_for(outcome: SyncOutcome) -> str:
        """Map an outcome to a stable log level name."""
        if outcome in {SyncOutcome.FAILED, SyncOutcome.ROLLBACK_FAILED}:
            return "error"
        if outcome in {SyncOutcome.REVERTED, SyncOutcome.SKIPPED}:
            return "warning"
        return "info"


def _states_to_versions(states: tuple[ServingState, ...]) -> dict[ServingTarget, str | None]:
    """Project serving states into a target-to-version mapping."""
    return {state.target: state.current_version for state in states}


def _version_map(versions: Mapping[ServingTarget, str | None]) -> dict[str, str | None]:
    """Convert target keys to stable strings for structured logging."""
    return {target.value: version for target, version in versions.items()}
