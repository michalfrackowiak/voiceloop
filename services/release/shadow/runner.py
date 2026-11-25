"""Background-only candidate/champion execution with digest-only results.

Nothing in this module is reachable from :meth:`services.release.shadow.mirror.ShadowMirror.mirror`.
``ShadowRunner.drain_once`` is the explicit background entry point: it dequeues
at most ``max_turns_per_drain`` mirrored turns, runs both model seams and
records allowlisted audit events. Runner failures are converted into failed
results and never raised into the caller.

> [!IMPORTANT]
> **FAKE ONLY by default.** ``ScriptedModelRunner`` is a deterministic in-memory
> stand-in that hashes scripted booking fields instead of decoding audio. Real
> converted-artifact adapters are documented seams; no model, GPU or network
> library is imported here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol

from .mirror import AuditEvent, AuditSink, Clock, MirroredTurn, MirrorSink
from .settings import ShadowSettings

RESULT_EVENT = "shadow.result"
RUN_EVENT = "shadow.run"


class SteppableClock(Clock, Protocol):
    """Test seam clock that can advance simulated time without sleeping."""

    def advance(self, seconds: float) -> None:
        """Move simulated time forward by the given number of seconds."""
        ...


class ModelSource(StrEnum):
    """Which side of the shadow comparison produced a result."""

    CANDIDATE = "candidate"
    CHAMPION = "champion"


class ResultStatus(StrEnum):
    """Outcome of one background model call."""

    OK = "ok"
    FAILED = "failed"


def field_digests(
    fields: Mapping[str, Any], ignored: frozenset[str] = frozenset()
) -> dict[str, str]:
    """Hash booking-field values so raw guest data never leaves the runner.

    Args:
        fields: Project-specific booking fields (may contain PII in memory).
        ignored: Top-level keys excluded from the comparison.

    Returns:
        A mapping of field name to SHA-256 of its canonical JSON value. Only
        these digests are stored in results, audit events and reports.

    """
    return {
        name: hashlib.sha256(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        for name, value in fields.items()
        if name not in ignored
    }


@dataclass(frozen=True)
class ModelResult:
    """PII-safe record of one shadow model execution."""

    source: ModelSource
    call_id: str
    turn_index: int
    artifact_uri: str
    model_id: str
    started_at: datetime
    completed_at: datetime
    synthetic: bool
    status: ResultStatus
    field_digests: Mapping[str, str] = field(default_factory=dict)
    error_type: str | None = None

    def to_audit_event(self, task_id: str) -> AuditEvent:
        """Return an allowlisted audit event for this result."""
        return AuditEvent(
            event=RESULT_EVENT,
            task_id=task_id,
            status=self.status.value,
            observed_at=self.completed_at.isoformat().replace("+00:00", "Z"),
            call_id=self.call_id,
            turn_index=self.turn_index,
            source=self.source.value,
            artifact_uri=self.artifact_uri,
            model_id=self.model_id,
            synthetic=self.synthetic,
            error_type=self.error_type,
            counts={"field_count": len(self.field_digests)},
        )


class CandidateRunner(Protocol):
    """Candidate artifact seam (converted/quantised artifact in production)."""

    @property
    def source(self) -> ModelSource:
        """Return ``ModelSource.CANDIDATE`` for every result."""
        ...

    @property
    def artifact_uri(self) -> str:
        """Return the artifact URI stamped on every result."""
        ...

    @property
    def model_id(self) -> str:
        """Return the non-secret model identifier stamped on every result."""
        ...

    def run(self, turn: MirroredTurn) -> ModelResult:
        """Run the candidate on the referenced audio and return a safe result."""
        ...


class ChampionRunner(Protocol):
    """Champion artifact seam (same shape as the candidate seam)."""

    @property
    def source(self) -> ModelSource:
        """Return ``ModelSource.CHAMPION`` for every result."""
        ...

    @property
    def artifact_uri(self) -> str:
        """Return the artifact URI stamped on every result."""
        ...

    @property
    def model_id(self) -> str:
        """Return the non-secret model identifier stamped on every result."""
        ...

    def run(self, turn: MirroredTurn) -> ModelResult:
        """Run the champion on the referenced audio and return a safe result."""
        ...


class ShadowRunner:
    """Drain mirrored turns in bounded batches; isolate every runner failure."""

    def __init__(
        self,
        settings: ShadowSettings,
        sink: MirrorSink,
        candidate: CandidateRunner,
        champion: ChampionRunner,
        clock: Clock,
        audit: AuditSink,
    ) -> None:
        """Inject settings, queue, both model seams, clock and audit sink."""
        self._settings = settings
        self._sink = sink
        self._candidate = candidate
        self._champion = champion
        self._clock = clock
        self._audit = audit
        self._results: list[ModelResult] = []

    @property
    def results(self) -> tuple[ModelResult, ...]:
        """Return all results recorded so far, in execution order."""
        return tuple(self._results)

    def drain_once(self) -> int:
        """Process at most one configured batch of queued turns.

        Returns:
            Number of mirrored turns processed (not number of model calls).
            Runner exceptions are converted into failed results and never
            escape to the caller.

        """
        processed = 0
        while processed < self._settings.max_turns_per_drain and self._sink.pending() > 0:
            turn = self._sink.dequeue()
            if turn is None:
                break
            self._run_one(self._candidate, turn)
            self._run_one(self._champion, turn)
            processed += 1

        failures = sum(1 for result in self._results if result.status is ResultStatus.FAILED)
        event = AuditEvent(
            event=RUN_EVENT,
            task_id=self._settings.task_id,
            status="ok",
            observed_at=self._clock.now().isoformat().replace("+00:00", "Z"),
            counts={
                "processed": processed,
                "results": len(self._results),
                "failures": failures,
            },
        )
        self._record_safely(event)
        return processed

    def _run_one(
        self,
        runner: CandidateRunner | ChampionRunner,
        turn: MirroredTurn,
    ) -> None:
        """Run one model seam and record success or a typed failure."""
        started = self._clock.now()
        try:
            result = runner.run(turn)
        except Exception as exc:
            result = ModelResult(
                source=runner.source,
                call_id=turn.call_id,
                turn_index=turn.turn_index,
                artifact_uri=runner.artifact_uri,
                model_id=runner.model_id,
                started_at=started,
                completed_at=self._clock.now(),
                synthetic=turn.synthetic,
                status=ResultStatus.FAILED,
                error_type=type(exc).__name__,
            )
        self._results.append(result)
        self._record_safely(result.to_audit_event(self._settings.task_id))

    def _record_safely(self, event: AuditEvent) -> None:
        """Emit an audit event without letting logging break background work."""
        try:
            self._audit.record(event)
        except Exception:
            return


class ScriptedModelRunner:
    """Deterministic FAKE model seam that hashes scripted booking fields.

    ``clock_steps`` advances the injected clock to simulate a slow adapter
    without sleeping. ``fail`` injects a runner error; the exception body is a
    PII-shaped trap that must never appear in audit events or artifacts.
    """

    def __init__(
        self,
        *,
        source: ModelSource,
        artifact_uri: str,
        model_id: str,
        clock: SteppableClock,
        slots_by_turn: Mapping[tuple[str, int], Mapping[str, Any]] | None = None,
        clock_steps: float = 0.0,
        fail: bool = False,
        error_message: str = "fake runner failure",
    ) -> None:
        """Configure the fake's identity, scripted slots and failure behaviour."""
        self._source = source
        self._artifact_uri = artifact_uri
        self._model_id = model_id
        self._clock = clock
        self._slots_by_turn = dict(slots_by_turn or {})
        self._clock_steps = clock_steps
        self._fail = fail
        self._error_message = error_message
        self.calls: list[MirroredTurn] = []

    @property
    def source(self) -> ModelSource:
        """Return the side this fake stands in for."""
        return self._source

    @property
    def artifact_uri(self) -> str:
        """Return the fake artifact URI."""
        return self._artifact_uri

    @property
    def model_id(self) -> str:
        """Return the fake model identifier."""
        return self._model_id

    def set_slots(self, call_id: str, turn_index: int, fields: Mapping[str, Any]) -> None:
        """Script booking fields for one turn without exposing raw data later."""
        self._slots_by_turn[(call_id, turn_index)] = dict(fields)

    def run(self, turn: MirroredTurn) -> ModelResult:
        """Return digest-only results from the scripted slot object."""
        self.calls.append(turn)
        started = self._clock.now()
        if self._clock_steps > 0:
            self._clock.advance(self._clock_steps)
        if self._fail:
            raise RuntimeError(self._error_message)
        slots = self._slots_by_turn.get((turn.call_id, turn.turn_index), {})
        return ModelResult(
            source=self._source,
            call_id=turn.call_id,
            turn_index=turn.turn_index,
            artifact_uri=self._artifact_uri,
            model_id=self._model_id,
            started_at=started,
            completed_at=self._clock.now(),
            synthetic=turn.synthetic,
            status=ResultStatus.OK,
            field_digests=field_digests(slots),
        )
