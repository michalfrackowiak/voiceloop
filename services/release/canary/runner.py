"""Background-only canary orchestration with deterministic injected fakes.

This module is never reachable from a live call path. It pulls at most a
configured number of pseudonymous call observations from an injected event
source, joins outcomes by ``call_id`` through the P3-07 outcome seam, scores
paired outcome metrics and (only when required) asks the bounded rollback
controller for a single alias flip. A fake clock supplies all timestamps; no
sleep, network, GPU, model runtime or registry client is used.

Every emitted report and audit event carries
``SYNTHETIC/FIXTURE STAND-IN - NOT LIVE CANARY EVIDENCE``. The runner makes no
promotion decision, writes no alias itself and never claims a champion.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import structlog

from services.evalgate.gate.aliases import InMemoryAliasStore
from services.evalgate.gate.audit import PromotionRecord
from services.release.shadow.mirror import Clock, FakeClock
from services.release.shadow.scoring import FakeOutcomeSource, OutcomeRecord, OutcomeSource
from services.release.sync.registry import ReadOnlyAliasStore

from .metrics import (
    CanaryMetrics,
    CanaryPair,
    PredictionRecord,
    RegressionVerdict,
    detect_regression,
    score_canary,
)
from .rollback import (
    AliasStore,
    CanaryDeployment,
    InMemoryRollbackAudit,
    RollbackAudit,
    RollbackController,
    RollbackResult,
    timestamp,
)
from .settings import MAX_CANARY_FRACTION, CanarySettings
from .split import Arm, TrafficSplit

REPO_ROOT = Path(__file__).resolve().parents[3]
DEMO_SPLIT_SALT = "voiceloop-canary-demo-0"


@dataclass(frozen=True)
class CanaryCallObservation:
    """One in-memory-only call observation from an injected event source.

    Raw booking fields are used solely to build digests and are never copied
    into results, audit events or reports.
    """

    call_id: str
    canary_fields: Mapping[str, Any]
    champion_fields: Mapping[str, Any]
    observed_at: datetime
    synthetic: bool = True


class EventSource(Protocol):
    """Background-only, bounded source of simulated call observations."""

    def next_call(self) -> CanaryCallObservation | None:
        """Return the next observation, or None when the source is exhausted."""
        ...


class FakeEventSource:
    """Deterministic in-memory FAKE event source (no live traffic)."""

    def __init__(self, observations: Sequence[CanaryCallObservation]) -> None:
        """Store an ordered tuple of synthetic observations."""
        self._observations = tuple(observations)
        self._index = 0

    def next_call(self) -> CanaryCallObservation | None:
        """Return the next stored observation or None at the end."""
        if self._index >= len(self._observations):
            return None
        observation = self._observations[self._index]
        self._index += 1
        return observation

    @property
    def emitted(self) -> int:
        """Return how many observations have been emitted."""
        return self._index


@dataclass(frozen=True)
class CanaryAuditEvent:
    """Allowlisted, PII-safe structured event emitted once per tick."""

    event: str
    task_id: str
    status: str
    observed_at: str
    evidence_label: str
    reason: str
    synthetic: bool
    alias: str | None = None
    writes: int = 0
    counts: Mapping[str, int] = field(default_factory=dict)
    error_type: str | None = None

    def to_fields(self) -> dict[str, object]:
        """Return a flat dict containing only allowlisted fields."""
        return {
            "event": self.event,
            "task_id": self.task_id,
            "status": self.status,
            "observed_at": self.observed_at,
            "evidence_label": self.evidence_label,
            "reason": self.reason,
            "synthetic": self.synthetic,
            "alias": self.alias,
            "writes": self.writes,
            "counts": dict(self.counts),
            "error_type": self.error_type,
        }


class CanaryAuditSink(Protocol):
    """Structured audit seam for background tick events."""

    def record(self, event: CanaryAuditEvent) -> None:
        """Persist or emit one allowlisted event."""
        ...


class CollectingCanaryAudit:
    """Deterministic in-memory FAKE audit sink used in tests and demos."""

    def __init__(self) -> None:
        """Initialise an empty event list."""
        self.events: list[CanaryAuditEvent] = []

    def record(self, event: CanaryAuditEvent) -> None:
        """Append one event to the in-memory list."""
        self.events.append(event)


class StructlogCanaryAudit:
    """Adapter from :class:`CanaryAuditEvent` to configured structlog."""

    def __init__(self, name: str = "release_canary") -> None:
        """Bind a structlog logger to the given component name."""
        self._logger: Any = structlog.get_logger(name)

    def record(self, event: CanaryAuditEvent) -> None:
        """Emit allowlisted fields as one structured event."""
        fields = event.to_fields()
        name = fields.pop("event")
        self._logger.info(str(name), **fields)


@dataclass(frozen=True)
class CanaryTickResult:
    """One labelled, PII-safe result of a bounded canary tick."""

    observed_at: str
    window_deadline: str
    split_fraction: float
    split_salt: str
    calls_observed: int
    calls_canary: int
    calls_control: int
    metrics: CanaryMetrics
    verdict: RegressionVerdict
    rollback: RollbackResult
    evidence_label: str
    synthetic: bool
    promotion_decision: str = "not_evaluated"
    champion_claim: bool = False

    def to_report(self) -> dict[str, Any]:
        """Return a labelled internal artifact, never an eval-report claim."""
        return {
            "artifact_type": "voiceloop.canary.rehearsal",
            "report_schema_version": "internal-1.0.0",
            "evidence_label": self.evidence_label,
            "synthetic": self.synthetic,
            "real_traffic": False,
            "observed_at": self.observed_at,
            "window_deadline": self.window_deadline,
            "traffic_split": {
                "unit": "call",
                "canary_fraction": self.split_fraction,
                "split_salt": self.split_salt,
                "calls_observed": self.calls_observed,
                "calls_canary": self.calls_canary,
                "calls_control": self.calls_control,
            },
            "metrics": self.metrics.to_dict(),
            "verdict": self.verdict.to_dict(),
            "rollback": self.rollback.to_dict(),
            "promotion_decision": self.promotion_decision,
            "champion_claim": self.champion_claim,
            "alias_writes": self.rollback.writes,
            "gate_edits": 0,
            "thresholds_changed": False,
        }


class CanaryRunner:
    """Drive bounded, deterministic canary ticks without any live I/O."""

    def __init__(
        self,
        settings: CanarySettings,
        promotion: PromotionRecord,
        event_source: EventSource,
        outcome_source: OutcomeSource,
        alias_store: AliasStore,
        audit: CanaryAuditSink,
        clock: Clock,
        rollback_audit: RollbackAudit,
        registry: ReadOnlyAliasStore | None = None,
    ) -> None:
        """Inject typed settings, all fakes/seams and the read-only registry."""
        self._settings = settings
        self._promotion = promotion
        self._events = event_source
        self._outcomes = outcome_source
        self._audit = audit
        self._clock = clock
        self._split = TrafficSplit(settings)
        self._rollback = RollbackController(
            settings,
            alias_store,
            rollback_audit,
            registry if registry is not None else alias_store,
        )
        self._pairs: list[CanaryPair] = []
        self._deployment: CanaryDeployment | None = None

    @property
    def rollback(self) -> RollbackController:
        """Return the bounded rollback controller (for inspection/tests)."""
        return self._rollback

    def tick(self) -> CanaryTickResult:
        """Process one bounded batch and decide whether a rollback is required.

        Returns:
            One labelled result with paired metrics, a fail-closed verdict and
            the rollback outcome. Expected source failures never raise here.

        """
        observed = self._clock.now()
        if self._deployment is None:
            self._deployment = CanaryDeployment.from_promotion(
                self._promotion, observed, self._settings.evidence_label
            )
        deadline = self._deployment.deadline(self._settings.window_seconds)

        if self._rollback.applied is None:
            self._consume(observed)

        metrics = score_canary(tuple(self._pairs), self._settings)
        verdict = detect_regression(metrics, self._settings)
        window_expired = observed >= deadline
        if verdict.rollback or window_expired:
            effective = verdict
            if not verdict.rollback:
                effective = RegressionVerdict(
                    rollback=True,
                    reason="window_expired_without_decision",
                    evidence_status=verdict.evidence_status,
                    sample_counts=verdict.sample_counts,
                    evidence_label=self._settings.evidence_label,
                )
            rollback = self._rollback.rollback(effective, self._deployment, observed)
        else:
            rollback = RollbackResult(
                status="not_needed",
                reason=verdict.reason,
                observed_at=timestamp(observed),
                alias=self._deployment.alias,
                current_version=None,
                target_version=self._deployment.previous_version,
                writes=0,
                evidence_label=self._settings.evidence_label,
            )

        counts = metrics.sample_counts
        self._audit_safely(
            CanaryAuditEvent(
                event="canary.tick",
                task_id=self._settings.task_id,
                status=rollback.status,
                observed_at=timestamp(observed),
                evidence_label=self._settings.evidence_label,
                reason=rollback.reason,
                synthetic=True,
                alias=self._deployment.alias,
                writes=rollback.writes,
                counts={
                    "calls_observed": counts["calls_observed"],
                    "calls_canary": counts["calls_canary"],
                    "calls_control": counts["calls_control"],
                    "calls_scored": counts["calls_scored"],
                },
                error_type=rollback.error_type,
            )
        )
        return CanaryTickResult(
            observed_at=timestamp(observed),
            window_deadline=timestamp(deadline),
            split_fraction=self._split.fraction,
            split_salt=self._split.salt,
            calls_observed=counts["calls_observed"],
            calls_canary=counts["calls_canary"],
            calls_control=counts["calls_control"],
            metrics=metrics,
            verdict=verdict,
            rollback=rollback,
            evidence_label=self._settings.evidence_label,
            synthetic=True,
        )

    def _consume(self, observed: datetime) -> None:
        """Consume at most one configured batch of safe, paired observations."""
        for _ in range(self._settings.max_calls_per_tick):
            if len(self._pairs) >= self._settings.max_pairs:
                break
            observation = self._events.next_call()
            if observation is None:
                break
            assignment = self._split.assign(observation.call_id)
            outcome = self._outcomes.for_call(observation.call_id)
            ignored = self._settings.ignored_outcome_fields()
            canary = PredictionRecord.from_fields(
                observation.call_id,
                Arm.CANARY,
                observation.canary_fields,
                ignored,
                observed_at=observation.observed_at,
                synthetic=observation.synthetic,
            )
            baseline = PredictionRecord.from_fields(
                observation.call_id,
                Arm.CHAMPION,
                observation.champion_fields,
                ignored,
                observed_at=observation.observed_at,
                synthetic=observation.synthetic,
            )
            self._pairs.append(
                CanaryPair(
                    call_id=assignment.call_id,
                    arm=assignment.arm,
                    canary=canary,
                    champion=baseline,
                    outcome=outcome,
                )
            )

    def _audit_safely(self, event: CanaryAuditEvent) -> None:
        """Emit one event without letting a logging failure break the tick."""
        try:
            self._audit.record(event)
        except Exception:
            pass


def run_fixture_rehearsal(settings: CanarySettings | None = None) -> CanaryTickResult:
    """Run one deterministic regression rehearsal over read-only fixtures.

    Args:
        settings: Optional typed overrides; the default uses a 20% canary
            fraction, a fixed salt and a 60-second fake-clock window.

    Returns:
        One labelled result with an applied rollback when the injected canary
        predictions regress on booking fields.

    """
    effective = settings or CanarySettings(
        canary_fraction=MAX_CANARY_FRACTION,
        split_salt=DEMO_SPLIT_SALT,
        min_paired_calls=2,
        max_calls_per_tick=16,
        max_pairs=32,
        window_seconds=60.0,
    )
    fixture_path = Path(effective.fixtures_path)
    if not fixture_path.is_absolute():
        fixture_path = REPO_ROOT / fixture_path

    clock = FakeClock()
    ignored = effective.ignored_outcome_fields()
    observations: list[CanaryCallObservation] = []
    outcomes: dict[str, OutcomeRecord] = {}
    fixture = fixture_path.read_text(encoding="utf-8")
    for line in fixture.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            continue
        call_id = str(row.get("call_id", ""))
        slots = row.get("expected_slots")
        if not call_id or not isinstance(slots, dict) or not slots:
            continue
        degraded = dict(slots)
        degraded["guests"] = -1  # deterministic booking-field regression
        observations.append(
            CanaryCallObservation(
                call_id=call_id,
                canary_fields=degraded,
                champion_fields=slots,
                observed_at=clock.now(),
                synthetic=True,
            )
        )
        outcomes[call_id] = OutcomeRecord.from_payload(
            call_id,
            {"status": "confirmed", "final_record": slots, "updated_at": ""},
            ignored,
            True,
        )

    promotion = PromotionRecord(
        actor_id="fake-gate-promoter",
        action="promote",
        reason="synthetic canary rehearsal",
        before_version="v1",
        after_version="v2",
        task_id="P4-01",
    )
    runner = CanaryRunner(
        settings=effective,
        promotion=promotion,
        event_source=FakeEventSource(observations),
        outcome_source=FakeOutcomeSource(outcomes),
        alias_store=InMemoryAliasStore({"@champion": "v2"}),
        audit=CollectingCanaryAudit(),
        clock=clock,
        rollback_audit=InMemoryRollbackAudit(),
    )
    return runner.tick()


def main() -> int:
    """CLI entry point: print one labelled synthetic rehearsal report."""
    report = run_fixture_rehearsal().to_report()
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0 if report["rollback"]["status"] in {"applied", "idempotent"} else 2


if __name__ == "__main__":  # pragma: no cover - exercised via python -m
    raise SystemExit(main())
