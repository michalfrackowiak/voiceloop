"""Deterministic P4-02 runner: replay windows -> detectors -> policy -> fake pipeline.

No real scheduler, training, MLflow, network or GPU is used. The pipeline step is
an injectable protocol and its default implementation only records the call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, TextIO, runtime_checkable

from pydantic import ValidationError as PydanticValidationError

from voiceloop_common import Clock, ConfigError, ValidationError, utcnow

from .detectors import (
    DetectorReport,
    InMemoryOutcomeWindowSource,
    InMemoryScoreWindowSource,
    OutcomeWindow,
    OutcomeWindowSource,
    ScoreWindowSource,
    TriageScoreWindow,
    assert_no_sensitive_text,
    detect_all,
)
from .policy import (
    AuditSink,
    InMemoryAuditSink,
    ManualOverride,
    TriggerDecision,
    decide_trigger,
    manual_override_from_settings,
)
from .settings import SYNTHETIC_LABEL, DriftSettings

REPO_ROOT = Path(__file__).resolve().parents[3]


@runtime_checkable
class RetrainingPipeline(Protocol):
    """Injected retraining seam; the runner never trains by itself."""

    def invoke(self, decision: Mapping[str, Any], evidence: Mapping[str, Any]) -> Mapping[str, Any]:
        """Start the downstream chain and return a compact result."""


@dataclass
class RecordingPipeline:
    """Fake pipeline that records invocations and performs no training."""

    invocations: list[tuple[dict[str, Any], dict[str, Any]]] = field(default_factory=list)

    def invoke(self, decision: Mapping[str, Any], evidence: Mapping[str, Any]) -> Mapping[str, Any]:
        """Record the request and return a labelled simulation result."""
        self.invocations.append((dict(decision), dict(evidence)))
        result: dict[str, Any] = {
            "label": SYNTHETIC_LABEL,
            "status": "simulated",
            "training_performed": False,
            "trigger_reason": decision.get("reason"),
        }
        assert_no_sensitive_text(result)
        return result


@dataclass(frozen=True, slots=True)
class RunnerReport:
    """One labelled drift evaluation and the resulting pipeline action."""

    decision: TriggerDecision
    detectors: DetectorReport
    invoked: bool
    pipeline_result: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a labelled, JSON-serialisable report with no raw observations."""
        payload: dict[str, Any] = {
            "label": SYNTHETIC_LABEL,
            "triggered": self.decision.triggered,
            "invoked": self.invoked,
            "decision": self.decision.to_dict(),
            "detectors": self.detectors.to_dict(),
            "pipeline_result": dict(self.pipeline_result) if self.pipeline_result else None,
            "no_training_performed": True,
            "calendar_tick_is_trigger": False,
        }
        assert_no_sensitive_text(payload)
        return payload


def run_once(
    *,
    score_source: ScoreWindowSource,
    outcome_source: OutcomeWindowSource,
    pipeline: RetrainingPipeline,
    new_verified_count: int,
    settings: DriftSettings | None = None,
    manual_override: ManualOverride | None = None,
    audit_sink: AuditSink | None = None,
    clock: Clock = utcnow,
    scheduled_tick: bool = False,
    last_run_at: datetime | str | None = None,
) -> RunnerReport:
    """Evaluate replayed windows and invoke the injected pipeline only if allowed.

    Args:
        score_source: Injected baseline/current triage score windows.
        outcome_source: Injected baseline/current outcome windows.
        pipeline: Injectable retraining seam; never a real trainer.
        new_verified_count: Newly verified examples available for retraining.
        settings: Typed drift settings and thresholds.
        manual_override: Audited operator override, or None.
        audit_sink: Write-ahead audit seam for manual overrides.
        clock: Injected UTC clock for deterministic decisions.
        scheduled_tick: Audit-only marker for a calendar-driven invocation.
        last_run_at: Audit-only previous-run timestamp.

    Returns:
        A report whose ``invoked`` flag is True only for an allowed trigger.

    """
    config = settings or DriftSettings()
    detectors = detect_all(score_source.load(), outcome_source.load(), config)
    previous = datetime.fromisoformat(last_run_at) if isinstance(last_run_at, str) else last_run_at
    decision = decide_trigger(
        new_verified_count=new_verified_count,
        drifted=detectors.drifted,
        settings=config,
        manual_override=manual_override,
        audit_sink=audit_sink,
        clock=clock,
        scheduled_tick=scheduled_tick,
        last_run_at=previous,
    )
    if not decision.triggered:
        return RunnerReport(decision=decision, detectors=detectors, invoked=False)

    result = pipeline.invoke(decision.to_dict(), detectors.to_dict())
    payload = dict(result)
    assert_no_sensitive_text(payload)
    return RunnerReport(
        decision=decision,
        detectors=detectors,
        invoked=True,
        pipeline_result=payload,
    )


class TurnScorer(Protocol):
    """Callable adapting a turn event to one P2-01-style failure score."""

    def __call__(self, event: Mapping[str, Any]) -> float:
        """Return a score in [0, 1] for a schema-valid turn event."""


def default_fixture_scorer() -> TurnScorer:
    """Build a pure-Python scorer adapter over the read-only P2-01 engine."""
    from services.triage.scoring import ScoringEngine

    engine = ScoringEngine()

    def score(event: Mapping[str, Any]) -> float:
        """Score one fixture event without persisting an explanation."""
        return engine.score(dict(event), persist=False).score

    return score


def _resolve(path: str | Path) -> Path:
    """Resolve a repo-relative path against the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    """Read a JSON Lines label manifest without modifying it."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigError(f"cannot read replay labels: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid replay JSON at line {number}") from exc
        if not isinstance(decoded, dict):
            raise ValidationError(f"replay line {number} must be an object")
        rows.append(decoded)
    return tuple(rows)


def fixture_sha256(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of the read-only fixture manifest."""
    return hashlib.sha256(_resolve(path).read_bytes()).hexdigest()


def verify_fixture_hashes(settings: DriftSettings | None = None) -> dict[str, bool]:
    """Check the fixture digest and sealed golden content hash read-only."""
    config = settings or DriftSettings()
    manifest_path = _resolve(config.golden_artifact_dir) / f"{config.golden_version}.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read sealed golden manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValidationError("sealed golden manifest must be an object")
    return {
        "fixture_sha256_ok": fixture_sha256(config.fixtures_path) == config.fixture_sha256,
        "golden_content_hash_ok": manifest.get("content_hash") == config.golden_content_hash,
    }


def build_fixture_replay_sources(
    path: str | Path | None = None,
    *,
    settings: DriftSettings | None = None,
    scorer: TurnScorer | None = None,
) -> tuple[InMemoryScoreWindowSource, InMemoryOutcomeWindowSource]:
    """Build deterministic baseline/current windows from synthetic fixtures.

    Calls are split by call ID (never by utterance); score values come from the
    injected scorer, and outcome statuses from the synthetic label manifest.
    The manifest and sealed golden stand-in are hash-checked read-only.
    """
    config = settings or DriftSettings()
    hashes = verify_fixture_hashes(config)
    if not all(hashes.values()):
        raise ValidationError("fixture or sealed golden hash mismatch; replay refused")

    rows = _read_jsonl(_resolve(path or config.fixtures_path))
    calls = sorted({str(row.get("call_id", "")) for row in rows})
    if len(calls) < 2:
        raise ValidationError("fixture replay needs at least two calls for a call-level split")
    baseline_calls = set(calls[: len(calls) // 2])
    score = scorer or default_fixture_scorer()

    baseline_scores: list[float] = []
    current_scores: list[float] = []
    baseline_outcomes: list[str] = []
    current_outcomes: list[str] = []
    for row in rows:
        call_id = str(row.get("call_id", ""))
        event_path = row.get("event_path")
        if not isinstance(event_path, str) or not event_path:
            raise ValidationError("replay row needs an event_path")
        try:
            event = json.loads(_resolve(event_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"cannot read replay event: {exc}") from exc
        if not isinstance(event, Mapping):
            raise ValidationError("replay event must be an object")
        value = score(event)
        score_window = baseline_scores if call_id in baseline_calls else current_scores
        score_window.append(float(value))

        expected = row.get("expected_outcome")
        if expected is not None:
            outcome_window = baseline_outcomes if call_id in baseline_calls else current_outcomes
            outcome_window.append(str(expected))

    return (
        InMemoryScoreWindowSource(
            TriageScoreWindow(baseline=tuple(baseline_scores), current=tuple(current_scores))
        ),
        InMemoryOutcomeWindowSource(
            OutcomeWindow(baseline=tuple(baseline_outcomes), current=tuple(current_outcomes))
        ),
    )


def write_report(report: Mapping[str, Any], path: str | Path) -> Path:
    """Validate and write one synthetic report as sorted UTF-8 JSON."""
    payload = dict(report)
    assert_no_sensitive_text(payload)
    destination = _resolve(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _validated_copy(settings: DriftSettings, updates: Mapping[str, Any]) -> DriftSettings:
    """Apply CLI overrides with full pydantic validation, not a blind copy."""
    payload = settings.model_dump()
    payload.update(dict(updates))
    return DriftSettings.model_validate(payload)


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: DriftSettings | None = None,
    score_source: ScoreWindowSource | None = None,
    outcome_source: OutcomeWindowSource | None = None,
    pipeline: RetrainingPipeline | None = None,
    audit_sink: AuditSink | None = None,
    clock: Clock = utcnow,
    stdout: TextIO | None = None,
) -> int:
    """Run one policy evaluation (no scheduler loop and no real training)."""
    writer = stdout if stdout is not None else sys.stdout
    parser = argparse.ArgumentParser(
        prog="python -m infra.pipelines.drift.runner",
        description="Evaluate synthetic drift windows; a calendar tick never triggers training.",
    )
    parser.add_argument("--fixtures", default=None, help="Read-only fixture label manifest.")
    parser.add_argument("--new-verified", type=int, default=0)
    parser.add_argument("--min-new-verified", type=int, default=None)
    parser.add_argument("--manual-actor", default=None)
    parser.add_argument("--manual-reason", default=None)
    parser.add_argument("--scheduled-tick", action="store_true")
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        config = settings or DriftSettings()
        if args.fixtures is not None:
            config = _validated_copy(config, {"fixtures_path": args.fixtures})
        if args.min_new_verified is not None:
            config = _validated_copy(config, {"min_new_verified": args.min_new_verified})
        if bool(args.manual_actor) != bool(args.manual_reason):
            raise ValidationError("manual override needs both actor and reason")
        if args.manual_actor and args.manual_reason:
            override: ManualOverride | None = ManualOverride(
                actor_id=args.manual_actor, reason_code=args.manual_reason
            )
        else:
            override = manual_override_from_settings(config)

        scores = score_source
        outcomes = outcome_source
        if scores is None or outcomes is None:
            scores, outcomes = build_fixture_replay_sources(config.fixtures_path, settings=config)
        sink = audit_sink if audit_sink is not None else InMemoryAuditSink()
        report = run_once(
            score_source=scores,
            outcome_source=outcomes,
            pipeline=pipeline if pipeline is not None else RecordingPipeline(),
            new_verified_count=args.new_verified,
            settings=config,
            manual_override=override,
            audit_sink=sink,
            clock=clock,
            scheduled_tick=args.scheduled_tick,
        )
        if args.output:
            write_report(report.to_dict(), args.output)
    except (ValidationError, PydanticValidationError, ConfigError) as exc:
        writer.write(f"ERROR: {type(exc).__name__}\n")
        return 2

    writer.write(f"DECISION: {json.dumps(report.decision.to_dict(), sort_keys=True)}\n")
    if isinstance(sink, InMemoryAuditSink):
        for entry in sink.entries:
            writer.write(f"AUDIT: {json.dumps(entry, sort_keys=True)}\n")
    writer.write(f"REPORT: {json.dumps(report.to_dict(), sort_keys=True)}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - console smoke seam
    raise SystemExit(main())
