"""Compare two candidate runners on our own fixture sets.

The P1-07 harness is reused read-only: fixture/event loading, per-turn scoring,
critical slices, paired bootstrap CIs and contract-valid per-candidate reports
all come from ``services.evalgate``. This module never edits inputs, never
imports a real ASR runtime and does not consult public leaderboards.

Both evaluated sets are synthetic: the sealed v1 golden *stand-in* labels and
the general fixture pool. Candidate A and candidate B are scored on identical,
ordered turns so turn difficulty cancels in the paired bootstrap.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from services.datasets.golden.seal import load_labels, verify_seal
from services.evalgate.bootstrap import paired_bootstrap_ci
from services.evalgate.harness import (
    build_report_timestamp,
    build_slice_metrics,
    load_event,
    load_fixture_rows,
)
from services.evalgate.metrics import (
    CORE_METRIC_NAMES,
    EntityPatterns,
    TurnMetrics,
    aggregate_metrics,
    evaluate_turn,
)
from services.evalgate.report import build_report
from voiceloop_common import ConfigError, ValidationError

from .report import SCHEMA_REPORT_LABEL, ComparisonData, SetEvaluation
from .runners import IDENTITY_NAME, CandidateRunner, available_candidates, get_candidate
from .settings import BenchmarkSettings

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class BenchmarkTurn:
    """One read-only labelled fixture turn plus its committed event."""

    row: Mapping[str, Any]
    event: Mapping[str, Any]


@dataclass(frozen=True)
class BenchmarkInputs:
    """Read-only general-pool rows, golden-stand-in rows and provenance."""

    general_rows: tuple[dict[str, Any], ...]
    golden_rows: tuple[dict[str, Any], ...]
    provenance: dict[str, Any]


def _resolve(path: str | Path) -> Path:
    """Resolve a repo-relative path against the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def load_benchmark_inputs(settings: BenchmarkSettings) -> BenchmarkInputs:
    """Load and verify the read-only fixture pool and sealed golden stand-in.

    Seal verification is deliberately called without audit context so it makes
    no audit-log entry. The sealed set is read for labels only: it is never
    trained on, tuned on or written to.

    Args:
        settings: Typed benchmark settings.

    Returns:
        General-pool rows, golden-stand-in-labelled rows and provenance.

    Raises:
        ValidationError: If the seal, hash, synthetic flags or label alignment
            do not match the committed v1 synthetic stand-in.

    """
    fixture_rows = load_fixture_rows(settings.fixtures_path)
    verification = verify_seal(settings.golden_artifact_dir, settings.golden_version)
    verification.assert_ok()
    manifest = verification.manifest
    if manifest is None:  # pragma: no cover - guaranteed by a passing seal
        raise ValidationError("verified golden stand-in has no manifest")
    if manifest.content_hash != settings.expected_golden_content_hash:
        raise ValidationError(
            "golden stand-in content hash changed; refusing to benchmark: "
            f"{manifest.content_hash!r}"
        )
    if not manifest.synthetic or not manifest.workflow_stand_in:
        raise ValidationError("golden stand-in is no longer labelled synthetic/workflow stand-in")
    if manifest.review_mode != "simulated_synthetic_labels":
        raise ValidationError("golden stand-in review mode is not the expected synthetic mode")

    labels_path = _resolve(settings.golden_artifact_dir) / manifest.labels_path
    labels = load_labels(labels_path, expected_version=settings.golden_version)
    label_by_source = {label.source_id: label for label in labels}
    if len(label_by_source) != len(labels):
        raise ValidationError("golden stand-in contains duplicate source IDs")
    if len(labels) != manifest.count or len(labels) != len(fixture_rows):
        raise ValidationError("golden stand-in labels and fixture rows are not 1:1")

    golden_rows: list[dict[str, Any]] = []
    for row in fixture_rows:
        source_id = f"{row['call_id']}:{row['turn_index']}"
        label = label_by_source.get(source_id)
        if label is None:
            raise ValidationError(f"fixture turn {source_id!r} has no golden stand-in label")
        if label.call_id != row["call_id"] or label.turn_index != row["turn_index"]:
            raise ValidationError(f"golden stand-in label {source_id!r} is misaligned")
        golden_row = dict(row)
        golden_row["expected_transcript"] = label.corrected_transcript
        golden_row["expected_slots"] = dict(label.slots)
        golden_rows.append(golden_row)

    source_pairs_identical = all(
        row["expected_transcript"] == golden_row["expected_transcript"]
        and row["expected_slots"] == golden_row["expected_slots"]
        for row, golden_row in zip(fixture_rows, golden_rows, strict=True)
    )
    provenance = {
        "fixtures_path": settings.fixtures_path,
        "fixture_sha256": hashlib.sha256(_resolve(settings.fixtures_path).read_bytes()).hexdigest(),
        "golden_artifact_dir": settings.golden_artifact_dir,
        "golden_version": manifest.version,
        "golden_dataset_id": manifest.dataset_id,
        "golden_content_hash": manifest.content_hash,
        "golden_seal_ok": True,
        "golden_synthetic": manifest.synthetic,
        "golden_workflow_stand_in": manifest.workflow_stand_in,
        "golden_review_mode": manifest.review_mode,
        "golden_used_for_training_or_tuning": False,
        "source_pairs_identical": source_pairs_identical,
        "turns": len(golden_rows),
        "calls": len({str(row["call_id"]) for row in golden_rows}),
        "public_leaderboards_used": False,
        "real_model_adapters_used": False,
        "synthetic_standin": True,
        "limitation": (
            "SYNTHETIC/FIXTURE STAND-IN: tone/noise/silence carriers and "
            "simulated labels, not human-reviewed audio or deployed artifacts."
        ),
    }
    return BenchmarkInputs(
        general_rows=tuple(fixture_rows),
        golden_rows=tuple(golden_rows),
        provenance=provenance,
    )


def _turns_from_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[BenchmarkTurn, ...]:
    """Pair each labelled row with its committed synthetic event."""
    return tuple(BenchmarkTurn(row=row, event=load_event(str(row["event_path"]))) for row in rows)


def _score_candidate(
    turns: Sequence[BenchmarkTurn],
    runner: CandidateRunner,
    patterns: EntityPatterns,
) -> tuple[TurnMetrics, ...]:
    """Score a runner's predictions with the P1-07 per-turn evaluator."""
    return tuple(
        evaluate_turn(
            str(turn.row["expected_transcript"]),
            dict(turn.row["expected_slots"]),
            runner.predict(turn.row, turn.event),
            patterns,
        )
        for turn in turns
    )


def _metric_statistic(metric: str) -> Callable[[Sequence[TurnMetrics]], float]:
    """Return the P1-07 aggregate statistic for one named core metric."""

    def statistic(turns: Sequence[TurnMetrics]) -> float:
        return aggregate_metrics(turns)[metric]

    return statistic


def _per_candidate_report(
    rows: Sequence[Mapping[str, Any]],
    runner: CandidateRunner,
    candidate_turns: Sequence[TurnMetrics],
    reference_turns: Sequence[TurnMetrics],
    settings: BenchmarkSettings,
    patterns: EntityPatterns,
) -> dict[str, Any]:
    """Build one contract-valid report with CIs of candidate minus identity."""
    intervals = {
        metric: paired_bootstrap_ci(
            reference_turns,
            candidate_turns,
            _metric_statistic(metric),
            level=settings.bootstrap_level,
            iterations=settings.bootstrap_iterations,
            seed=settings.bootstrap_seed,
        )
        for metric in CORE_METRIC_NAMES
    }
    return build_report(
        artifact_uri=runner.artifact_uri,
        eval_sets=[SCHEMA_REPORT_LABEL],
        metrics=aggregate_metrics(candidate_turns),
        confidence_intervals=intervals,
        slice_metrics=build_slice_metrics(
            rows, candidate_turns, patterns, settings.other_language_pattern
        ),
        decoding_settings=settings.decoding_settings(),
        gate_decision={
            "outcome": "not_evaluated",
            "reason": (
                "Synthetic benchmark stand-in; CIs bound candidate minus "
                "identity. e2e_booking_accuracy from audio is primary; "
                "WER alone never promotes."
            ),
            "evaluated_at": build_report_timestamp(),
        },
    )


def _delta_intervals(
    candidate_a: Sequence[TurnMetrics],
    candidate_b: Sequence[TurnMetrics],
    settings: BenchmarkSettings,
) -> dict[str, dict[str, float]]:
    """Return paired-bootstrap intervals for ``A - B`` on all five metrics."""
    a_metrics = aggregate_metrics(candidate_a)
    b_metrics = aggregate_metrics(candidate_b)
    deltas: dict[str, dict[str, float]] = {}
    for metric in CORE_METRIC_NAMES:
        interval = paired_bootstrap_ci(
            candidate_b,
            candidate_a,
            _metric_statistic(metric),
            level=settings.bootstrap_level,
            iterations=settings.bootstrap_iterations,
            seed=settings.bootstrap_seed,
        )
        deltas[metric] = {
            "point": a_metrics[metric] - b_metrics[metric],
            "lower": interval.lower,
            "upper": interval.upper,
            "level": interval.level,
        }
    return deltas


def evaluate_set(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[CandidateRunner],
    settings: BenchmarkSettings,
    patterns: EntityPatterns,
) -> SetEvaluation:
    """Score all candidates on one aligned set and compute their ``A - B`` CIs."""
    if len(candidates) != 2:
        raise ConfigError("a benchmark comparison requires exactly two candidates")
    turns = _turns_from_rows(rows)
    reference = _score_candidate(turns, get_candidate(IDENTITY_NAME), patterns)
    scored = {runner.name: _score_candidate(turns, runner, patterns) for runner in candidates}
    reports = tuple(
        _per_candidate_report(rows, runner, scored[runner.name], reference, settings, patterns)
        for runner in candidates
    )
    candidate_a, candidate_b = candidates[0].name, candidates[1].name
    return SetEvaluation(
        name=name,
        label=SCHEMA_REPORT_LABEL,
        turns=len(rows),
        calls=len({str(row["call_id"]) for row in rows}),
        candidate_reports=reports,
        metrics_by_candidate={name: aggregate_metrics(turns) for name, turns in scored.items()},
        slice_by_candidate={
            name: build_slice_metrics(rows, turns, patterns, settings.other_language_pattern)
            for name, turns in scored.items()
        },
        deltas=_delta_intervals(scored[candidate_a], scored[candidate_b], settings),
        candidate_a=candidate_a,
        candidate_b=candidate_b,
    )


def compare_candidates(settings: BenchmarkSettings) -> ComparisonData:
    """Compare the two configured candidates on both read-only fixture sets.

    Args:
        settings: Typed benchmark settings; candidate order defines ``A - B``.

    Returns:
        Raw comparison data ready for the artifact builder in ``report.py``.

    Raises:
        ConfigError: If the comparison does not name exactly two candidates.

    """
    names = settings.candidate_names()
    if len(names) != 2:
        raise ConfigError(
            "benchmark comparison requires exactly two candidates; "
            f"got {len(names)} ({', '.join(names) or 'none'}); available: "
            f"{', '.join(available_candidates())}"
        )
    candidates = tuple(get_candidate(name) for name in names)
    inputs = load_benchmark_inputs(settings)
    patterns = EntityPatterns().with_overrides(settings.entity_pattern_overrides)
    sets = (
        evaluate_set("golden_standin", inputs.golden_rows, candidates, settings, patterns),
        evaluate_set("general_capability", inputs.general_rows, candidates, settings, patterns),
    )
    return ComparisonData(
        candidates=(names[0], names[1]),
        provenance=inputs.provenance,
        settings=settings,
        sets=sets,
    )
