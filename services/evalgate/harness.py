"""Fixture evaluation harness and CLI for the P1-07 evaluation contract.

The CLI reads the read-only synthetic labels in ``tests/fixtures/turns.jsonl``,
runs two settings-selectable fake artifacts (champion and challenger) over the
same turns, computes paired-bootstrap CIs on their differences, and writes a
contract-valid ``report.json``.

End-to-end rule (primary metric): predicted slots are the artifact's parser
output for its STT transcript of the turn's audio. The fake ``identity``
artifact replays each recorded event's ``stt.text`` and ``parser.output``;
``oracle`` uses the synthetic labels and ``no_slots`` emits an empty transcript
and empty slots. Gold slots are ``expected_slots`` in the labels manifest. A
turn counts as correct only when every booking field matches gold exactly
(including no-booking turns, where both sides must be empty).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

from voiceloop_common import ConfigError, ValidationError, utcnow

from .bootstrap import paired_bootstrap_ci
from .metrics import (
    CORE_METRIC_NAMES,
    ENTITY_TYPES,
    EntityPatterns,
    EvaluationError,
    Prediction,
    TurnMetrics,
    aggregate_metrics,
    evaluate_turn,
    extract_entities,
)
from .report import build_report, write_report
from .settings import ArtifactKind, EvalgateSettings, EvalSetName

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_KINDS: tuple[str, str, str] = ("identity", "oracle", "no_slots")
EVAL_SET_NAMES: tuple[str, str, str] = ("golden", "regression", "general_capability")
CRITICAL_SLICES: tuple[str, str, str, str, str] = (
    "names",
    "dates",
    "numbers",
    "silence",
    "other_languages",
)
REQUIRED_LABEL_FIELDS: tuple[str, ...] = (
    "call_id",
    "turn_index",
    "event_path",
    "expected_transcript",
    "expected_slots",
    "audio_kind",
)


def _resolve(path: str | Path) -> Path:
    """Resolve a repo-relative path against the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def load_fixture_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load and minimally validate the read-only labelled turns manifest.

    Args:
        path: Path to a UTF-8 JSON Lines file of labelled fixture turns.

    Returns:
        One decoded record per non-empty line, in file order.

    """
    fixture_path = _resolve(path)
    try:
        lines = fixture_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvaluationError(f"cannot read fixtures: {fixture_path}") from exc

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"invalid JSON in fixtures line {line_number}: {exc}") from exc
        if not isinstance(decoded, dict):
            raise EvaluationError(f"fixtures line {line_number} must be a JSON object")
        missing = [field for field in REQUIRED_LABEL_FIELDS if field not in decoded]
        if missing:
            raise EvaluationError(f"fixtures line {line_number} missing: {', '.join(missing)}")
        if not isinstance(decoded["expected_slots"], dict):
            raise EvaluationError(f"fixtures line {line_number}: expected_slots must be an object")
        if not isinstance(decoded["expected_transcript"], str):
            raise EvaluationError(f"fixtures line {line_number}: expected_transcript must be text")
        rows.append(decoded)
    if not rows:
        raise EvaluationError("fixtures contain no turns")
    return rows


def load_event(event_path: str) -> dict[str, Any]:
    """Read a fixture turn event for the identity fake artifact."""
    path = _resolve(event_path)
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read fixture event: {path}") from exc
    if not isinstance(decoded, dict):
        raise EvaluationError(f"fixture event must be a JSON object: {path}")
    return decoded


def predict(kind: ArtifactKind, row: Mapping[str, Any], event: Mapping[str, Any]) -> Prediction:
    """Produce one turn prediction from a settings-selectable fake artifact.

    Args:
        kind: ``identity`` (replay event STT/parser), ``oracle`` (synthetic
            labels) or ``no_slots`` (empty transcript and slots).
        row: Labelled fixture turn.
        event: Recorded turn event, used only by ``identity``.

    Returns:
        Predicted transcript and parser slots.

    """
    if kind == "identity":
        stt = event.get("stt")
        parser = event.get("parser")
        transcript = stt.get("text", "") if isinstance(stt, dict) else ""
        slots = parser.get("output", {}) if isinstance(parser, dict) else {}
    elif kind == "oracle":
        transcript = row["expected_transcript"]
        slots = row["expected_slots"]
    elif kind == "no_slots":
        transcript, slots = "", {}
    else:  # pragma: no cover - guarded by settings and argparse choices
        raise EvaluationError(f"unknown fake artifact: {kind!r}")
    if not isinstance(transcript, str) or not isinstance(slots, dict):
        raise EvaluationError(f"invalid prediction produced by {kind!r}")
    return Prediction(transcript=transcript, slots=dict(slots))


def _reference(row: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return the labelled transcript and slots used as evaluation gold."""
    return str(row["expected_transcript"]), dict(row["expected_slots"])


def evaluate_rows(
    rows: Sequence[Mapping[str, Any]],
    champion_kind: ArtifactKind,
    challenger_kind: ArtifactKind,
    patterns: EntityPatterns | None = None,
) -> tuple[list[TurnMetrics], list[TurnMetrics]]:
    """Score champion and challenger over identical, ordered fixture turns."""
    champion: list[TurnMetrics] = []
    challenger: list[TurnMetrics] = []
    for row in rows:
        reference_transcript, reference_slots = _reference(row)
        event = (
            load_event(str(row["event_path"]))
            if "identity" in (champion_kind, challenger_kind)
            else {}
        )
        for kind, results in ((champion_kind, champion), (challenger_kind, challenger)):
            prediction = predict(kind, row, event)
            results.append(
                evaluate_turn(reference_transcript, reference_slots, prediction, patterns)
            )
    return champion, challenger


def slice_names_for_row(
    row: Mapping[str, Any],
    patterns: EntityPatterns | None = None,
    other_language_pattern: str = r"[^\x00-\x7f]",
) -> set[str]:
    """Classify one turn into zero or more critical slices.

    Names/dates/numbers slices are driven by entities in the *reference*
    transcript; ``silence`` by the fixture's ``audio_kind``; ``other_languages``
    by a configurable regex (no fixture turn matches the default, since the
    synthetic set is English-only). Turns may belong to several slices.
    """
    transcript = str(row.get("expected_transcript", ""))
    extracted: dict[str, Counter[str]] = extract_entities(transcript, patterns)
    names = {
        entity_type for entity_type in ENTITY_TYPES if sum(extracted[entity_type].values()) > 0
    }
    if str(row.get("audio_kind", "")) == "silence":
        names.add("silence")
    try:
        if re.search(other_language_pattern, transcript):
            names.add("other_languages")
    except re.error as exc:
        raise EvaluationError(f"invalid other_language_pattern: {exc}") from exc
    return names


def _neutral_slice_metrics() -> dict[str, float]:
    """Neutral values for a slice with no turns: no evidence, not a regression."""
    return {
        "turns": 0.0,
        "wer": 0.0,
        "entity_error": 0.0,
        "slot_f1": 1.0,
        "schema_valid_rate": 1.0,
        "e2e_booking_accuracy": 1.0,
    }


def build_slice_metrics(
    rows: Sequence[Mapping[str, Any]],
    challenger_turns: Sequence[TurnMetrics],
    patterns: EntityPatterns | None = None,
    other_language_pattern: str = r"[^\x00-\x7f]",
) -> dict[str, dict[str, float]]:
    """Aggregate challenger turns by critical slice for gate regression checks."""
    if len(rows) != len(challenger_turns):
        raise EvaluationError("rows and per-turn metrics must be aligned")
    groups: dict[str, list[TurnMetrics]] = {name: [] for name in CRITICAL_SLICES}
    for row, metrics in zip(rows, challenger_turns, strict=True):
        for name in slice_names_for_row(row, patterns, other_language_pattern):
            groups.setdefault(name, []).append(metrics)

    result: dict[str, dict[str, float]] = {}
    for name in CRITICAL_SLICES:
        group = groups[name]
        if group:
            values = aggregate_metrics(group)
            values["turns"] = float(len(group))
            result[name] = values
        else:
            result[name] = _neutral_slice_metrics()
    return result


def _turn_metric(turns: Sequence[TurnMetrics], metric: str) -> float:
    """Reduce a resample of turns to one named aggregate metric."""
    return aggregate_metrics(turns)[metric]


def evaluate(
    artifact_uri: str | None = None,
    eval_sets: Sequence[EvalSetName] | None = None,
    config: EvalgateSettings | None = None,
) -> dict[str, Any]:
    """Implement ``evaluate(artifact_uri, eval_sets, config) -> report.json``.

    Returns the validated report payload; use :func:`write_report` to persist
    it. Artifact URIs and eval-set names override settings when supplied.
    """
    settings = config or EvalgateSettings()
    patterns = EntityPatterns().with_overrides(settings.entity_pattern_overrides)
    rows = load_fixture_rows(settings.fixtures_path)
    champion, challenger = evaluate_rows(
        rows, settings.champion_artifact, settings.challenger_artifact, patterns
    )
    intervals = {
        name: paired_bootstrap_ci(
            champion,
            challenger,
            partial(_turn_metric, metric=name),
            level=settings.bootstrap_level,
            iterations=settings.bootstrap_iterations,
            seed=settings.bootstrap_seed,
        )
        for name in CORE_METRIC_NAMES
    }
    unknown_sets = sorted(set(eval_sets or settings.eval_sets) - set(EVAL_SET_NAMES))
    if unknown_sets:
        raise EvaluationError(f"unknown eval_sets: {', '.join(unknown_sets)}")
    return build_report(
        artifact_uri=artifact_uri or settings.artifact_uri,
        eval_sets=eval_sets or settings.eval_sets,
        metrics=aggregate_metrics(challenger),
        confidence_intervals=intervals,
        slice_metrics=build_slice_metrics(
            rows, challenger, patterns, settings.other_language_pattern
        ),
        decoding_settings={
            "hotwords": list(settings.hotwords),
            "initial_prompt": settings.initial_prompt,
            "condition_on_previous_text": settings.condition_on_previous_text,
            "language": settings.language,
        },
        gate_decision={
            "outcome": "not_evaluated",
            "reason": (
                "Synthetic fixture smoke run; never promote on WER. "
                "e2e_booking_accuracy from audio is the primary promotion metric."
            ),
            "evaluated_at": build_report_timestamp(),
        },
    )


def build_report_timestamp() -> str:
    """Return the current UTC timestamp in the report's RFC 3339 format."""
    return utcnow().isoformat(timespec="seconds").replace("+00:00", "Z")


def _error_code(exc: Exception) -> str:
    """Map a known exception class to the structured CLI error code."""
    if isinstance(exc, ValidationError):
        return "validation_error"
    if isinstance(exc, ConfigError):
        return "config_error"
    if isinstance(exc, EvaluationError):
        return "evaluation_error"
    return "internal_error"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the harness CLI and write ``report.json``.

    Returns 0 on success and 2 on a known evaluation/configuration failure,
    printing a structured ``{code, message}`` error on stderr.
    """
    base = EvalgateSettings()
    parser = argparse.ArgumentParser(description="Evaluate fake artifacts on synthetic fixtures.")
    parser.add_argument("--fixtures", default=base.fixtures_path, help="Labelled turns JSONL path.")
    parser.add_argument("--output", default=base.output_path, help="Destination report.json path.")
    parser.add_argument("--artifact-uri", default=base.artifact_uri)
    parser.add_argument("--champion", choices=ARTIFACT_KINDS, default=base.champion_artifact)
    parser.add_argument("--challenger", choices=ARTIFACT_KINDS, default=base.challenger_artifact)
    parser.add_argument("--level", type=float, default=base.bootstrap_level)
    parser.add_argument("--iterations", type=int, default=base.bootstrap_iterations)
    parser.add_argument("--seed", type=int, default=base.bootstrap_seed)
    parser.add_argument(
        "--eval-set",
        action="append",
        choices=EVAL_SET_NAMES,
        dest="eval_sets",
        help="Override the eval sets (repeatable).",
    )
    args = parser.parse_args(argv)

    overrides: dict[str, Any] = {
        "fixtures_path": args.fixtures,
        "output_path": args.output,
        "artifact_uri": args.artifact_uri,
        "champion_artifact": args.champion,
        "challenger_artifact": args.challenger,
        "bootstrap_level": args.level,
        "bootstrap_iterations": args.iterations,
        "bootstrap_seed": args.seed,
    }
    if args.eval_sets:
        overrides["eval_sets"] = args.eval_sets
    settings = base.model_copy(update=overrides)

    try:
        report = evaluate(settings.artifact_uri, settings.eval_sets, settings)
        destination = write_report(report, settings.output_path)
    except (EvaluationError, ConfigError, ValidationError) as exc:
        print(
            json.dumps({"code": _error_code(exc), "message": str(exc)}),
            file=sys.stderr,
        )
        return 2

    primary = report["metrics"]["e2e_booking_accuracy"]
    print(f"Wrote report: {destination}")
    print(f"Primary metric (e2e_booking_accuracy from audio): {primary:.3f}")
    print("WER alone never justifies promotion (context.md section 2 invariant 4).")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via python -m
    raise SystemExit(main())
