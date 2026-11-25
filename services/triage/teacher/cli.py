"""One-command offline teacher rehearsal and its injectable orchestration seam."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from voiceloop_common import ConfigError, ValidationError

from .batch import (
    DeterministicFixtureTeacher,
    FixtureTurnSource,
    InMemoryTurnSource,
    TeacherTranscriber,
    TurnSource,
    run_batch,
)
from .disagreement import EventProductionTranscriptSource, ProductionTranscriptSource
from .report import (
    FixtureReviewedLabelSource,
    JsonFileReportSink,
    ReportSink,
    ReviewedLabelSource,
    build_report,
    evaluate_precision,
    verify_read_only_inputs,
)
from .scorer import (
    BaselineQueueSelector,
    BaselineScorer,
    P2_01BaselineScorer,
    QueueSelector,
    TeacherQueueSelector,
    build_candidates,
)
from .settings import RUN_LABEL, TeacherSettings


def run_teacher_rehearsal(
    settings: TeacherSettings | None = None,
    *,
    turns: TurnSource | None = None,
    teacher: TeacherTranscriber | None = None,
    production: ProductionTranscriptSource | None = None,
    baseline: BaselineScorer | None = None,
    baseline_selector: QueueSelector | None = None,
    enhanced_selector: QueueSelector | None = None,
    labels: ReviewedLabelSource | None = None,
    sink: ReportSink | None = None,
) -> dict[str, Any]:
    """Run batch transcription, one signal, and exploratory synthetic arithmetic.

    Every collaborator is injectable; defaults are clearly labelled in-memory
    or fixture-backed fakes. Read-only input hashes are checked before and after
    the run so an accidental fixture or golden-seal mutation fails closed.

    Args:
        settings: Typed settings; defaults to ``VOICELOOP_TEACHER_*`` environment.
        turns: Turn-event source; defaults to committed fixtures.
        teacher: Offline transcriber seam; defaults to the metadata fake.
        production: Production transcript seam; defaults to event STT text.
        baseline: Baseline score seam; defaults to read-only P2-01.
        baseline_selector: Selector ignoring the teacher signal.
        enhanced_selector: Selector consuming the one teacher signal.
        labels: Synthetic reviewed-label seam used only for precision.
        sink: Report destination; defaults to the configured JSON file.

    Returns:
        The internal report artifact, after writing it to the sink.

    Raises:
        ValidationError: If read-only inputs changed during the run or seams disagree.

    """
    config = settings or TeacherSettings()
    before = verify_read_only_inputs(config)
    turn_records = tuple((turns or FixtureTurnSource(config.fixtures_path)).iter_turns())
    teacher_seam = teacher or DeterministicFixtureTeacher(
        config.teacher_model_id,
        low_confidence_ceiling=config.low_confidence_ceiling,
        no_speech_prob_ceiling=config.no_speech_prob_ceiling,
        algorithm=config.hash_algorithm,
    )
    batch = run_batch(
        InMemoryTurnSource(turn_records),
        teacher_seam,
        config,
        fixtures_sha256=before["fixtures_sha256"],
    )
    candidates = build_candidates(
        turn_records,
        batch,
        production or EventProductionTranscriptSource(),
        baseline or P2_01BaselineScorer(),
        config,
    )
    comparison = evaluate_precision(
        candidates,
        turn_records,
        labels or FixtureReviewedLabelSource(config.fixtures_path),
        baseline_selector or BaselineQueueSelector(),
        enhanced_selector or TeacherQueueSelector(),
        config,
    )
    after = verify_read_only_inputs(config)
    if before != after:
        raise ValidationError("read-only fixture or golden-seal hashes changed during the run")
    artifact = build_report(config, batch, candidates, comparison, before)
    if sink is not None:
        sink.write(artifact)
    else:
        JsonFileReportSink(config.output_path).write(artifact)
    return artifact


def _error_code(exc: Exception) -> str:
    """Map a known exception class to a structured CLI error code."""
    if isinstance(exc, (ConfigError, PydanticValidationError)):
        return "config_error"
    if isinstance(exc, ValidationError):
        return "validation_error"
    return "internal_error"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the teacher rehearsal CLI and write one internal report artifact.

    Returns:
        Exit code 0 on success, 2 on a known configuration or validation error.

    """
    default_settings = TeacherSettings()
    parser = argparse.ArgumentParser(
        description=(
            "Offline fixture stand-in: batch teacher transcripts, one "
            "disagreement signal, and a synthetic queue precision comparison."
        )
    )
    parser.add_argument("--fixtures", default=default_settings.fixtures_path)
    parser.add_argument("--output", default=default_settings.output_path)
    parser.add_argument("--queue-size", type=int, default=default_settings.queue_size)
    parser.add_argument("--signal-weight", type=float, default=default_settings.signal_weight)
    parser.add_argument("--teacher-model-id", default=default_settings.teacher_model_id)
    args = parser.parse_args(argv)

    try:
        settings = TeacherSettings(
            fixtures_path=args.fixtures,
            output_path=args.output,
            queue_size=args.queue_size,
            signal_weight=args.signal_weight,
            teacher_model_id=args.teacher_model_id,
        )
        artifact = run_teacher_rehearsal(settings)
    except (ConfigError, ValidationError, PydanticValidationError) as exc:
        print(json.dumps({"code": _error_code(exc), "message": str(exc)}), file=sys.stderr)
        return 2

    precision = artifact["precision"]
    print(f"Wrote teacher report: {settings.output_path}")
    print(
        f"Turns: {precision['turns']} / calls: {precision['calls']} / "
        f"queue cap: {precision['queue_size']}"
    )
    print(
        "Baseline synthetic ratio (overall / transcription): "
        f"{precision['baseline']['precision_overall']} / "
        f"{precision['baseline']['precision_transcription']}"
    )
    print(
        "Teacher-enhanced synthetic ratio (overall / transcription): "
        f"{precision['teacher_enhanced']['precision_overall']} / "
        f"{precision['teacher_enhanced']['precision_transcription']}"
    )
    print(
        "Synthetic deltas (overall / transcription): "
        f"{precision['delta_overall']} / {precision['delta_transcription']}"
    )
    print(
        "EXPLORATORY ONLY: depth_curve is post-hoc synthetic arithmetic with "
        "small n and overlapping intervals; no operating point was pre-declared."
    )
    print(
        f"{RUN_LABEL}: not a production precision claim or a precision-improvement "
        "claim; P4-04 is NOT DONE; no gate decision is made."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
