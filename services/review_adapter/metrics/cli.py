"""One-command CLI for the fixture-only P2-04 review metrics report."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from pydantic import ValidationError as PydanticValidationError

from voiceloop_common import ValidationError

from .agreement import SealedGoldenCandidateSource
from .events import (
    CandidatePairSource,
    FixedClock,
    FixtureReviewResultSource,
    JsonFileReportSink,
    ResultPairSource,
    ReviewResultSource,
    SequenceClock,
)
from .report import MetricsInputs, generate_metrics_report
from .settings import ReviewMetricsSettings


def build_inputs(settings: ReviewMetricsSettings) -> MetricsInputs:
    """Wire the committed fixture-only fakes and injectable seams for one run.

    Args:
        settings: Typed review metrics settings.

    Returns:
        Inputs with a fixture review source, explicit pairs, read-only golden
        candidates, a fake clock and a JSON file sink.

    """
    generation_clock = SequenceClock(
        settings.start_at, step_seconds=settings.scripted_duration_seconds
    )
    results: ReviewResultSource = FixtureReviewResultSource(settings, generation_clock)
    pairs: CandidatePairSource = ResultPairSource(results)
    return MetricsInputs(
        settings=settings,
        results=results,
        pairs=pairs,
        golden=SealedGoldenCandidateSource(settings),
        clock=FixedClock(settings.start_at),
        sink=JsonFileReportSink(settings.output_path),
    )


def _error_code(exc: Exception) -> str:
    """Map a known exception class to a structured CLI error code."""
    if isinstance(exc, (ValidationError, PydanticValidationError, ValueError, OSError)):
        return "validation_error"
    return "internal_error"


def main(argv: Sequence[str] | None = None) -> int:
    """Generate one review metrics report from committed fixture reviews.

    Returns:
        ``0`` on success; ``2`` on a known validation, input or output failure
        with ``{"code", "message"}`` printed to stderr. No UI decision,
        promotion, alias flip or threshold change is performed.

    """
    base = ReviewMetricsSettings()
    parser = argparse.ArgumentParser(
        description=(
            "Generate a SYNTHETIC/FIXTURE STAND-IN review UX metrics report "
            "from P2-03 fake review results; input to P4-03 go/no-go only."
        )
    )
    parser.add_argument("--output", default=base.output_path, help="Report destination path.")
    parser.add_argument("--fixtures", default=base.fixtures_path, help="Read-only fixture labels.")
    parser.add_argument(
        "--golden-artifact-dir",
        default=base.golden_artifact_dir,
        help="Read-only sealed golden stand-in directory.",
    )
    parser.add_argument("--golden-version", default=base.golden_version)
    parser.add_argument("--start-at", default=base.start_at, help="Injected fake start timestamp.")
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=base.scripted_duration_seconds,
        help="Committed scripted fake review duration in seconds.",
    )
    args = parser.parse_args(argv)

    try:
        settings = ReviewMetricsSettings(
            fixtures_path=args.fixtures,
            golden_artifact_dir=args.golden_artifact_dir,
            golden_version=args.golden_version,
            output_path=args.output,
            start_at=args.start_at,
            scripted_duration_seconds=args.duration_seconds,
        )
        report, destination = generate_metrics_report(build_inputs(settings))
    except (ValidationError, PydanticValidationError, ValueError, OSError) as exc:
        print(
            json.dumps({"code": _error_code(exc), "message": str(exc)}),
            file=sys.stderr,
        )
        return 2

    per_item = report["sections"]["per_item"]
    print(f"Wrote review metrics report: {destination}")
    print(f"Fake review results measured: {per_item['time_summary']['count']}")
    print(f"Routing labels: {per_item['routing_label_distribution']}")
    print("SYNTHETIC/FIXTURE STAND-IN: no human reviews, UI, network or promotion.")
    print("Input to P4-03 go/no-go only; no custom-UI decision made.")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via python -m
    raise SystemExit(main())
