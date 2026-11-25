"""One-command CLI for synthetic base-model candidate comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from services.evalgate.metrics import EvaluationError
from voiceloop_common import ConfigError, ValidationError

from .compare import compare_candidates
from .report import build_comparison_artifact, write_comparison_artifact
from .settings import BenchmarkSettings


def _error_code(exc: Exception) -> str:
    """Map a known exception class to a structured CLI error code."""
    if isinstance(exc, ConfigError):
        return "config_error"
    if isinstance(exc, ValidationError):
        return "validation_error"
    if isinstance(exc, EvaluationError):
        return "evaluation_error"
    return "internal_error"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark CLI and write one internal comparison artifact.

    Returns 0 on success, and 2 on a known configuration, validation or
    evaluation failure with a structured error on stderr.
    """
    base = BenchmarkSettings()
    parser = argparse.ArgumentParser(
        description=(
            "Compare two deterministic fake candidates on synthetic fixtures "
            "and the sealed v1 golden stand-in; deltas are candidate A - B."
        )
    )
    parser.add_argument(
        "--candidates",
        default=base.candidates_csv,
        help="Comma-separated candidate pair A,B (default: identity,degraded).",
    )
    parser.add_argument("--output", default=base.output_path, help="Comparison artifact path.")
    parser.add_argument("--fixtures", default=base.fixtures_path, help="Labelled turns JSONL path.")
    parser.add_argument(
        "--golden-artifact-dir",
        default=base.golden_artifact_dir,
        help="Sealed golden stand-in artifact directory (read-only).",
    )
    parser.add_argument("--golden-version", default=base.golden_version)
    parser.add_argument("--level", type=float, default=base.bootstrap_level)
    parser.add_argument("--iterations", type=int, default=base.bootstrap_iterations)
    parser.add_argument("--seed", type=int, default=base.bootstrap_seed)
    args = parser.parse_args(argv)

    try:
        settings = BenchmarkSettings(
            candidates_csv=args.candidates,
            output_path=args.output,
            fixtures_path=args.fixtures,
            golden_artifact_dir=args.golden_artifact_dir,
            golden_version=args.golden_version,
            bootstrap_level=args.level,
            bootstrap_iterations=args.iterations,
            bootstrap_seed=args.seed,
        )
        data = compare_candidates(settings)
        artifact = build_comparison_artifact(data)
        destination = write_comparison_artifact(artifact, settings.output_path)
    except (ConfigError, ValidationError, EvaluationError) as exc:
        print(
            json.dumps({"code": _error_code(exc), "message": str(exc)}),
            file=sys.stderr,
        )
        return 2

    primary = artifact["metric_semantics"]["primary_metric"]
    print(f"Wrote comparison artifact: {destination}")
    print(f"Candidates compared (A - B): {data.candidates[0]} - {data.candidates[1]}")
    print(f"Primary metric: {primary} (end-to-end booking-field accuracy from audio)")
    print("SYNTHETIC/FIXTURE STAND-IN: not a production benchmark; no public leaderboards used.")
    print("WER alone never justifies promotion (context.md section 2 invariant 4).")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via python -m
    raise SystemExit(main())
