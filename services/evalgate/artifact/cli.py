"""CLI for exact-artifact STT safety reports and injected failure detection."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from services.evalgate.gate import GateError
from services.evalgate.metrics import EvaluationError
from voiceloop_common import ConfigError, ValidationError

from .artifact import ReadOnlyArtifactStore, create_fake_artifact
from .report import EXTENSION_KEY, evaluate_artifact, validate_stt_report, write_stt_report
from .settings import ArtifactBehaviour, ArtifactSettings

ARTIFACT_BEHAVIOURS: tuple[ArtifactBehaviour, ...] = (
    "healthy",
    "empty",
    "truncated",
    "hallucinating",
    "forgetting",
)


def _error_code(exc: Exception) -> str:
    """Map a known exception to a stable structured error code."""
    if isinstance(exc, ConfigError):
        return "config_error"
    if isinstance(exc, ValidationError):
        return "validation_error"
    if isinstance(exc, EvaluationError):
        return "evaluation_error"
    if isinstance(exc, GateError):
        return "gate_error"
    return "internal_error"


def _csv(value: str) -> list[str]:
    """Parse a comma-separated CLI value into trimmed non-empty entries."""
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    """Run artifact probes, write a validated report and fail closed on errors.

    Returns 0 when probes pass, and 2 with a structured error on stderr when the
    artifact fails a probe or evaluation cannot be completed. The P2-09 gate is
    always evaluated read-only; no promotion or alias flip is performed.
    """
    base = ArtifactSettings()
    parser = argparse.ArgumentParser(description="Evaluate a fake deployed STT artifact safely.")
    parser.add_argument("--fixtures", default=base.fixtures_path, help="Read-only labelled turns.")
    parser.add_argument("--output", default=base.output_path, help="STT report output path.")
    parser.add_argument("--artifact-store", default=base.artifact_store_dir)
    parser.add_argument("--checkpoint-uri", default=base.checkpoint_uri)
    parser.add_argument("--behaviour", choices=ARTIFACT_BEHAVIOURS, default=base.artifact_behaviour)
    parser.add_argument("--runtime-label", default=base.runtime_label)
    parser.add_argument("--hotwords", default=",".join(base.hotwords))
    parser.add_argument("--initial-prompt", default=base.initial_prompt)
    parser.add_argument(
        "--condition-on-previous-text",
        action=argparse.BooleanOptionalAction,
        default=base.condition_on_previous_text,
    )
    parser.add_argument("--language", default=base.language)
    parser.add_argument("--served-languages", default=base.served_languages_csv)
    parser.add_argument("--level", type=float, default=base.bootstrap_level)
    parser.add_argument("--iterations", type=int, default=base.bootstrap_iterations)
    parser.add_argument("--seed", type=int, default=base.bootstrap_seed)
    args = parser.parse_args(argv)

    overrides: dict[str, Any] = {
        "fixtures_path": args.fixtures,
        "output_path": args.output,
        "artifact_store_dir": args.artifact_store,
        "checkpoint_uri": args.checkpoint_uri,
        "artifact_behaviour": args.behaviour,
        "runtime_label": args.runtime_label,
        "hotwords": _csv(args.hotwords),
        "initial_prompt": args.initial_prompt,
        "condition_on_previous_text": args.condition_on_previous_text,
        "language": args.language,
        "served_languages_csv": args.served_languages,
        "bootstrap_level": args.level,
        "bootstrap_iterations": args.iterations,
        "bootstrap_seed": args.seed,
    }
    settings = base.model_copy(update=overrides)

    try:
        artifact = create_fake_artifact(settings.artifact_behaviour, settings.runtime_label)
        store = ReadOnlyArtifactStore(settings.artifact_store_dir)
        store.put(artifact)
        evaluation = evaluate_artifact(settings, artifact)
        validate_stt_report(evaluation.report)
        destination = write_stt_report(evaluation.report, settings.output_path)
    except (ConfigError, ValidationError, EvaluationError, GateError) as exc:
        print(json.dumps({"code": _error_code(exc), "message": str(exc)}), file=sys.stderr)
        return 2

    report = evaluation.report
    extension = report["decoding_settings"][EXTENSION_KEY]
    print(f"Wrote STT artifact report: {destination}")
    print(f"Artifact SHA-256: {artifact.sha256}")
    print(f"Decoding settings SHA-256: {extension['decoding']['sha256']}")
    print(f"Silence hallucination rate: {report['metrics']['silence_hallucination_rate']:.3f}")
    print(
        "Multilingual forgetting rate: "
        f"{report['metrics']['multilingual_forgetting_rate']:.3f} "
        f"({extension['multilingual_forgetting']['status']})"
    )
    print("Primary metric: e2e_booking_accuracy (end-to-end booking accuracy from audio)")
    print("SYNTHETIC/FIXTURE STAND-IN: no real runtime, GPU, network or human labels used.")
    print("WER alone never justifies promotion (context.md section 2 invariant 4).")

    if evaluation.probe_suite.status == "fail":
        print(
            json.dumps(
                {
                    "code": "artifact_probe_failed",
                    "message": "injected artifact behaviour failed closed",
                    "artifact_sha256": artifact.sha256,
                    **extension["probes"],
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via python -m
    raise SystemExit(main())
