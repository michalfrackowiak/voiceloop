"""One-command CLI for the fixture-only P3-09 golden maintenance rehearsal.

``propose`` builds a separate synthetic candidate set, runs two scripted
reviewers, audits both reviews and seals a new ``candidate.vN`` version.
``verify`` checks a version and its predecessor chain read-only. ``drift``
writes a descriptive-only fixture replay comparison. No command changes the
sealed golden artifacts, performs a promotion or changes a threshold.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from voiceloop_common import ValidationError

from .candidates import build_candidate_set
from .drift import build_drift_report, write_drift_report
from .review import ReviewAuditLog, record_candidate_build, run_scripted_review
from .settings import (
    ARTIFACT_LABEL,
    GoldenMaintenanceSettings,
    assert_allowed_actor,
    assert_outside_sealed,
    assert_safe_text,
    canonical_json,
    contains_pii_pattern,
    resolve_path,
)
from .versioning import (
    SealedCandidateArtifact,
    create_candidate_version,
    verify_candidate_version,
)


def _error_code(exc: Exception) -> str:
    """Map a known exception class to a structured CLI error code."""
    if isinstance(exc, (ValidationError, PydanticValidationError, ValueError, OSError)):
        return "validation_error"
    return "internal_error"


def _safe_message(exc: Exception) -> str:
    """Return a diagnostic message that cannot echo PII-like input."""
    message = str(exc)
    if contains_pii_pattern(message):
        return "rejected input that failed PII screening"
    return message


def _propose(args: argparse.Namespace, base: GoldenMaintenanceSettings) -> dict[str, Any]:
    """Run the audited scripted extension rehearsal and seal a candidate version."""
    settings = GoldenMaintenanceSettings(
        fixtures_path=args.fixtures,
        sealed_artifact_dir=args.sealed_artifact_dir,
        output_dir=args.output_dir,
        candidate_version=args.version,
        candidate_calls=args.candidate_calls,
        candidate_seed=args.candidate_seed,
        task_id=args.task_id,
    )
    assert_outside_sealed(args.output_dir, settings)
    assert_allowed_actor(args.actor, settings)
    assert_safe_text(args.purpose, "purpose")
    candidate = build_candidate_set(settings)
    audit = ReviewAuditLog(resolve_path(args.output_dir) / "review_audit.jsonl")
    record_candidate_build(audit, candidate, settings, actor_id=args.actor, purpose=args.purpose)
    bundle = run_scripted_review(candidate, settings, audit)
    artifact: SealedCandidateArtifact = create_candidate_version(
        args.output_dir,
        candidate,
        bundle,
        settings=settings,
        actor_id=args.actor,
        purpose=args.purpose,
        task_id=args.task_id,
        predecessor_version=args.predecessor,
    )
    return {
        "ok": True,
        "version": artifact.manifest.version,
        "content_hash": artifact.manifest.content_hash,
        "candidate_count": artifact.manifest.candidate_count,
        "sealed_content_hash": artifact.manifest.sealed_content_hash,
        "predecessor_content_hash": artifact.manifest.predecessor_content_hash,
        "label": artifact.manifest.label,
    }


def _verify(
    args: argparse.Namespace, base: GoldenMaintenanceSettings
) -> tuple[int, dict[str, Any]]:
    """Verify one candidate version without appending to any audit log."""
    settings = GoldenMaintenanceSettings(
        fixtures_path=args.fixtures,
        sealed_artifact_dir=args.sealed_artifact_dir,
        output_dir=args.output_dir,
        candidate_version=args.version,
    )
    report = verify_candidate_version(args.output_dir, args.version, settings)
    result: dict[str, Any] = {
        "ok": report.ok,
        "errors": list(report.errors),
        "version": args.version,
        "label": ARTIFACT_LABEL,
    }
    return (0 if report.ok else 2), result


def _drift(args: argparse.Namespace, base: GoldenMaintenanceSettings) -> dict[str, Any]:
    """Generate a descriptive-only drift report from synthetic fixture replay."""
    settings = GoldenMaintenanceSettings(
        fixtures_path=args.fixtures,
        sealed_artifact_dir=args.sealed_artifact_dir,
        drift_calls=args.drift_calls,
        drift_seed=args.drift_seed,
    )
    report = build_drift_report(settings)
    destination = write_drift_report(report, args.output, settings) if args.output else None
    return {
        "ok": True,
        "label": report["label"],
        "sample_counts": report["sample_counts"],
        "interpretation": report["interpretation"],
        "human_review_gate": report["human_review_gate"],
        "output": str(destination) if destination else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Parse commands and print canonical JSON results.

    Returns:
        ``0`` on success, ``2`` on validation or verification failure. Structured
        errors go to stderr; no command promotes a model or edits sealed files.

    """
    base = GoldenMaintenanceSettings()
    parser = argparse.ArgumentParser(
        description=(
            "SYNTHETIC/FIXTURE STAND-IN - NOT A HUMAN-REVIEWED GOLDEN SET. "
            "Audited P3-09 candidate rehearsal and descriptive drift check."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    propose = sub.add_parser("propose", help="Build, review and seal a candidate version.")
    propose.add_argument("--output-dir", default=base.output_dir)
    propose.add_argument("--version", default=base.candidate_version)
    propose.add_argument("--fixtures", default=base.fixtures_path)
    propose.add_argument("--sealed-artifact-dir", default=base.sealed_artifact_dir)
    propose.add_argument("--candidate-calls", type=int, default=base.candidate_calls)
    propose.add_argument("--candidate-seed", type=int, default=base.candidate_seed)
    propose.add_argument("--predecessor", default=None, help="Previous candidate.vN, if any.")
    propose.add_argument("--actor", required=True, help="Pseudonymous operator ID.")
    propose.add_argument("--purpose", required=True, help="PII-screened reason for the change.")
    propose.add_argument("--task-id", default=base.task_id)

    verify = sub.add_parser("verify", help="Verify a candidate version read-only.")
    verify.add_argument("--output-dir", default=base.output_dir)
    verify.add_argument("--version", default=base.candidate_version)
    verify.add_argument("--fixtures", default=base.fixtures_path)
    verify.add_argument("--sealed-artifact-dir", default=base.sealed_artifact_dir)

    drift = sub.add_parser("drift", help="Descriptive fixture-replay drift comparison.")
    drift.add_argument("--output", default=None, help="Optional JSON report destination.")
    drift.add_argument("--fixtures", default=base.fixtures_path)
    drift.add_argument("--sealed-artifact-dir", default=base.sealed_artifact_dir)
    drift.add_argument("--drift-calls", type=int, default=base.drift_calls)
    drift.add_argument("--drift-seed", type=int, default=base.drift_seed)

    args = parser.parse_args(argv)
    try:
        if args.command == "propose":
            result = _propose(args, base)
            code = 0
        elif args.command == "verify":
            code, result = _verify(args, base)
        else:
            result = _drift(args, base)
            code = 0
    except (ValidationError, PydanticValidationError, ValueError, OSError) as exc:
        print(
            canonical_json({"code": _error_code(exc), "message": _safe_message(exc)}),
            file=sys.stderr,
        )
        return 2
    except Exception:  # pragma: no cover - defensive boundary for untrusted input
        error = canonical_json({"code": "internal_error", "message": "operation rejected"})
        print(error, file=sys.stderr)
        return 2
    print(canonical_json(result))
    return code


if __name__ == "__main__":  # pragma: no cover - exercised via python -m
    raise SystemExit(main())
