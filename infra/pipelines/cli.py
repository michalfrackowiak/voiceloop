"""CLI entry point for the fixture pipeline: ``python -m infra.pipelines.cli``.

The default mode deliberately refuses to run: production step adapters are
wired by the lead after P2-05/P2-06 land. ``--dry-run`` uses fake steps on the
synthetic fixtures and exercises the entire chain without ML deps or network.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from pydantic import ValidationError as PydanticValidationError

from voiceloop_common import ConfigError, ValidationError

from .chain import (
    PipelineStepError,
    PipelineSteps,
    RunRecord,
    TriggerNotSatisfied,
    build_fixture_steps,
    read_fixture_labels,
    run_chain,
    verified_example_ids,
)
from .settings import PipelineSettings
from .triggers import count_new_verified, decide_trigger


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m infra.pipelines.cli",
        description="Trigger and run the VoiceLoop dataset -> train -> evaluate chain.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full chain with fake steps on synthetic fixtures (no implicit override).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only evaluate and print the trigger decision; never run a chain (for cron).",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Explicit operator override; --dry-run alone never records a manual trigger.",
    )
    parser.add_argument(
        "--drift-alarm",
        action="store_true",
        help="Raise the drift-alarm trigger input for this invocation.",
    )
    parser.add_argument(
        "--fixtures", metavar="PATH", help="Override the fixture labels manifest path."
    )
    parser.add_argument(
        "--min-new-verified",
        type=int,
        metavar="N",
        help="Override VOICELOOP_PIPELINES_MIN_NEW_VERIFIED for this invocation.",
    )
    parser.add_argument(
        "--eval-set",
        action="append",
        choices=["general_capability"],
        metavar="NAME",
        help="Evaluation set name; the fake evaluator supports only general_capability.",
    )
    parser.add_argument(
        "--state",
        metavar="PATH",
        help=(
            "Run-record path used to count new verified IDs since the last run and store this run."
        ),
    )
    return parser


def _load_state(path: Path | None) -> tuple[frozenset[str], datetime | None]:
    """Load verified IDs and finish time from the previous run record."""
    if path is None or not path.exists():
        return frozenset(), None
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read pipeline state at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"pipeline state at {path} must be a JSON object")
    ids = payload.get("verified_ids", [])
    if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
        raise ConfigError(f"pipeline state at {path} has invalid 'verified_ids'")
    last: datetime | None = None
    finished_at = payload.get("finished_at")
    if isinstance(finished_at, str):
        try:
            last = datetime.fromisoformat(finished_at)
        except ValueError:
            last = None
    return frozenset(ids), last


def _write_state(path: Path, record: RunRecord) -> None:
    """Persist a successful run record for the next trigger check."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record.to_dict(), sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot write pipeline state at {path}: {exc}") from exc


def _print_record(record: RunRecord | None, out: TextIO) -> None:
    """Print each recorded step and the full run record as JSON."""
    if record is None:
        return
    for step in record.steps:
        print(
            f"STEP {step.name}: {json.dumps(step.summary, sort_keys=True)}",
            file=out,
        )
    print(f"RUN RECORD: {json.dumps(record.to_dict(), sort_keys=True)}", file=out)


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: PipelineSettings | None = None,
    steps: PipelineSteps | None = None,
    stdout: TextIO | None = None,
) -> int:
    """Run the CLI and return a process exit code (0 ok, 1 step failure, 2 config).

    Args:
        argv: Argument list; ``sys.argv[1:]`` when omitted.
        settings: Injected typed settings, primarily for tests.
        steps: Injected pipeline steps; otherwise ``--dry-run`` builds fakes.
        stdout: Injected text stream, primarily for tests.

    Returns:
        0 on success, no-op, or trigger-only check; 1 on a step failure;
        2 on configuration or input errors.

    """
    args = _build_parser().parse_args(argv)
    out = stdout if stdout is not None else sys.stdout
    try:
        updates: dict[str, Any] = {}
        if args.fixtures is not None:
            updates["fixtures_path"] = args.fixtures
        if args.min_new_verified is not None:
            updates["min_new_verified"] = args.min_new_verified
        if args.state is not None:
            updates["run_record_path"] = args.state
        if args.eval_set is not None:
            updates["eval_sets"] = tuple(args.eval_set)
        # Rebuild through the typed settings model so CLI overrides cannot
        # bypass field constraints such as min_new_verified >= 1.
        base = (settings or PipelineSettings()).model_dump()
        config = PipelineSettings.model_validate({**base, **updates})

        if steps is not None:
            pipeline_steps = steps
        elif args.dry_run:
            pipeline_steps = build_fixture_steps(config)
        else:
            print(
                "ERROR: no production steps are configured yet; "
                "use --dry-run for the fixture chain",
                file=out,
            )
            return 2

        state_path = Path(config.run_record_path) if config.run_record_path else None
        seen_ids, last_run_at = _load_state(state_path)
        labels = read_fixture_labels(config.fixtures_path)
        new_count = count_new_verified(verified_example_ids(labels), seen_ids)
        manual = args.manual or config.manual
        decision = decide_trigger(
            new_verified_count=new_count,
            settings=config,
            drift_alarm=args.drift_alarm or config.drift_alarm,
            manual=manual,
            last_run_at=last_run_at,
        )
        print("TRIGGER: " + json.dumps(decision.to_dict(), sort_keys=True), file=out)

        if args.check:
            return 0
        if not decision.triggered:
            print(
                "SKIPPED: no data-volume, drift, or manual trigger; time alone is not a trigger",
                file=out,
            )
            return 0

        try:
            record = run_chain(
                steps=pipeline_steps,
                trigger=decision,
                eval_sets=tuple(config.eval_sets),
            )
        except TriggerNotSatisfied as exc:
            _print_record(exc.record, out)
            print("SKIPPED: trigger policy not satisfied", file=out)
            return 0
        except PipelineStepError as exc:
            _print_record(exc.record, out)
            print(f"ERROR: {exc}", file=out)
            return 1

        _print_record(record, out)
        if state_path is not None:
            _write_state(state_path, record)
        return 0
    except (ConfigError, ValidationError, PydanticValidationError, OSError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=out)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
