"""One-command fake-only shadow rehearsal and comparison report.

Reads the committed synthetic fixture manifest read-only, mirrors its turn
references through the bounded fake sink, drains both fake model seams and
writes an internal comparison artifact. No live traffic, network, GPU or real
serving stack is involved, and no gate or alias is touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from voiceloop_common import ConfigError, ValidationError

from .mirror import (
    BoundedMirrorSink,
    CollectingAuditSink,
    FakeClock,
    FakeTurnSource,
    ShadowMirror,
    ShadowTurn,
)
from .report import FileReportSink, build_shadow_report
from .runner import ModelSource, ScriptedModelRunner, ShadowRunner
from .scoring import FakeOutcomeSource, OutcomeRecord, score_comparison
from .settings import ShadowSettings


def _load_fixture_labels(path: str) -> list[dict[str, Any]]:
    """Read the synthetic label manifest read-only and return its rows."""
    rows: list[dict[str, Any]] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ConfigError(f"fixture line {line_number} is not an object")
            rows.append(payload)
    if not rows:
        raise ConfigError("fixture manifest has no turns")
    return rows


def _scripted_slots(
    rows: Sequence[dict[str, Any]], *, degrade_champion: bool
) -> dict[tuple[str, int], dict[str, Any]]:
    """Build digest inputs from synthetic expected slots (raw data stays local)."""
    scripted: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        call_id = str(row["call_id"])
        turn_index = int(row["turn_index"])
        slots = row.get("expected_slots", {})
        fields = dict(slots) if isinstance(slots, dict) else {}
        if degrade_champion:
            suffix = call_id.rsplit("-", 1)[-1]
            if suffix.isdigit() and int(suffix) % 2 == 0:
                fields.pop("guests", None)
        scripted[(call_id, turn_index)] = fields
    return scripted


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fake shadow rehearsal and write one internal comparison report.

    Returns:
        0 on success; 2 with a structured ``{"code","message"}`` error on
        stderr for known configuration, fixture or validation failures.

    """
    base = ShadowSettings()
    parser = argparse.ArgumentParser(
        description=(
            "Mirror synthetic fixture turn references through a bounded fake "
            "sink, score candidate vs champion by call_id outcome and write "
            "a not_evaluated shadow comparison report."
        )
    )
    parser.add_argument("--fixtures", default=base.fixtures_path, help="Read-only label JSONL.")
    parser.add_argument("--output", default=base.output_path, help="Comparison artifact path.")
    parser.add_argument("--max-pending", type=int, default=base.max_pending)
    parser.add_argument("--budget-ms", type=float, default=base.mirror_budget_ms)
    args = parser.parse_args(argv)

    try:
        settings = ShadowSettings(
            fixtures_path=args.fixtures,
            output_path=args.output,
            max_pending=args.max_pending,
            mirror_budget_ms=args.budget_ms,
        )
        rows = _load_fixture_labels(settings.fixtures_path)
        clock = FakeClock()
        turns = tuple(
            ShadowTurn(
                call_id=str(row["call_id"]),
                turn_index=int(row["turn_index"]),
                audio_uri=str(row["audio_path"]),
                synthetic=bool(row.get("synthetic", True)),
                observed_at=clock.now(),
            )
            for row in rows
        )

        audit = CollectingAuditSink()
        sink = BoundedMirrorSink(clock=clock, capacity=settings.max_pending)
        mirror = ShadowMirror(settings, sink, clock, audit)
        turn_source = FakeTurnSource(turns)
        while (turn := turn_source.next_turn()) is not None:
            mirror.mirror(turn)

        candidate = ScriptedModelRunner(
            source=ModelSource.CANDIDATE,
            artifact_uri="fake://shadow/candidate-v1",
            model_id="fake-candidate",
            clock=clock,
            slots_by_turn=_scripted_slots(rows, degrade_champion=False),
        )
        champion = ScriptedModelRunner(
            source=ModelSource.CHAMPION,
            artifact_uri="fake://shadow/champion-v1",
            model_id="fake-champion",
            clock=clock,
            slots_by_turn=_scripted_slots(rows, degrade_champion=True),
        )
        runner = ShadowRunner(settings, sink, candidate, champion, clock, audit)
        runner.drain_once()

        outcomes: dict[str, OutcomeRecord] = {}
        for row in rows:
            status = row.get("expected_outcome")
            if not status:
                continue
            call_id = str(row["call_id"])
            slots = row.get("expected_slots", {})
            outcomes[call_id] = OutcomeRecord.from_payload(
                call_id,
                {
                    "status": status,
                    "final_record": slots if isinstance(slots, dict) else {},
                    "updated_at": "",
                },
                settings.ignored_outcome_fields(),
            )
        outcome_source = FakeOutcomeSource(outcomes)

        data = score_comparison(
            settings=settings,
            results=runner.results,
            outcome_source=outcome_source,
            mirror_stats=mirror.stats,
            candidate=candidate,
            champion=champion,
            synthetic=True,
        )
        report = build_shadow_report(data, settings.output_path)
        destination = FileReportSink().write(report, settings.output_path)
    except (ConfigError, ValidationError, PydanticValidationError, OSError, ValueError) as exc:
        code = (
            "config_error"
            if isinstance(exc, (ConfigError, PydanticValidationError, ValueError))
            else "validation_error"
            if isinstance(exc, ValidationError)
            else "io_error"
        )
        print(json.dumps({"code": code, "message": str(exc)}), file=sys.stderr)
        return 2

    counts = data.sample_counts
    candidate_metrics = data.metrics["candidate"]
    champion_metrics = data.metrics["champion"]
    print(f"Wrote shadow comparison artifact: {destination}")
    print(
        "Turns observed/mirrored/dropped: "
        f"{counts['turns_observed']}/{counts['turns_mirrored']}/{counts['turns_dropped']}"
    )
    print(
        f"Calls scored: {counts['calls_scored']} "
        f"(candidate {candidate_metrics['outcome_call_accuracy']:.4f} vs "
        f"champion {champion_metrics['outcome_call_accuracy']:.4f}; "
        f"delta {data.deltas['outcome_call_accuracy']:+.4f})"
    )
    print(
        "Primary outcome metric: outcome_call_accuracy "
        "(booking-field accuracy vs final record, joined by call_id)"
    )
    print("SYNTHETIC/FIXTURE STAND-IN: not live traffic and not a promotion decision.")
    print("promotion_decision: not_evaluated; no alias writes, gate edits or threshold changes.")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via python -m
    raise SystemExit(main())
