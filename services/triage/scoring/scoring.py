"""Turn scoring orchestration, per-turn storage and the fixture CLI."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .explanation import (
    ExplanationStore,
    FiredSignal,
    InMemoryExplanationStore,
    ScoreExplanation,
)
from .mismatch import detect_mismatches
from .repair import build_repair_context, detect_repair, load_prior_turns
from .settings import TriageScoringSettings
from .signals import (
    FakeTeacherModel,
    Observation,
    TeacherModel,
    extract_signals,
    teacher_observation,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIXTURES_PATH = "tests/fixtures/turns.jsonl"


class ScoringError(ValueError):
    """Raised when a turn event cannot be scored."""


@dataclass(frozen=True)
class FixtureScore:
    """One read-only fixture label paired with its explanation."""

    label: dict[str, Any]
    explanation: ScoreExplanation


def _resolve(path: str | Path) -> Path:
    """Resolve a repo-relative path against the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _fixture_prior_loader(event: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    """Load prior hypotheses for a fixture event from its context JSONL."""
    return load_prior_turns(event, root=REPO_ROOT)


class ScoringEngine:
    """Score turn events deterministically and persist their explanations."""

    def __init__(
        self,
        settings: TriageScoringSettings | None = None,
        *,
        store: ExplanationStore | None = None,
        teacher: TeacherModel | None = None,
        prior_turn_loader: Callable[[Mapping[str, Any]], Sequence[Mapping[str, Any]]] | None = None,
    ) -> None:
        """Create an engine; all collaborators have in-memory/default fakes.

        Args:
            settings: Typed weights and thresholds, defaulting to environment.
            store: Explanation persistence seam; defaults to an in-memory store.
            teacher: Optional offline teacher adapter; disabled unless configured.
            prior_turn_loader: Optional resolver for ``prior_turns_ref`` records.

        """
        self.settings = settings if settings is not None else TriageScoringSettings()
        self.store = store if store is not None else InMemoryExplanationStore()
        self.teacher = teacher if teacher is not None else FakeTeacherModel()
        self.prior_turn_loader = prior_turn_loader

    def score(
        self,
        event: Mapping[str, Any],
        *,
        prior_turns: Sequence[Mapping[str, Any]] | None = None,
        persist: bool = True,
    ) -> ScoreExplanation:
        """Score one schema-valid turn event and explain every fired signal.

        Args:
            event: Turn event dictionary conforming to ``turn_event`` 1.0.0.
            prior_turns: Optional decoded dialogue context; takes precedence over
                the configured prior-turn loader.
            persist: Whether to upsert the explanation into the configured store.

        Returns:
            An immutable explanation with score in [0, 1].

        Raises:
            ScoringError: If required identifiers are missing or context cannot load.

        """
        event_id = event.get("event_id")
        call_id = event.get("call_id")
        turn_index = event.get("turn_index")
        if not isinstance(event_id, str) or not event_id:
            raise ScoringError("turn event requires a non-empty event_id")
        if not isinstance(call_id, str) or not call_id:
            raise ScoringError("turn event requires a non-empty call_id")
        if isinstance(turn_index, bool) or not isinstance(turn_index, int) or turn_index < 0:
            raise ScoringError("turn event requires a non-negative integer turn_index")

        context_turns = prior_turns
        if context_turns is None and self.prior_turn_loader is not None:
            try:
                context_turns = self.prior_turn_loader(event)
            except (OSError, ValueError) as exc:
                raise ScoringError(f"cannot load dialogue context: {exc}") from exc

        context = build_repair_context(event, context_turns)
        observations: list[Observation] = []
        observations.extend(extract_signals(event, self.settings))
        observations.extend(detect_repair(event, self.settings, context))
        observations.extend(detect_mismatches(event, self.settings))
        observations.extend(teacher_observation(event, self.settings, self.teacher))

        raw_score = sum(observation.contribution for observation in observations)
        score = round(1.0 - math.exp(-raw_score), 6)
        score = min(1.0, max(0.0, score))
        signals = tuple(
            FiredSignal(
                name=observation.name,
                weight=observation.weight,
                severity=observation.severity,
                reason=observation.reason,
            )
            for observation in observations
        )
        explanation = ScoreExplanation(
            event_id=event_id,
            call_id=call_id,
            turn_index=turn_index,
            score=score,
            signals=signals,
        )
        if persist:
            self.store.put(explanation)
        return explanation


def score_turn(
    event: Mapping[str, Any],
    *,
    settings: TriageScoringSettings | None = None,
    store: ExplanationStore | None = None,
    teacher: TeacherModel | None = None,
    prior_turns: Sequence[Mapping[str, Any]] | None = None,
    persist: bool = True,
) -> ScoreExplanation:
    """Score a single turn with a fresh engine and default fakes."""
    engine = ScoringEngine(settings=settings, store=store, teacher=teacher)
    return engine.score(event, prior_turns=prior_turns, persist=persist)


def load_fixture_scores(
    turns_path: str | Path = DEFAULT_FIXTURES_PATH,
    *,
    settings: TriageScoringSettings | None = None,
    store: ExplanationStore | None = None,
    teacher: TeacherModel | None = None,
    prior_turn_loader: Callable[[Mapping[str, Any]], Sequence[Mapping[str, Any]]] | None = None,
    persist: bool = True,
) -> list[FixtureScore]:
    """Score read-only fixture turns without using their failure labels.

    The labels manifest is used only to locate events and to attach metadata to
    the returned rows. ``expected_slots``, ``expected_transcript`` and
    ``failure_type`` are never passed into scoring.

    Args:
        turns_path: JSON Lines labels manifest, repo-relative by default.
        settings: Typed weights and thresholds.
        store: Explanation store receiving every score when ``persist`` is true.
        teacher: Optional offline teacher adapter.
        prior_turn_loader: Resolver for dialogue-context references.
        persist: Whether explanations are written to ``store``.

    Returns:
        One :class:`FixtureScore` per manifest row, in file order.

    Raises:
        ScoringError: If the manifest, an event file or identifiers are invalid.

    """
    manifest = _resolve(turns_path)
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ScoringError(f"cannot read fixtures: {manifest}") from exc

    engine = ScoringEngine(
        settings=settings, store=store, teacher=teacher, prior_turn_loader=prior_turn_loader
    )
    results: list[FixtureScore] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            label = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ScoringError(f"invalid fixture JSON at line {line_number}") from exc
        if not isinstance(label, dict) or not isinstance(label.get("event_path"), str):
            raise ScoringError(f"fixture line {line_number} needs an event_path")
        event_path = _resolve(label["event_path"])
        try:
            event = json.loads(event_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ScoringError(f"cannot read fixture event: {event_path}") from exc
        if not isinstance(event, dict):
            raise ScoringError(f"fixture event must be an object: {event_path}")
        for key in ("call_id", "turn_index"):
            if event.get(key) != label.get(key):
                raise ScoringError(f"fixture/event mismatch for {key} at line {line_number}")
        explanation = engine.score(event)
        results.append(FixtureScore(label=label, explanation=explanation))
    return results


def main(argv: Sequence[str] | None = None) -> int:
    """Run the scoring CLI over the synthetic fixtures.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code (0 on success).

    """
    parser = argparse.ArgumentParser(description="Score synthetic turns for triage.")
    parser.add_argument(
        "--fixtures",
        default=DEFAULT_FIXTURES_PATH,
        help="Path to the read-only labels manifest (JSON Lines).",
    )
    parser.add_argument(
        "--output", default=None, help="Optional JSON output path; stdout by default."
    )
    args = parser.parse_args(argv)

    store = InMemoryExplanationStore()
    rows = load_fixture_scores(args.fixtures, store=store, prior_turn_loader=_fixture_prior_loader)
    by_failure_type: dict[str, dict[str, float]] = {}
    for row in rows:
        failure_type = str(row.label.get("failure_type", "unknown"))
        bucket = by_failure_type.setdefault(failure_type, {"turns": 0.0, "mean_score": 0.0})
        bucket["turns"] += 1.0
        bucket["mean_score"] += row.explanation.score
    for bucket in by_failure_type.values():
        if bucket["turns"]:
            bucket["mean_score"] = round(bucket["mean_score"] / bucket["turns"], 6)

    report: dict[str, Any] = {
        "turns": len(rows),
        "by_failure_type": by_failure_type,
        "explanations": [row.explanation.to_dict() for row in rows],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
