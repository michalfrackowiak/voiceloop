"""Rule orchestration and a read-only golden regression check.

The check uses the committed P1-07 harness primitives (``evaluate_rows``,
``predict``, ``evaluate_turn``, ``aggregate_metrics``) on the sealed v1
synthetic golden stand-in. The sealed artifacts and fixtures are read-only;
all rule application happens in memory. Every reported improvement is
SYNTHETIC/FIXTURE STAND-IN, not evidence from a live STT runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from services.datasets.golden.seal import load_labels, verify_seal
from services.evalgate.harness import evaluate_rows, load_event, load_fixture_rows, predict
from services.evalgate.metrics import Prediction, TurnMetrics, aggregate_metrics, evaluate_turn

from .aliases import AliasTable, derive_alias_table
from .boosting import BoostConfig, SttRuntime, apply_boosting, generate_boosting_config
from .normalise import normalise_slots, normalise_transcript
from .settings import VocabSettings

SYNTHETIC_WARNING = (
    "SYNTHETIC/FIXTURE STAND-IN: rules are applied in memory by stdlib fakes; "
    "no real STT runtime, GPU, parser model or human-labelled production golden "
    "set was used, and no phrase boosting was applied in a live decoder."
)
_METRIC_TOLERANCE = 1e-12


@dataclass(frozen=True)
class RuleOutcome:
    """Transcript, slots and the names of rules that changed them."""

    transcript: str
    slots: dict[str, Any]
    rules: tuple[str, ...] = field(default_factory=tuple)

    @property
    def changed(self) -> bool:
        """Return whether any rule fired."""
        return bool(self.rules)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable outcome."""
        return {"transcript": self.transcript, "slots": self.slots, "rules": list(self.rules)}


class VocabRuleSet:
    """Deterministic, offline alias + boosting + normalisation rule pipeline."""

    def __init__(
        self,
        settings: VocabSettings | None = None,
        *,
        alias_table: AliasTable | None = None,
        boost_config: BoostConfig | None = None,
        runtime: SttRuntime | None = None,
    ) -> None:
        """Build a pipeline with optional injected aliases, config and STT seam."""
        self.settings = settings or VocabSettings()
        self.alias_table = alias_table or derive_alias_table(
            self.settings.fixtures_path,
            name_overrides=self.settings.name_alias_overrides,
            room_overrides=self.settings.room_alias_overrides,
        )
        self.boost_config = boost_config or generate_boosting_config(self.settings)
        self.runtime = runtime

    def apply(
        self,
        transcript: str,
        slots: Mapping[str, Any],
        *,
        words: Sequence[Mapping[str, Any]] = (),
        use_boosting: bool = True,
    ) -> RuleOutcome:
        """Apply all enabled rules without mutating the inputs.

        Args:
            transcript: STT hypothesis text.
            slots: Slots parsed from that hypothesis.
            words: Word-level confidences used by the synthetic boosting seam.
            use_boosting: Allows unit tests to isolate the other rule types.

        Returns:
            The transformed transcript/slots and fired rule names.

        """
        fired: list[str] = []
        current_text = transcript
        current_slots = dict(slots)

        if self.settings.enable_aliases:
            aliased_text = self.alias_table.apply_text(current_text, kinds=("name",))
            if aliased_text != current_text:
                fired.append("alias_name")
                current_text = aliased_text
            aliased_slots = self.alias_table.canonicalise_slots(current_slots)
            if aliased_slots.get("guest_name") != current_slots.get("guest_name"):
                fired.append("alias_guest_name")
            if aliased_slots.get("room_type") != current_slots.get("room_type"):
                fired.append("alias_room_type")
            current_slots = aliased_slots

        if self.settings.enable_boosting and use_boosting:
            boosted = apply_boosting(current_text, words, self.boost_config, self.runtime)
            if boosted != current_text:
                fired.append("phrase_boost_synthetic")
                current_text = boosted

        if self.settings.enable_normalisation:
            if self.settings.rewrite_transcript:
                canonical_text = normalise_transcript(current_text)
                if canonical_text != current_text:
                    fired.append("normalise_transcript")
                    current_text = canonical_text
            normalised_slots, number_rules = normalise_slots(current_text, current_slots)
            fired.extend(number_rules)
            current_slots = normalised_slots

        return RuleOutcome(transcript=current_text, slots=current_slots, rules=tuple(fired))


def _artifact_hashes(directory: str | Path, version: str) -> dict[str, str]:
    """Hash all sealed artifact bytes before and after a regression run."""
    base = Path(directory)
    names = (f"{version}.json", f"{version}.labels.json", f"{version}.seal.json", "audit.jsonl")
    return {name: hashlib.sha256((base / name).read_bytes()).hexdigest() for name in names}


def _is_higher_better(metric: str) -> bool:
    """Return the improvement direction for one aggregate metric name."""
    if metric == "wer" or metric.startswith("entity_error"):
        return False
    return True


def _build_golden_rows(settings: VocabSettings) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build harness rows from sealed labels plus read-only fixture events."""
    sealed = verify_seal(settings.golden_artifact_dir, settings.golden_version)
    sealed.assert_ok()
    manifest = sealed.manifest
    if manifest is None:
        raise AssertionError("verified seal has no manifest")
    if manifest.content_hash != settings.expected_golden_hash:
        raise AssertionError("sealed golden content hash changed; refusing to evaluate")
    labels = load_labels(
        Path(settings.golden_artifact_dir) / manifest.labels_path,
        expected_version=settings.golden_version,
    )
    fixture_rows = load_fixture_rows(settings.fixtures_path)
    by_source = {(str(row["call_id"]), int(row["turn_index"])): row for row in fixture_rows}
    rows: list[dict[str, Any]] = []
    for label in labels:
        key = (label.call_id, label.turn_index)
        if key not in by_source:
            raise AssertionError(f"sealed label {key} has no read-only fixture event")
        fixture = by_source[key]
        rows.append(
            {
                "call_id": label.call_id,
                "turn_index": label.turn_index,
                "case_id": fixture.get("case_id"),
                "event_path": fixture["event_path"],
                "expected_transcript": label.corrected_transcript,
                "expected_slots": label.slots,
            }
        )
    return rows, manifest.to_dict()


def _score_rows(
    rows: Sequence[Mapping[str, Any]],
    ruleset: VocabRuleSet,
    *,
    apply_rules: bool,
) -> tuple[list[TurnMetrics], list[RuleOutcome]]:
    """Score rows with harness primitives, optionally applying rules in memory."""
    if not apply_rules:
        _, challenger = evaluate_rows(rows, "identity", "identity")
        return challenger, []

    turns: list[TurnMetrics] = []
    outcomes: list[RuleOutcome] = []
    for row in rows:
        event = load_event(str(row["event_path"]))
        base = predict("identity", row, event)
        stt = event.get("stt") or {}
        words = stt.get("words", []) if isinstance(stt, Mapping) else []
        if not isinstance(words, Sequence) or isinstance(words, (str, bytes)):
            words = []
        outcome = ruleset.apply(base.transcript, base.slots, words=words)
        prediction = Prediction(transcript=outcome.transcript, slots=outcome.slots)
        turns.append(
            evaluate_turn(
                str(row["expected_transcript"]),
                dict(row["expected_slots"]),
                prediction,
            )
        )
        outcomes.append(outcome)
    return turns, outcomes


def check_golden_regression(
    settings: VocabSettings | None = None,
    ruleset: VocabRuleSet | None = None,
) -> dict[str, Any]:
    """Apply rules in memory and assert no golden metric regresses.

    The comparison uses the read-only sealed v1 synthetic stand-in as gold and
    the read-only fixture events as identity-fake STT/parser predictions. Every
    numeric metric (including diagnostics) is compared in its improvement
    direction; artifact bytes are hash-checked before and after.
    """
    settings = settings or VocabSettings()
    ruleset = ruleset or VocabRuleSet(settings)
    hashes_before = _artifact_hashes(settings.golden_artifact_dir, settings.golden_version)
    golden_rows, manifest = _build_golden_rows(settings)
    fixture_rows = load_fixture_rows(settings.fixtures_path)

    before_turns, _ = _score_rows(golden_rows, ruleset, apply_rules=False)
    after_turns, outcomes = _score_rows(golden_rows, ruleset, apply_rules=True)
    before_metrics = aggregate_metrics(before_turns)
    after_metrics = aggregate_metrics(after_turns)
    deltas = {name: after_metrics[name] - before_metrics[name] for name in before_metrics}
    regressions = sorted(
        name
        for name, delta in deltas.items()
        if (
            (delta < -_METRIC_TOLERANCE) if _is_higher_better(name) else (delta > _METRIC_TOLERANCE)
        )
    )

    fixture_before, _ = _score_rows(fixture_rows, ruleset, apply_rules=False)
    fixture_after, fixture_outcomes = _score_rows(fixture_rows, ruleset, apply_rules=True)
    fixed: list[dict[str, Any]] = []
    for row, before, after, outcome in zip(
        fixture_rows, fixture_before, fixture_after, fixture_outcomes, strict=True
    ):
        if after.e2e_correct > before.e2e_correct:
            fixed.append(
                {
                    "case_id": row.get("case_id"),
                    "call_id": row["call_id"],
                    "turn_index": row["turn_index"],
                    "rules": list(outcome.rules),
                }
            )

    hashes_after = _artifact_hashes(settings.golden_artifact_dir, settings.golden_version)
    seal_after = verify_seal(settings.golden_artifact_dir, settings.golden_version)
    seal_after.assert_ok()
    ok = (
        not regressions
        and hashes_before == hashes_after
        and seal_after.manifest is not None
        and seal_after.manifest.content_hash == settings.expected_golden_hash
    )
    return {
        "synthetic_warning": SYNTHETIC_WARNING,
        "golden_dataset_id": manifest["dataset_id"],
        "golden_content_hash": manifest["content_hash"],
        "seal_ok_before": True,
        "seal_ok_after": seal_after.ok,
        "artifact_hashes_unchanged": hashes_before == hashes_after,
        "metrics_before": before_metrics,
        "metrics_after": after_metrics,
        "deltas": deltas,
        "regressions": regressions,
        "turns_changed": sum(1 for outcome in outcomes if outcome.changed),
        "rule_counts": _count_rules(outcomes),
        "fixture_e2e_fixed": fixed,
        "ok": ok,
    }


def _count_rules(outcomes: Sequence[RuleOutcome]) -> dict[str, int]:
    """Count fired rule names across all outcomes."""
    counts: dict[str, int] = {}
    for outcome in outcomes:
        for name in outcome.rules:
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: print a golden regression report and optionally write config/report."""
    parser = argparse.ArgumentParser(description="Check P2-08 rules against the sealed stand-in.")
    parser.add_argument("--report", help="Optional path for the JSON regression report.")
    parser.add_argument("--config-output", help="Optional path for the generated boosting JSON.")
    args = parser.parse_args(argv)
    settings = VocabSettings()
    ruleset = VocabRuleSet(settings)
    report = check_golden_regression(settings, ruleset)
    if args.config_output:
        path = Path(args.config_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(ruleset.boost_config.to_json(), encoding="utf-8")
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True, indent=2))
    assert report["ok"], "P2-08 golden regression check failed"
    print(SYNTHETIC_WARNING)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via python -m
    raise SystemExit(main())
