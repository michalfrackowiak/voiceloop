"""Contract-valid STT artifact reports and read-only P2-09 gate integration.

The generated JSON is a full ``eval_report`` 1.0.0 payload. Because that schema
closes top-level properties, the P3-04 extension is nested under the optional
``decoding_settings`` object: it adds hashes, probe summaries and provenance
without removing, renaming or weakening any required contract field. The
extension is validated before every write and is documented in the README.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from services.evalgate.bootstrap import ConfidenceInterval, paired_bootstrap_ci
from services.evalgate.gate import (
    Gate,
    GateDecision,
    GateError,
    GateSettings,
    InMemoryAliasStore,
    PromotionAction,
    PromotionRecord,
)
from services.evalgate.gate.gate import ConvertedArtifact
from services.evalgate.harness import build_report_timestamp, build_slice_metrics
from services.evalgate.metrics import (
    CORE_METRIC_NAMES,
    EntityPatterns,
    Prediction,
    TurnMetrics,
    aggregate_metrics,
    evaluate_turn,
)
from services.evalgate.report import build_report
from voiceloop_common import ConfigError, ValidationError, validate

from .artifact import FakeArtifact, artifact_hash_from_uri, create_fake_artifact
from .decoding import (
    DecodeCombination,
    decode_combinations,
    decoding_hash_from_fields,
    describe_combination,
    language_combinations,
    production_combination,
)
from .probes import (
    ProbeInputs,
    ProbeSuiteResult,
    fake_transcribe,
    hallucination_summary,
    load_probe_inputs,
    multilingual_summary,
    run_probe_suite,
    verify_inputs,
)
from .settings import ArtifactSettings

REPORT_SCHEMA = "eval_report"
EXTENSION_KEY = "artifact_extension"
EXTENSION_VERSION = "internal-1.0.0"
HALLUCINATION_METRIC = "silence_hallucination_rate"
FORGETTING_METRIC = "multilingual_forgetting_rate"
EXTRA_METRICS: tuple[str, str] = (HALLUCINATION_METRIC, FORGETTING_METRIC)
EVAL_SETS: tuple[str, ...] = ("general_capability",)
_FORBIDDEN_TEXT_FRAGMENTS: tuple[str, ...] = (
    "ada fixture",
    "sam sample",
    "morgan mock",
    "@example.com",
    "+1 555 010",
    "thank you for calling",
)
_FORBIDDEN_KEYS = frozenset(
    {"transcript", "expected_transcript", "hypothesis", "prior_turns", "audio_bytes", "wav_bytes"}
)


@dataclass(frozen=True)
class ArtifactEvaluation:
    """Everything produced by one exact-artifact evaluation run."""

    report: dict[str, Any]
    champion_report: dict[str, Any]
    probe_suite: ProbeSuiteResult
    gate_summary: Mapping[str, Any]
    hallucination: Mapping[str, Any]
    multilingual: Mapping[str, Any]
    provenance: Mapping[str, Any]


def _metric_statistic(turns: Sequence[TurnMetrics], metric: str) -> float:
    """Reduce one resample of turns to a named aggregate metric."""
    return aggregate_metrics(turns)[metric]


def _score_turns(
    inputs: ProbeInputs,
    artifact: FakeArtifact,
    combination: DecodeCombination,
    settings: ArtifactSettings,
    patterns: EntityPatterns,
) -> tuple[TurnMetrics, ...]:
    """Score every committed fixture turn with one fake artifact and decode setup."""
    turns: list[TurnMetrics] = []
    for row, event, prior_count in zip(
        inputs.rows, inputs.events, inputs.prior_turn_counts, strict=True
    ):
        transcript, slots = fake_transcribe(
            artifact, row, event, combination.settings, settings, prior_count
        )
        turns.append(
            evaluate_turn(
                str(row.get("expected_transcript", "")),
                dict(row.get("expected_slots", {})),
                Prediction(transcript=transcript, slots=slots),
                patterns,
            )
        )
    return tuple(turns)


def _core_intervals(
    champion: Sequence[TurnMetrics],
    challenger: Sequence[TurnMetrics],
    settings: ArtifactSettings,
) -> dict[str, ConfidenceInterval]:
    """Build paired challenger-minus-champion intervals for the five core metrics."""
    return {
        name: paired_bootstrap_ci(
            champion,
            challenger,
            partial(_metric_statistic, metric=name),
            level=settings.bootstrap_level,
            iterations=settings.bootstrap_iterations,
            seed=settings.bootstrap_seed,
        )
        for name in CORE_METRIC_NAMES
    }


def _interval_from_summary(summary: Mapping[str, Any]) -> ConfidenceInterval:
    """Convert a diagnostic delta summary into a contract confidence interval."""
    values = summary["delta_ci"]
    return ConfidenceInterval(
        lower=float(values["lower"]),
        upper=float(values["upper"]),
        level=float(values["level"]),
    )


def _build_extension(
    *,
    artifact: FakeArtifact,
    settings: ArtifactSettings,
    primary: DecodeCombination,
    combinations: Sequence[DecodeCombination],
    probe_suite: ProbeSuiteResult,
    hallucination: Mapping[str, Any],
    multilingual: Mapping[str, Any],
    provenance: Mapping[str, Any],
    gate: Mapping[str, Any],
    source_checkpoint_uri: str | None = None,
) -> dict[str, Any]:
    """Build the validated internal extension for champion or challenger reports."""
    return {
        "extension_version": EXTENSION_VERSION,
        "contract": (
            "Internal P3-04 extension nested in eval_report 1.0.0 decoding_settings; "
            "it never removes or weakens a contract field."
        ),
        "artifact": {
            "sha256": artifact.sha256,
            "uri": artifact.uri,
            "runtime_label": artifact.runtime_label,
            "behaviour": artifact.behaviour,
            "conversion_recipe": artifact.conversion_recipe,
            "source_checkpoint_uri": (
                settings.checkpoint_uri if source_checkpoint_uri is None else source_checkpoint_uri
            ),
            "evaluated_artifact_uri": artifact.uri,
            "checkpoint_uri_evaluated": False,
            "content_addressed": True,
            "read_only_after_creation": True,
        },
        "decoding": {
            "sha256": primary.settings.sha256,
            "combination_count": len(combinations),
            "combinations": [describe_combination(combination) for combination in combinations],
        },
        "probes": probe_suite.summary(settings.max_error_details),
        "hallucination": dict(hallucination),
        "multilingual_forgetting": dict(multilingual),
        "provenance": dict(provenance),
        "gate": dict(gate),
        "metric_semantics": {
            "primary_metric": "e2e_booking_accuracy",
            "primary_definition": "end-to-end booking-field accuracy from audio",
            "extra_metrics_are_diagnostics": list(EXTRA_METRICS),
            "ci_definition": (
                "paired bootstrap over shared fixture turns; challenger minus champion"
            ),
            "promotion": "not_performed",
        },
    }


class ExactArtifactConverter:
    """Converter that always substitutes a content-addressed artifact for a checkpoint."""

    def __init__(self, artifact: FakeArtifact) -> None:
        """Bind the converter to one immutable fake artifact."""
        self._artifact = artifact

    def convert(self, source_uri: str) -> ConvertedArtifact:
        """Return the deployable artifact URI, ignoring the checkpoint URI content."""
        if not isinstance(source_uri, str) or not source_uri.strip():
            raise ValidationError("source checkpoint URI must be non-blank")
        return ConvertedArtifact(artifact_uri=self._artifact.uri, source_uri=source_uri)


class HashedReportEvaluator:
    """Evaluator that refuses every artifact URI except the exact hashed artifact."""

    def __init__(self, artifact: FakeArtifact, report: Mapping[str, Any]) -> None:
        """Bind the evaluator to one artifact and its precomputed report."""
        self._artifact = artifact
        self._report = dict(report)

    def evaluate(self, artifact_uri: str, eval_sets: Sequence[str]) -> dict[str, Any]:
        """Return a validated report only for the exact content-addressed artifact."""
        del eval_sets
        validate_stt_report(self._report)
        if artifact_uri != self._artifact.uri:
            raise GateError(f"refusing to evaluate non-deployable URI: {artifact_uri!r}")
        if artifact_hash_from_uri(artifact_uri) != self._artifact.sha256:
            raise GateError("artifact URI hash does not match the evaluated artifact")
        validate(self._report, REPORT_SCHEMA)
        return deepcopy(self._report)


class InMemoryAuditLog:
    """No-write audit sink for read-only gate evaluation."""

    def __init__(self) -> None:
        """Initialise an empty in-memory record list."""
        self.records: list[PromotionRecord] = []

    def record(
        self,
        *,
        actor_id: str,
        action: PromotionAction,
        reason: str,
        before_version: str,
        after_version: str,
        task_id: str = "P2-09",
    ) -> PromotionRecord:
        """Append one in-memory promotion record and return it."""
        record = PromotionRecord(
            actor_id=actor_id,
            action=action,
            reason=reason,
            before_version=before_version,
            after_version=after_version,
            task_id=task_id,
        )
        self.records.append(record)
        return record


def run_exact_artifact_gate(
    artifact: FakeArtifact,
    report: Mapping[str, Any],
    champion_report: Mapping[str, Any],
    *,
    settings: GateSettings | None = None,
    checkpoint_uri: str = "fake://trainer/checkpoint",
) -> GateDecision:
    """Run the read-only P2-09 gate against the exact evaluated artifact.

    Args:
        artifact: Content-addressed fake artifact under evaluation.
        report: Contract-valid challenger report for that artifact.
        champion_report: Contract-valid champion comparison report.
        settings: P2-09 settings (read-only; no threshold is changed here).
        checkpoint_uri: Stand-in source URI that the converter must replace.

    Returns:
        The P2-09 decision, including reasons and slice deltas.

    """
    validate_stt_report(report)
    validate_stt_report(champion_report)
    converter = ExactArtifactConverter(artifact)
    evaluator = HashedReportEvaluator(artifact, report)
    gate = Gate(
        settings=settings or GateSettings(),
        converter=converter,
        evaluator=evaluator,
        alias_store=InMemoryAliasStore(
            {"@champion": "fixture-champion", "@challenger": "fixture-challenger"}
        ),
        audit_log=InMemoryAuditLog(),
    )
    decision, _ = gate.evaluate(checkpoint_uri, champion_report)
    return decision


def _assert_no_payload_leak(value: Any) -> None:
    """Reject transcripts, audio bytes or fixture PII-shaped tokens in a report."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and key.lower() in _FORBIDDEN_KEYS:
                raise ValidationError("report extension must not carry transcripts or audio bytes")
            _assert_no_payload_leak(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _assert_no_payload_leak(item)
        return
    if isinstance(value, str):
        folded = value.casefold()
        if any(fragment in folded for fragment in _FORBIDDEN_TEXT_FRAGMENTS):
            raise ValidationError("report contains fixture text or PII-shaped data")
        if re.search(r"RIFF\d{4}WAVE", value):
            raise ValidationError("report contains WAV audio bytes")


def validate_stt_report(report: Mapping[str, Any]) -> None:
    """Validate the 1.0.0 contract and every mandatory P3-04 extension field.

    Raises:
        ValidationError: If the contract, hashes, diagnostic metrics, slices or
            provenance are absent or internally inconsistent.

    """
    validate(dict(report), REPORT_SCHEMA)
    decoding = report.get("decoding_settings")
    if not isinstance(decoding, Mapping):
        raise ValidationError("STT report is missing decoding_settings")
    extension = decoding.get(EXTENSION_KEY)
    if not isinstance(extension, Mapping):
        raise ValidationError("STT report is missing the artifact extension")
    if extension.get("extension_version") != EXTENSION_VERSION:
        raise ValidationError("unsupported artifact extension version")
    artifact_data = extension.get("artifact")
    if not isinstance(artifact_data, Mapping):
        raise ValidationError("artifact extension is missing artifact provenance")
    reported_hash = artifact_hash_from_uri(str(report["artifact_uri"]))
    if artifact_data.get("sha256") != reported_hash:
        raise ValidationError("reported artifact hash does not match artifact_uri")
    recorded_hash = decoding.get("decoding_settings_sha256")
    if recorded_hash != decoding_hash_from_fields(decoding):
        raise ValidationError("reported decoding settings hash does not match the settings")
    metrics = report["metrics"]
    for metric in EXTRA_METRICS:
        value = metrics.get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValidationError(f"STT report is missing numeric {metric}")
        if not math.isfinite(float(value)):
            raise ValidationError(f"STT report has a non-finite {metric}")
    slices = report["slice_metrics"]
    silence = slices.get("silence")
    other_languages = slices.get("other_languages")
    if not isinstance(silence, Mapping) or "hallucination_rate" not in silence:
        raise ValidationError("silence slice must report hallucination_rate")
    if not isinstance(other_languages, Mapping) or "forgetting_rate" not in other_languages:
        raise ValidationError("other_languages slice must report forgetting_rate")
    if "turns" not in silence or "turns" not in other_languages:
        raise ValidationError("safety slices must report turn counts")
    for required in ("probes", "hallucination", "multilingual_forgetting", "provenance", "gate"):
        if required not in extension:
            raise ValidationError(f"artifact extension is missing {required}")
    _assert_no_payload_leak(report)


def write_stt_report(report: Mapping[str, Any], path: str | Path) -> Path:
    """Validate and write one STT artifact report as sorted UTF-8 JSON."""
    validate_stt_report(report)
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def evaluate_artifact(
    settings: ArtifactSettings,
    artifact: FakeArtifact,
    *,
    gate_settings: GateSettings | None = None,
) -> ArtifactEvaluation:
    """Evaluate one content-addressed fake artifact on committed fixtures.

    Args:
        settings: Typed artifact settings.
        artifact: Fake artifact to evaluate; its hash is recorded in the report.
        gate_settings: Optional read-only P2-09 gate settings.

    Returns:
        The validated report plus diagnostics and gate decision.

    Raises:
        ValidationError: If committed fixture/golden hashes change during the run.

    """
    inputs = load_probe_inputs(settings)
    provenance_before = dict(inputs.provenance)
    patterns = EntityPatterns()
    primary = production_combination(settings)
    base_combinations = (*decode_combinations(settings), primary)
    combinations = (
        *base_combinations,
        *language_combinations(base_combinations, settings.non_english_languages()),
    )
    probe_suite = run_probe_suite(inputs, artifact, combinations, settings)

    healthy = create_fake_artifact("healthy", settings.runtime_label)
    champion_probe_suite = run_probe_suite(inputs, healthy, combinations, settings)
    champion_turns = _score_turns(inputs, healthy, primary, settings, patterns)
    challenger_turns = _score_turns(inputs, artifact, primary, settings, patterns)

    intervals = _core_intervals(champion_turns, challenger_turns, settings)
    hallucination = hallucination_summary(inputs, artifact, primary, settings)
    multilingual = multilingual_summary(inputs, artifact, primary, settings)
    intervals[HALLUCINATION_METRIC] = _interval_from_summary(hallucination)
    intervals[FORGETTING_METRIC] = _interval_from_summary(multilingual)

    champion_hallucination = hallucination_summary(inputs, healthy, primary, settings)
    champion_multilingual = multilingual_summary(inputs, healthy, primary, settings)
    champion_intervals = _core_intervals(champion_turns, champion_turns, settings)
    champion_intervals[HALLUCINATION_METRIC] = _interval_from_summary(champion_hallucination)
    champion_intervals[FORGETTING_METRIC] = _interval_from_summary(champion_multilingual)

    metrics = aggregate_metrics(challenger_turns)
    metrics[HALLUCINATION_METRIC] = float(hallucination["rate"])
    metrics[FORGETTING_METRIC] = float(multilingual["forgetting_rate"])

    slice_metrics = build_slice_metrics(
        inputs.rows, challenger_turns, patterns, settings.other_language_pattern
    )
    slice_metrics["silence"]["hallucination_rate"] = float(hallucination["rate"])
    slice_metrics["other_languages"]["forgetting_rate"] = float(multilingual["forgetting_rate"])

    champion_metrics = aggregate_metrics(champion_turns)
    champion_metrics[HALLUCINATION_METRIC] = float(champion_hallucination["rate"])
    champion_metrics[FORGETTING_METRIC] = float(champion_multilingual["forgetting_rate"])

    champion_slices = build_slice_metrics(
        inputs.rows, champion_turns, patterns, settings.other_language_pattern
    )
    champion_slices["silence"]["hallucination_rate"] = float(champion_hallucination["rate"])
    champion_slices["other_languages"]["forgetting_rate"] = float(
        champion_multilingual["forgetting_rate"]
    )

    champion_extension = _build_extension(
        artifact=healthy,
        settings=settings,
        primary=primary,
        combinations=combinations,
        probe_suite=champion_probe_suite,
        hallucination=champion_hallucination,
        multilingual=champion_multilingual,
        provenance=provenance_before,
        gate={
            "status": "not_applicable_healthy_fake_baseline",
            "reason": "Healthy fake champion is a comparison baseline, not a promotion candidate.",
        },
        source_checkpoint_uri="not_applicable_healthy_fake_baseline",
    )
    champion_report = build_report(
        artifact_uri=healthy.uri,
        eval_sets=EVAL_SETS,
        metrics=champion_metrics,
        confidence_intervals=champion_intervals,
        slice_metrics=champion_slices,
        decoding_settings={
            **primary.settings.to_report_fields(),
            EXTENSION_KEY: champion_extension,
        },
    )
    champion_report["gate_decision"] = {
        "outcome": "not_evaluated",
        "reason": "Healthy fake champion baseline; no promotion or alias change was made.",
        "evaluated_at": build_report_timestamp(),
    }
    validate_stt_report(champion_report)

    extension = _build_extension(
        artifact=artifact,
        settings=settings,
        primary=primary,
        combinations=combinations,
        probe_suite=probe_suite,
        hallucination=hallucination,
        multilingual=multilingual,
        provenance=provenance_before,
        gate={"status": "pending"},
    )

    report = build_report(
        artifact_uri=artifact.uri,
        eval_sets=EVAL_SETS,
        metrics=metrics,
        confidence_intervals=intervals,
        slice_metrics=slice_metrics,
        decoding_settings={**primary.settings.to_report_fields(), EXTENSION_KEY: extension},
    )
    decision = run_exact_artifact_gate(
        artifact,
        report,
        champion_report,
        settings=gate_settings,
        checkpoint_uri=settings.checkpoint_uri,
    )
    extension["gate"] = {
        "status": "evaluated",
        "evaluated_artifact_uri": artifact.uri,
        "source_checkpoint_uri": settings.checkpoint_uri,
        **decision.to_dict(),
    }
    if probe_suite.status == "fail":
        codes = ", ".join(sorted(extension["probes"]["errors_by_code"]))
        report["gate_decision"] = {
            "outcome": "fail",
            "reason": f"artifact probe suite failed ({codes}); no promotion attempted",
            "evaluated_at": build_report_timestamp(),
        }
    else:
        report["gate_decision"] = {
            "outcome": "not_evaluated",
            "reason": (
                "Artifact safety run only; read-only P2-09 decision is recorded "
                "in the extension. No promotion or threshold change was made."
            ),
            "evaluated_at": build_report_timestamp(),
        }

    provenance_after = verify_inputs(settings)
    if provenance_after != provenance_before:
        raise ValidationError("committed fixture or golden provenance changed during evaluation")
    champion_extension["provenance"] = provenance_after
    extension["provenance"] = provenance_after
    validate_stt_report(champion_report)
    validate_stt_report(report)
    return ArtifactEvaluation(
        report=report,
        champion_report=champion_report,
        probe_suite=probe_suite,
        gate_summary=extension["gate"],
        hallucination=hallucination,
        multilingual=multilingual,
        provenance=provenance_after,
    )


def evaluate_checkpoint_uris_do_not_change_result(
    settings: ArtifactSettings,
    artifact: FakeArtifact,
    other_checkpoint_uri: str,
) -> bool:
    """Return whether two checkpoint URIs produce the same evaluated artifact hash.

    This is a small proof helper: the exact-artifact converter substitutes the
    content-addressed artifact, so changing checkpoint lineage cannot change the
    artifact evaluated by the gate.
    """
    if not other_checkpoint_uri.strip():
        raise ConfigError("comparison checkpoint URI must be non-blank")
    converter = ExactArtifactConverter(artifact)
    first = converter.convert(settings.checkpoint_uri)
    second = converter.convert(other_checkpoint_uri)
    return first.artifact_uri == second.artifact_uri == artifact.uri


__all__ = [
    "EXTENSION_KEY",
    "EXTENSION_VERSION",
    "EXTRA_METRICS",
    "ArtifactEvaluation",
    "ExactArtifactConverter",
    "HashedReportEvaluator",
    "validate_stt_report",
    "write_stt_report",
    "evaluate_artifact",
    "evaluate_checkpoint_uris_do_not_change_result",
    "run_exact_artifact_gate",
]
