"""Test-only builders for P3-07-shaped shadow comparison reports."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from services.evalgate.gate import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    Gate,
    GateSettings,
    InMemoryAliasStore,
    PrefixArtifactConverter,
    PromotionAuditLog,
    RefusalSink,
    ShadowEvidenceSettings,
    ShadowEvidenceSource,
    StaticReportEvaluator,
)

SYNTHETIC_STATUS = "SYNTHETIC/FIXTURE STAND-IN - NOT PRODUCTION SHADOW EVIDENCE"
CANDIDATE_URI = "converted://stt-v4"
CHAMPION_URI = "converted://stt-v3"
REPORT_URI = "fake://shadow/comparison.json"
NOW = datetime(2026, 11, 25, 12, 0, 0, tzinfo=UTC)


def make_shadow_report(
    *,
    generated_at: str | None = None,
    candidate_uri: str = CANDIDATE_URI,
    champion_uri: str = CHAMPION_URI,
    turns_observed: int = 36,
    calls_scored: int = 6,
    successful_results: int = 6,
    status: str = SYNTHETIC_STATUS,
    synthetic: bool = True,
    fixture_standin: bool = True,
    real_traffic: bool = False,
) -> dict[str, Any]:
    """Build a complete internal shadow artifact matching P3-07's shape."""
    timestamp = generated_at or NOW.isoformat(timespec="seconds").replace("+00:00", "Z")
    return {
        "artifact_type": "voiceloop.shadow.comparison",
        "comparison_schema_version": "internal-1.0.0",
        "schema_contract": "internal; not schemas/eval_report.schema.json",
        "generated_at": timestamp,
        "status": status,
        "task_id": "P3-07",
        "promotion_decision": "not_evaluated",
        "promotion": {
            "decision": "not_evaluated",
            "alias_writes": 0,
            "champion_flips": 0,
            "gate_edits": 0,
            "thresholds_changed": False,
            "evidence_only_for": "P3-08",
        },
        "provenance": {
            "synthetic": synthetic,
            "fixture_standin": fixture_standin,
            "real_traffic": real_traffic,
            "network_calls": False,
            "audio_bytes_logged": False,
            "transcripts_logged": False,
            "join_key": "call_id",
            "models": [
                {
                    "source": "candidate",
                    "artifact_uri": candidate_uri,
                    "model_id": "stt-candidate-v4",
                    "synthetic": synthetic,
                },
                {
                    "source": "champion",
                    "artifact_uri": champion_uri,
                    "model_id": "stt-champion-v3",
                    "synthetic": synthetic,
                },
            ],
            "timestamps": {
                "first_started_at": timestamp,
                "last_completed_at": timestamp,
            },
            "successful_results": successful_results,
        },
        "isolation": {
            "max_pending": 64,
            "mirror_budget_ms": 2.0,
            "max_turns_per_drain": 128,
            "models_invoked_by_call_path": False,
            "observed_max_elapsed_seconds": 0.0005,
            "within_budget": True,
            "clock": "injected clock; no real sleeps in this package",
        },
        "sample_counts": {
            "turns_observed": turns_observed,
            "turns_mirrored": turns_observed,
            "turns_dropped": 0,
            "sink_errors": 0,
            "results_candidate": 6,
            "results_champion": 6,
            "failed_results": 0,
            "calls_observed": 6,
            "calls_with_outcome": calls_scored,
            "calls_scored": calls_scored,
            "calls_missing_outcome": 0,
            "calls_excluded_by_status": 0,
            "calls_empty_final_record": 0,
            "calls_missing_model_result": 0,
        },
        "metrics": {
            "candidate": {
                "outcome_call_accuracy": 1.0,
                "outcome_error_rate": 0.0,
                "outcome_field_accuracy": 1.0,
            },
            "champion": {
                "outcome_call_accuracy": 0.5,
                "outcome_error_rate": 0.5,
                "outcome_field_accuracy": 0.5,
            },
        },
        "deltas": {
            "outcome_call_accuracy": 0.5,
            "outcome_error_rate": -0.5,
            "outcome_field_accuracy": 0.5,
        },
        "metric_semantics": {
            "primary_metric": "outcome_call_accuracy",
            "delta_definition": "candidate - champion",
            "promotion_rule": "Evidence only; no alias is flipped.",
            "limitations": "Synthetic/fixture stand-in; not production evidence.",
        },
        "shadow_evidence": {
            "report_uri": REPORT_URI,
            "turns_observed": turns_observed,
            "outcome_error_rate": 0.0,
            "source": "candidate",
            "join_key": "call_id",
            "synthetic_standin": synthetic,
        },
        "report_uri": REPORT_URI,
    }


def make_stt_gate(
    tmp_path: Path,
    challenger_report: dict[str, Any],
    *,
    shadow_source: ShadowEvidenceSource | None,
    refusal_log: RefusalSink,
    shadow_settings: ShadowEvidenceSettings | None = None,
) -> tuple[Gate, InMemoryAliasStore, PromotionAuditLog]:
    """Assemble an STT gate with fake evaluation, aliases and shadow seams."""
    store = InMemoryAliasStore({CHAMPION_ALIAS: "v3", CHALLENGER_ALIAS: "v4"})
    promotion_log = PromotionAuditLog(tmp_path / "promotion_audit.jsonl")
    gate = Gate(
        settings=GateSettings(min_e2e_margin=0.01),
        converter=PrefixArtifactConverter(),
        evaluator=StaticReportEvaluator({CANDIDATE_URI: challenger_report}),
        alias_store=store,
        audit_log=promotion_log,
        shadow_source=shadow_source,
        shadow_settings=shadow_settings or ShadowEvidenceSettings(),
        refusal_log=refusal_log,
        clock=lambda: NOW,
    )
    return gate, store, promotion_log
