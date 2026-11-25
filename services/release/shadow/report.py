"""Internal comparison artifact for P3-07 shadow evidence.

The report is an **internal shape**, not ``schemas/eval_report.schema.json``
1.0.0. It contains outcome-based aggregate metrics, candidate-minus-champion
deltas, sample counts and call-path isolation evidence. It is labelled as a
synthetic/fixture stand-in and always carries ``promotion_decision:
not_evaluated``: it is input evidence for P3-08, never a promotion decision.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from voiceloop_common import ValidationError, utcnow

from .runner import ModelResult, ResultStatus
from .scoring import JOIN_KEY, ComparisonData

SHADOW_ARTIFACT_TYPE = "voiceloop.shadow.comparison"
SHADOW_SCHEMA_VERSION = "internal-1.0.0"
PROMOTION_DECISION = "not_evaluated"
PRIMARY_METRIC = "outcome_call_accuracy"
METRIC_NAMES = ("outcome_call_accuracy", "outcome_error_rate", "outcome_field_accuracy")

_TOP_LEVEL_KEYS = frozenset(
    {
        "artifact_type",
        "comparison_schema_version",
        "schema_contract",
        "generated_at",
        "status",
        "task_id",
        "promotion_decision",
        "promotion",
        "provenance",
        "isolation",
        "sample_counts",
        "metrics",
        "deltas",
        "metric_semantics",
        "shadow_evidence",
        "report_uri",
    }
)
_BANNED_KEYS = frozenset(
    {
        "transcript",
        "audio_bytes",
        "final_record",
        "guest_name",
        "email",
        "phone",
        "card_number",
        "credentials",
        "password",
        "secret",
        "token",
    }
)


def _iso(value: Any) -> str | None:
    """Return an ISO UTC string for a datetime-like value, or None."""
    if not hasattr(value, "isoformat"):
        return None
    return str(value.isoformat()).replace("+00:00", "Z")


def _timestamps(results: tuple[ModelResult, ...]) -> dict[str, str | None]:
    """Return safe first-start and last-completion timestamps for results."""
    started = [result.started_at for result in results]
    completed = [result.completed_at for result in results]
    return {
        "first_started_at": _iso(min(started)) if started else None,
        "last_completed_at": _iso(max(completed)) if completed else None,
    }


def build_shadow_report(data: ComparisonData, report_uri: str) -> dict[str, Any]:
    """Build and validate the internal shadow comparison artifact.

    Args:
        data: Outcome-based comparison produced by :func:`score_comparison`.
        report_uri: URI/path where the artifact will be written; referenced by
            the ``shadow_evidence`` block for P3-08.

    Returns:
        A JSON-serialisable mapping containing only allowlisted fields.

    """
    settings = data.settings
    counts = dict(data.sample_counts)
    successful = [result for result in data.results if result.status is ResultStatus.OK]
    report: dict[str, Any] = {
        "artifact_type": SHADOW_ARTIFACT_TYPE,
        "comparison_schema_version": SHADOW_SCHEMA_VERSION,
        "schema_contract": "internal; not schemas/eval_report.schema.json",
        "generated_at": utcnow().isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": settings.report_status_label,
        "task_id": settings.task_id,
        "promotion_decision": PROMOTION_DECISION,
        "promotion": {
            "decision": PROMOTION_DECISION,
            "alias_writes": 0,
            "champion_flips": 0,
            "gate_edits": 0,
            "thresholds_changed": False,
            "evidence_only_for": "P3-08",
        },
        "provenance": {
            "synthetic": data.synthetic,
            "fixture_standin": data.synthetic,
            "real_traffic": False,
            "network_calls": False,
            "audio_bytes_logged": False,
            "transcripts_logged": False,
            "join_key": JOIN_KEY,
            "models": [
                {
                    "source": "candidate",
                    "artifact_uri": data.candidate_artifact_uri,
                    "model_id": data.candidate_model_id,
                    "synthetic": data.synthetic,
                },
                {
                    "source": "champion",
                    "artifact_uri": data.champion_artifact_uri,
                    "model_id": data.champion_model_id,
                    "synthetic": data.synthetic,
                },
            ],
            "timestamps": _timestamps(data.results),
            "successful_results": len(successful),
        },
        "isolation": {
            "max_pending": settings.max_pending,
            "mirror_budget_ms": settings.mirror_budget_ms,
            "max_turns_per_drain": settings.max_turns_per_drain,
            "models_invoked_by_call_path": False,
            "observed_max_elapsed_seconds": data.mirror_stats.max_elapsed_seconds,
            "within_budget": data.mirror_stats.max_elapsed_seconds
            <= settings.mirror_budget_ms / 1000.0,
            "clock": "injected clock; no real sleeps in this package",
        },
        "sample_counts": counts,
        "metrics": {name: dict(data.metrics[name]) for name in ("candidate", "champion")},
        "deltas": {name: float(data.deltas[name]) for name in METRIC_NAMES},
        "metric_semantics": {
            "primary_metric": PRIMARY_METRIC,
            "primary_definition": (
                "all-or-nothing agreement between final parsed booking fields "
                "and the call's final booking record"
            ),
            "field_metric_definition": (
                "exact canonical-value matches over the union of reference and "
                "predicted field names"
            ),
            "error_rate_definition": (
                "1 - outcome_call_accuracy; 0.0 with no scorable calls (no evidence)"
            ),
            "delta_definition": "candidate - champion",
            "positive_delta_favors": "candidate",
            "join_key": JOIN_KEY,
            "promotion_rule": (
                "Evidence only. No threshold is applied and no alias is flipped; "
                "WER is not used as a promotion signal (context.md section 2 "
                "invariant 4)."
            ),
            "limitations": (
                "Synthetic/fixture stand-in. No human listened to audio and no "
                "real converted artifact served traffic (context.md section 2 "
                "invariants 2 and 3)."
            ),
        },
        "shadow_evidence": {
            "report_uri": report_uri,
            "turns_observed": counts["turns_observed"],
            "outcome_error_rate": float(data.metrics["candidate"]["outcome_error_rate"]),
            "source": "candidate",
            "join_key": JOIN_KEY,
            "synthetic_standin": data.synthetic,
        },
        "report_uri": report_uri,
    }
    validate_shadow_report(report)
    return report


def validate_shadow_report(report: Mapping[str, Any]) -> None:
    """Check the internal shape, promotion label and PII-safe key allowlist.

    Raises:
        ValidationError: If a required field is missing, a metric or count is
            malformed, a disallowed key appears, or the report claims a
            promotion decision.

    """
    missing = sorted(_TOP_LEVEL_KEYS - set(report))
    if missing:
        raise ValidationError(f"shadow report missing: {', '.join(missing)}")
    extra = sorted(set(report) - _TOP_LEVEL_KEYS)
    if extra:
        raise ValidationError(f"shadow report has unexpected fields: {', '.join(extra)}")
    if report["artifact_type"] != SHADOW_ARTIFACT_TYPE:
        raise ValidationError("unexpected shadow artifact_type")
    if report["comparison_schema_version"] != SHADOW_SCHEMA_VERSION:
        raise ValidationError("unsupported shadow comparison_schema_version")
    if report["promotion_decision"] != PROMOTION_DECISION:
        raise ValidationError("shadow report must not claim a promotion decision")
    promotion = report["promotion"]
    if not isinstance(promotion, Mapping) or promotion.get("decision") != PROMOTION_DECISION:
        raise ValidationError("promotion block must be not_evaluated")
    if promotion.get("alias_writes") != 0 or promotion.get("champion_flips") != 0:
        raise ValidationError("shadow report must not perform alias writes or champion flips")
    if promotion.get("thresholds_changed") is not False or promotion.get("gate_edits") != 0:
        raise ValidationError("shadow report must not change gates or thresholds")

    counts = report["sample_counts"]
    if not isinstance(counts, Mapping):
        raise ValidationError("sample_counts must be an object")
    for name, value in counts.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValidationError(f"sample_counts.{name} must be a non-negative integer")

    metrics = report["metrics"]
    if not isinstance(metrics, Mapping) or set(metrics) != {"candidate", "champion"}:
        raise ValidationError("metrics must contain candidate and champion")
    for side in metrics.values():
        if not isinstance(side, Mapping) or set(side) != set(METRIC_NAMES):
            raise ValidationError("each metrics side needs the three outcome metrics")
        for value in side.values():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValidationError("metric values must be numeric")
            if not 0.0 <= float(value) <= 1.0:
                raise ValidationError("metric values must lie in [0, 1]")

    deltas = report["deltas"]
    if not isinstance(deltas, Mapping) or set(deltas) != set(METRIC_NAMES):
        raise ValidationError("deltas must cover all three outcome metrics")
    for value in deltas.values():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValidationError("delta values must be numeric")

    evidence = report["shadow_evidence"]
    if not isinstance(evidence, Mapping):
        raise ValidationError("shadow_evidence must be an object")
    if not evidence.get("report_uri") or evidence.get("turns_observed") != counts.get(
        "turns_observed"
    ):
        raise ValidationError("shadow_evidence must reference this report and its turn count")
    if evidence.get("outcome_error_rate") != metrics["candidate"]["outcome_error_rate"]:
        raise ValidationError("shadow_evidence error rate must match the candidate metric")

    _reject_banned_keys(report)


def _reject_banned_keys(value: Any, path: str = "report") -> None:
    """Recursively reject keys that would carry raw guest data."""
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str) and key in _BANNED_KEYS:
                raise ValidationError(f"{path} contains a disallowed field: {key}")
            _reject_banned_keys(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_banned_keys(nested, f"{path}[{index}]")


class ReportSink(Protocol):
    """Destination seam for the comparison artifact."""

    def write(self, report: Mapping[str, Any], destination: str) -> str:
        """Persist the report and return the resolved destination URI/path."""
        ...


class FileReportSink:
    """Local-file FAKE/PRODUCTION-neutral report sink (no network client)."""

    def write(self, report: Mapping[str, Any], destination: str) -> str:
        """Validate and write the report as sorted UTF-8 JSON."""
        return str(write_shadow_report(report, destination))


class CollectingReportSink:
    """Deterministic in-memory FAKE report sink for tests."""

    def __init__(self) -> None:
        """Initialise an empty recorded-reports list."""
        self.reports: list[dict[str, Any]] = []
        self.destinations: list[str] = []

    def write(self, report: Mapping[str, Any], destination: str) -> str:
        """Store a deep-enough copy of the report and return the destination."""
        validate_shadow_report(report)
        self.reports.append(dict(report))
        self.destinations.append(destination)
        return destination


def write_shadow_report(report: Mapping[str, Any], path: str | Path) -> Path:
    """Validate and write the internal shadow comparison artifact.

    Args:
        report: Payload from :func:`build_shadow_report`.
        path: Destination path; parent directories are created if needed.

    Returns:
        The resolved destination path.

    """
    validate_shadow_report(report)
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination
