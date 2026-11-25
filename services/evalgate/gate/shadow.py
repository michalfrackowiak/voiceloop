"""Read-only shadow-evidence requirement for STT promotions (P3-08).

P3-07 emits an internal ``voiceloop.shadow.comparison`` artifact through a
report sink. This module consumes that artifact through the typed
:class:`ShadowEvidenceSource` protocol; it never runs a shadow candidate,
flips an alias, changes a threshold or edits a gate. Every rejection is a
structured :class:`GateRefusal` plus an append-only refusal audit record, and
every candidate/champion pairing and timestamp is checked before the evidence
can be used.

Shadow point deltas are **evidence only**: they are never promotion proof. The
P2-09 paired e2e delta CI and critical-slice rules still decide accuracy; this
module only adds a fail-closed precondition for STT changes
(``context.md`` section 9). All shipped evidence is a synthetic/fixture
stand-in, not a production-readiness claim.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field

from voiceloop_common import BaseSettings, ValidationError, settings_config, utcnow

from .audit import canonical_json, resolve_path, sha256_hex, utc_timestamp
from .rules import GateError

SHADOW_ARTIFACT_TYPE = "voiceloop.shadow.comparison"
SHADOW_SCHEMA_VERSION = "internal-1.0.0"
SHADOW_PROMOTION_DECISION = "not_evaluated"
SHADOW_PRIMARY_METRIC = "outcome_call_accuracy"
SHADOW_JOIN_KEY = "call_id"
SYNTHETIC_STATUS_MARKER = "SYNTHETIC"
TASK_ID = "P3-08"
# Bounded skew tolerated between the shadow writer's clock and the gate's clock.
MAX_FUTURE_CLOCK_SKEW_SECONDS = 60

RefusalCode = Literal[
    "shadow_evidence_missing",
    "shadow_evidence_incomplete",
    "shadow_evidence_stale",
    "shadow_evidence_not_synthetic",
    "shadow_evidence_promotion_claim",
    "shadow_evidence_mismatch",
    "shadow_evidence_invalid",
]
REFUSAL_CODES: frozenset[str] = frozenset(
    {
        "shadow_evidence_missing",
        "shadow_evidence_incomplete",
        "shadow_evidence_stale",
        "shadow_evidence_not_synthetic",
        "shadow_evidence_promotion_claim",
        "shadow_evidence_mismatch",
        "shadow_evidence_invalid",
    }
)

_REQUIRED_SECTIONS = (
    "status",
    "generated_at",
    "promotion",
    "provenance",
    "isolation",
    "sample_counts",
    "metrics",
    "deltas",
    "metric_semantics",
    "shadow_evidence",
    "report_uri",
)
_REQUIRED_COUNT_KEYS = (
    "turns_observed",
    "turns_mirrored",
    "results_candidate",
    "results_champion",
    "failed_results",
    "calls_observed",
    "calls_with_outcome",
    "calls_scored",
)
_METRIC_NAMES = ("outcome_call_accuracy", "outcome_error_rate", "outcome_field_accuracy")


class ShadowEvidenceSettings(BaseSettings):
    """Typed settings for the gate's read-only shadow-evidence check."""

    model_config = settings_config("VOICELOOP_GATE_SHADOW_")

    # Evidence minimums, not promotion thresholds: they prevent a report with
    # no scored turns or no successful side from counting as shadow evidence.
    max_report_age_hours: float = Field(default=24.0, gt=0)
    min_turns_observed: int = Field(default=1, ge=1)
    min_successful_results: int = Field(default=1, ge=1)
    refusal_log_path: str = "reports/evalgate/shadow_refusals.jsonl"


@dataclass(frozen=True)
class ShadowEvidenceRequest:
    """Identity and freshness context for one shadow-evidence lookup."""

    candidate_version: str
    champion_version: str
    candidate_artifact_uri: str
    champion_artifact_uri: str
    evaluated_at: str

    def __post_init__(self) -> None:
        """Reject blank identifiers before they reach a source or audit log."""
        for name in (
            "candidate_version",
            "champion_version",
            "candidate_artifact_uri",
            "champion_artifact_uri",
            "evaluated_at",
        ):
            _require_text(getattr(self, name), name)


class ShadowEvidenceSource(Protocol):
    """Read-only seam for loading a P3-07 comparison report."""

    def load(self, request: ShadowEvidenceRequest) -> Mapping[str, Any] | None:
        """Return the matching report, or None when no evidence exists."""
        ...


class StaticShadowEvidenceSource:
    """In-memory fake keyed by ``(candidate_version, champion_version)``."""

    def __init__(self, reports: Mapping[tuple[str, str], Mapping[str, Any]] | None = None) -> None:
        """Copy the supplied reports so a caller cannot mutate stored evidence."""
        self._reports: dict[tuple[str, str], dict[str, Any]] = {
            key: deepcopy(dict(value)) for key, value in (reports or {}).items()
        }

    def load(self, request: ShadowEvidenceRequest) -> Mapping[str, Any] | None:
        """Return a copy of the report for this version pair, if present."""
        report = self._reports.get((request.candidate_version, request.champion_version))
        return deepcopy(report) if report is not None else None


class CollectingShadowEvidenceSource:
    """Fake source that records requests before delegating them."""

    def __init__(self, delegate: ShadowEvidenceSource) -> None:
        """Wrap a delegate and start with an empty request log."""
        self.delegate = delegate
        self.requests: list[ShadowEvidenceRequest] = []

    def load(self, request: ShadowEvidenceRequest) -> Mapping[str, Any] | None:
        """Record and forward one read-only lookup."""
        self.requests.append(request)
        return self.delegate.load(request)


@dataclass(frozen=True)
class ShadowEvidenceSummary:
    """Safe, non-PII summary of accepted shadow evidence."""

    turns_observed: int
    calls_scored: int
    successful_results: int
    generated_at: str
    synthetic: bool


@dataclass(frozen=True)
class GateRefusal:
    """Structured reason a gate run did not promote a candidate."""

    code: RefusalCode
    reasons: tuple[str, ...]
    task_id: str = TASK_ID

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable refusal summary for logs."""
        return {"code": self.code, "reasons": list(self.reasons), "task_id": self.task_id}


class ShadowEvidenceError(GateError):
    """Internal fail-closed signal carrying a stable refusal code."""

    def __init__(self, code: RefusalCode, reasons: Sequence[str] | str) -> None:
        """Initialise with a stable code and PII-free reason strings."""
        self.code = code
        self.reasons = (reasons,) if isinstance(reasons, str) else tuple(reasons)
        super().__init__("; ".join(self.reasons))


@dataclass(frozen=True)
class RefusalAuditRecord:
    """One immutable, hash-chained shadow-refusal audit entry."""

    task_id: str
    actor_id: str
    code: RefusalCode
    reasons: tuple[str, ...]
    champion_version: str
    challenger_version: str
    candidate_artifact_uri: str
    timestamp: str
    action: Literal["shadow_refusal"] = "shadow_refusal"
    previous_hash: str = ""
    entry_hash: str = ""

    def payload(self) -> dict[str, Any]:
        """Return the fields covered by ``entry_hash``."""
        data = asdict(self)
        data.pop("entry_hash")
        return data

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of the record."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RefusalAuditRecord:
        """Rebuild a record from its on-disk JSON object."""
        return cls(
            task_id=str(data["task_id"]),
            actor_id=str(data["actor_id"]),
            code=data["code"],
            reasons=tuple(str(reason) for reason in data["reasons"]),
            champion_version=str(data["champion_version"]),
            challenger_version=str(data["challenger_version"]),
            candidate_artifact_uri=str(data["candidate_artifact_uri"]),
            timestamp=str(data["timestamp"]),
            action=data.get("action", "shadow_refusal"),
            previous_hash=str(data.get("previous_hash", "")),
            entry_hash=str(data.get("entry_hash", "")),
        )


class RefusalSink(Protocol):
    """Append-only audit seam for shadow refusals."""

    def record_refusal(
        self,
        *,
        actor_id: str,
        code: RefusalCode,
        reasons: Sequence[str],
        champion_version: str,
        challenger_version: str,
        candidate_artifact_uri: str,
        task_id: str = TASK_ID,
    ) -> RefusalAuditRecord:
        """Append one refusal record and return it."""
        ...


def _new_refusal_record(
    *,
    actor_id: str,
    code: RefusalCode,
    reasons: Sequence[str],
    champion_version: str,
    challenger_version: str,
    candidate_artifact_uri: str,
    task_id: str,
) -> RefusalAuditRecord:
    """Build one validated refusal record without writing it."""
    for name, value in (
        ("actor_id", actor_id),
        ("champion_version", champion_version),
        ("challenger_version", challenger_version),
        ("candidate_artifact_uri", candidate_artifact_uri),
        ("task_id", task_id),
    ):
        _require_text(value, name)
    if code not in REFUSAL_CODES:
        raise ValidationError(f"Unknown refusal code: {code!r}")
    reason_tuple = tuple(reasons)
    if not reason_tuple or any(
        not isinstance(reason, str) or not reason.strip() for reason in reason_tuple
    ):
        raise ValidationError("A refusal requires at least one non-blank reason")
    return RefusalAuditRecord(
        task_id=task_id,
        actor_id=actor_id,
        code=code,
        reasons=reason_tuple,
        champion_version=champion_version,
        challenger_version=challenger_version,
        candidate_artifact_uri=candidate_artifact_uri,
        timestamp=utc_timestamp(),
    )


class PromotionRefusalLog:
    """Append-only JSONL refusal log with a SHA-256 entry chain."""

    def __init__(self, path: str | Path) -> None:
        """Initialise the log path; the file is created on first write."""
        self.path = resolve_path(path)

    def entries(self) -> tuple[RefusalAuditRecord, ...]:
        """Return all refusal records in file order, or an empty tuple."""
        if not self.path.exists():
            return ()
        found: list[RefusalAuditRecord] = []
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                decoded = json.loads(line)
            except ValueError as exc:
                raise ValidationError(f"Refusal log line {number} is not valid JSON") from exc
            if not isinstance(decoded, dict):
                raise ValidationError(f"Refusal log line {number} must be an object")
            found.append(RefusalAuditRecord.from_dict(decoded))
        return tuple(found)

    def head_hash(self) -> str:
        """Return the newest refusal hash, or an empty string when empty."""
        entries = self.entries()
        return entries[-1].entry_hash if entries else ""

    def record_refusal(
        self,
        *,
        actor_id: str,
        code: RefusalCode,
        reasons: Sequence[str],
        champion_version: str,
        challenger_version: str,
        candidate_artifact_uri: str,
        task_id: str = TASK_ID,
    ) -> RefusalAuditRecord:
        """Append one refusal record and return it with chain hashes."""
        record = _new_refusal_record(
            actor_id=actor_id,
            code=code,
            reasons=reasons,
            champion_version=champion_version,
            challenger_version=challenger_version,
            candidate_artifact_uri=candidate_artifact_uri,
            task_id=task_id,
        )
        record = replace(record, previous_hash=self.head_hash())
        record = replace(record, entry_hash=sha256_hex(canonical_json(record.payload())))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(record.to_dict()) + "\n")
        return record

    def verify_chain(self) -> tuple[str, ...]:
        """Return integrity errors; an empty tuple means the chain is intact."""
        errors: list[str] = []
        previous = ""
        for number, record in enumerate(self.entries(), start=1):
            if record.entry_hash != sha256_hex(canonical_json(record.payload())):
                errors.append(f"refusal record {number}: hash mismatch")
            if record.previous_hash != previous:
                errors.append(f"refusal record {number}: broken chain")
            previous = record.entry_hash
        return tuple(errors)


class CollectingRefusalLog:
    """Deterministic in-memory fake refusal sink for tests."""

    def __init__(self) -> None:
        """Start with an empty record list."""
        self.records: list[RefusalAuditRecord] = []

    def record_refusal(
        self,
        *,
        actor_id: str,
        code: RefusalCode,
        reasons: Sequence[str],
        champion_version: str,
        challenger_version: str,
        candidate_artifact_uri: str,
        task_id: str = TASK_ID,
    ) -> RefusalAuditRecord:
        """Append and return one validated refusal record."""
        record = _new_refusal_record(
            actor_id=actor_id,
            code=code,
            reasons=reasons,
            champion_version=champion_version,
            challenger_version=challenger_version,
            candidate_artifact_uri=candidate_artifact_uri,
            task_id=task_id,
        )
        self.records.append(record)
        return record


def verify_shadow_evidence(
    report: Mapping[str, Any] | None,
    request: ShadowEvidenceRequest,
    settings: ShadowEvidenceSettings,
    *,
    now: datetime | None = None,
) -> ShadowEvidenceSummary:
    """Validate one read-only shadow report or raise a structured refusal.

    Args:
        report: P3-07 internal comparison artifact, or None when unavailable.
        request: Version, artifact and timestamp context for this evaluation.
        settings: Freshness and evidence-minimum settings.
        now: Injectable UTC time for deterministic staleness checks.

    Returns:
        A PII-free summary of accepted evidence.

    Raises:
        ShadowEvidenceError: On missing, stale, non-synthetic, incomplete,
            promotion-claiming, mismatched or malformed evidence.

    """
    if not isinstance(report, Mapping):
        raise _fail("shadow_evidence_missing", "no shadow comparison report was supplied")

    _check_no_promotion_claim(report)
    if (
        report.get("artifact_type") != SHADOW_ARTIFACT_TYPE
        or report.get("comparison_schema_version") != SHADOW_SCHEMA_VERSION
    ):
        raise _fail("shadow_evidence_invalid", "unsupported shadow artifact type or version")
    for name in _REQUIRED_SECTIONS:
        if name in {"status", "generated_at", "report_uri"}:
            if not isinstance(report.get(name), str) or not str(report[name]).strip():
                raise _fail("shadow_evidence_incomplete", f"shadow report has no {name}")
        elif not isinstance(report.get(name), Mapping):
            raise _fail("shadow_evidence_incomplete", f"shadow report has no {name} object")

    provenance = report["provenance"]
    _check_synthetic_label(report, provenance)
    successful = _non_negative_int(
        provenance.get("successful_results"),
        "successful_results",
        "shadow_evidence_incomplete",
    )
    counts = _check_counts(report["sample_counts"], settings, successful)
    if successful < settings.min_successful_results:
        raise _fail("shadow_evidence_incomplete", "no successful shadow results were recorded")
    _check_metrics(report)
    _check_models(provenance.get("models"), request)
    _check_isolation(report["isolation"])
    _check_evidence_block(report)

    generated_at = _parse_timestamp(report["generated_at"], "generated_at")
    evaluated_at = _parse_timestamp(request.evaluated_at, "evaluated_at")
    now_dt = _as_utc(now if now is not None else utcnow())
    if generated_at > now_dt + timedelta(seconds=MAX_FUTURE_CLOCK_SKEW_SECONDS):
        raise _fail("shadow_evidence_stale", "shadow report is dated in the future")
    if now_dt - generated_at > timedelta(hours=settings.max_report_age_hours):
        raise _fail("shadow_evidence_stale", "shadow report exceeds the maximum allowed age")
    if generated_at < evaluated_at:
        raise _fail("shadow_evidence_stale", "shadow report predates the evaluated artifact report")

    return ShadowEvidenceSummary(
        turns_observed=counts["turns_observed"],
        calls_scored=counts["calls_scored"],
        successful_results=successful,
        generated_at=report["generated_at"],
        synthetic=True,
    )


def _check_no_promotion_claim(report: Mapping[str, Any]) -> None:
    """Reject any report that claims a gate, alias, threshold or promotion edit."""
    promotion = report.get("promotion")
    if report.get("promotion_decision") != SHADOW_PROMOTION_DECISION:
        raise _fail("shadow_evidence_promotion_claim", "shadow report claims a promotion decision")
    if not isinstance(promotion, Mapping):
        raise _fail("shadow_evidence_incomplete", "shadow report has no promotion object")
    if promotion.get("decision") != SHADOW_PROMOTION_DECISION:
        raise _fail("shadow_evidence_promotion_claim", "promotion block is not not_evaluated")
    if promotion.get("alias_writes") != 0 or promotion.get("champion_flips") != 0:
        raise _fail("shadow_evidence_promotion_claim", "shadow report claims an alias write")
    if promotion.get("thresholds_changed") is not False or promotion.get("gate_edits") != 0:
        raise _fail(
            "shadow_evidence_promotion_claim", "shadow report claims a gate or threshold edit"
        )


def _check_synthetic_label(report: Mapping[str, Any], provenance: Mapping[str, Any]) -> None:
    """Accept only explicitly synthetic, fixture-labelled evidence."""
    status = report.get("status")
    if not isinstance(status, str) or SYNTHETIC_STATUS_MARKER not in status.upper():
        raise _fail("shadow_evidence_not_synthetic", "shadow report is not synthetic-labelled")
    if provenance.get("synthetic") is not True or provenance.get("fixture_standin") is not True:
        raise _fail(
            "shadow_evidence_not_synthetic", "provenance does not mark synthetic fixture data"
        )
    for name in ("real_traffic", "network_calls", "audio_bytes_logged", "transcripts_logged"):
        if provenance.get(name) is not False:
            raise _fail("shadow_evidence_not_synthetic", f"provenance claims {name}")
    if provenance.get("join_key") != SHADOW_JOIN_KEY:
        raise _fail("shadow_evidence_incomplete", "shadow report does not join outcomes by call_id")


def _check_counts(
    counts: Mapping[str, Any],
    settings: ShadowEvidenceSettings,
    successful_results: int,
) -> dict[str, int]:
    """Require complete, non-negative and internally bounded sample counts."""
    checked: dict[str, int] = {}
    for name in _REQUIRED_COUNT_KEYS:
        checked[name] = _non_negative_int(counts.get(name), name, "shadow_evidence_incomplete")
    if checked["turns_observed"] < settings.min_turns_observed:
        raise _fail("shadow_evidence_incomplete", "too few shadow turns were observed")
    if checked["calls_scored"] < 1:
        raise _fail("shadow_evidence_incomplete", "no calls were scored in shadow mode")
    if checked["results_candidate"] < 1 or checked["results_champion"] < 1:
        raise _fail("shadow_evidence_incomplete", "one shadow side has no results")
    if checked["calls_scored"] > checked["turns_observed"]:
        raise _fail("shadow_evidence_incomplete", "calls_scored exceeds turns_observed")
    if checked["results_candidate"] > checked["turns_observed"]:
        raise _fail("shadow_evidence_incomplete", "candidate results exceed turns_observed")
    if checked["results_champion"] > checked["turns_observed"]:
        raise _fail("shadow_evidence_incomplete", "champion results exceed turns_observed")
    if successful_results > checked["results_candidate"] + checked["results_champion"]:
        raise _fail("shadow_evidence_incomplete", "successful results exceed total model results")
    return checked


def _check_metrics(report: Mapping[str, Any]) -> None:
    """Require all three outcome metrics on both sides and their deltas."""
    metrics = report["metrics"]
    deltas = report["deltas"]
    if set(metrics) != {"candidate", "champion"}:
        raise _fail("shadow_evidence_incomplete", "metrics must cover candidate and champion")
    if set(deltas) != set(_METRIC_NAMES):
        raise _fail("shadow_evidence_incomplete", "deltas do not cover outcome metrics")
    for side in ("candidate", "champion"):
        values = metrics[side]
        if not isinstance(values, Mapping) or set(values) != set(_METRIC_NAMES):
            raise _fail("shadow_evidence_incomplete", f"{side} metrics are incomplete")
        for value in values.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise _fail("shadow_evidence_incomplete", "shadow metrics must be numeric")
            if not 0.0 <= float(value) <= 1.0:
                raise _fail("shadow_evidence_incomplete", "shadow metrics are outside [0, 1]")
    for value in deltas.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _fail("shadow_evidence_incomplete", "shadow deltas must be numeric")
    if report["metric_semantics"].get("primary_metric") != SHADOW_PRIMARY_METRIC:
        raise _fail("shadow_evidence_invalid", "unexpected shadow primary metric")


def _check_models(models: Any, request: ShadowEvidenceRequest) -> None:
    """Bind the report to the exact candidate and champion artifacts."""
    if not isinstance(models, Sequence) or isinstance(models, (str, bytes)) or len(models) != 2:
        raise _fail("shadow_evidence_incomplete", "shadow provenance must list two models")
    by_source: dict[str, Mapping[str, Any]] = {}
    for model in models:
        if not isinstance(model, Mapping):
            raise _fail("shadow_evidence_incomplete", "shadow model entry is malformed")
        source = model.get("source")
        if source not in ("candidate", "champion") or source in by_source:
            raise _fail("shadow_evidence_incomplete", "shadow model sources are invalid")
        by_source[source] = model
    expected = {
        "candidate": request.candidate_artifact_uri,
        "champion": request.champion_artifact_uri,
    }
    for source, uri in expected.items():
        if by_source[source].get("artifact_uri") != uri:
            raise _fail("shadow_evidence_mismatch", f"shadow {source} artifact does not match")
        if by_source[source].get("synthetic") is not True:
            raise _fail("shadow_evidence_not_synthetic", f"shadow {source} model is not synthetic")


def _check_isolation(isolation: Mapping[str, Any]) -> None:
    """Require the P3-07 call-path isolation claims to be intact."""
    if isolation.get("models_invoked_by_call_path") is not False:
        raise _fail("shadow_evidence_invalid", "shadow report invokes models on the call path")
    if isolation.get("within_budget") is not True:
        raise _fail("shadow_evidence_invalid", "shadow mirror exceeded its budget")


def _check_evidence_block(report: Mapping[str, Any]) -> Mapping[str, Any]:
    """Check that the embedded evidence block agrees with the whole report."""
    evidence = report["shadow_evidence"]
    if (
        evidence.get("report_uri") != report["report_uri"]
        or not str(evidence.get("report_uri", "")).strip()
    ):
        raise _fail("shadow_evidence_mismatch", "embedded evidence does not reference this report")
    counts = report["sample_counts"]
    if evidence.get("turns_observed") != counts.get("turns_observed"):
        raise _fail("shadow_evidence_mismatch", "embedded turn count does not match sample counts")
    candidate_error = report["metrics"]["candidate"].get("outcome_error_rate")
    if evidence.get("outcome_error_rate") != candidate_error:
        raise _fail(
            "shadow_evidence_mismatch", "embedded error rate does not match candidate metric"
        )
    if evidence.get("synthetic_standin") is not True or evidence.get("join_key") != SHADOW_JOIN_KEY:
        raise _fail("shadow_evidence_not_synthetic", "embedded evidence lacks synthetic labelling")
    return evidence


def _parse_timestamp(value: Any, label: str) -> datetime:
    """Parse an RFC 3339 UTC timestamp without echoing its value in errors."""
    if not isinstance(value, str) or not value.strip():
        raise _fail("shadow_evidence_incomplete", f"shadow report has no {label}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _fail("shadow_evidence_incomplete", f"{label} is not a valid timestamp") from exc
    if parsed.tzinfo is None:
        raise _fail("shadow_evidence_incomplete", f"{label} is not UTC-aware")
    return parsed.astimezone(UTC)


def _as_utc(value: datetime) -> datetime:
    """Return a timezone-aware UTC datetime, treating naive inputs as UTC."""
    if not isinstance(value, datetime):
        return utcnow()
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _non_negative_int(value: Any, label: str, code: RefusalCode) -> int:
    """Return a non-negative integer, rejecting booleans and strings."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _fail(code, f"shadow {label} must be a non-negative integer")
    return value


def _require_text(value: Any, label: str) -> None:
    """Reject blank or non-string identifiers."""
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label} must be a non-blank string")


def _fail(code: RefusalCode, reason: str) -> ShadowEvidenceError:
    """Build one stable, PII-free shadow refusal."""
    return ShadowEvidenceError(code, (reason,))
