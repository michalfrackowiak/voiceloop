"""Converted-artifact evaluation and promotion orchestration.

The gate never evaluates a training checkpoint: an injected
:class:`ArtifactConverter` produces the deployable artifact, and an injected
:class:`ArtifactEvaluator` returns the eval report for that exact artifact URI
(``context.md`` section 2 invariant 3). A report whose ``artifact_uri`` differs
is refused. P3-04 will supply the real converted-artifact runner; this module
ships only fakes and protocol hooks.

P3-08 adds a read-only shadow-evidence precondition for STT promotions. It is
additive: the P2-09 paired e2e accuracy rule and critical-slice checks still
apply, parser promotions keep their original behaviour, and a shadow refusal
never flips an alias or changes a threshold.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol

from voiceloop_common import Clock, ValidationError, validate

from .aliases import CHAMPION_ALIAS, AliasStore, promote, rollback_promotion
from .audit import AuditLog, PromotionRecord
from .rules import GateDecision, GateError, decide_gate
from .settings import GateSettings
from .shadow import (
    TASK_ID,
    GateRefusal,
    PromotionRefusalLog,
    RefusalSink,
    ShadowEvidenceError,
    ShadowEvidenceRequest,
    ShadowEvidenceSettings,
    ShadowEvidenceSource,
    ShadowEvidenceSummary,
    verify_shadow_evidence,
)

EVAL_REPORT_SCHEMA = "eval_report"
ChangeKind = Literal["parser", "stt"]
CHANGE_KINDS: frozenset[str] = frozenset({"parser", "stt"})


@dataclass(frozen=True)
class ConvertedArtifact:
    """The deployable form of a model, produced by an injected converter."""

    artifact_uri: str
    source_uri: str

    def __post_init__(self) -> None:
        """Reject blank URIs before they can reach an evaluator."""
        if not self.artifact_uri.strip() or not self.source_uri.strip():
            raise ValidationError("ConvertedArtifact URIs must be non-blank strings")


class ArtifactConverter(Protocol):
    """Hook for the conversion step; P3-04 supplies the real implementation."""

    def convert(self, source_uri: str) -> ConvertedArtifact:
        """Convert a checkpoint URI into the artifact that will actually serve."""
        ...


class ArtifactEvaluator(Protocol):
    """Hook for the evaluation runner; P3-04 supplies the real implementation."""

    def evaluate(self, artifact_uri: str, eval_sets: Sequence[str]) -> Mapping[str, Any]:
        """Return a contract-valid eval report for the given artifact URI."""
        ...


@dataclass(frozen=True)
class GateResult:
    """Outcome of one gate run, including any alias promotion performed."""

    decision: GateDecision
    converted_artifact_uri: str
    promotion: PromotionRecord | None = None
    refusal: GateRefusal | None = None
    shadow_evidence: ShadowEvidenceSummary | None = None


class PrefixArtifactConverter:
    """Fake converter mapping a source URI to a distinct converted URI."""

    def __init__(self, prefix: str = "converted://") -> None:
        """Initialise with the URI prefix used for fake converted artifacts."""
        if not prefix.strip():
            raise ValidationError("converter prefix must be non-blank")
        self._prefix = prefix

    def convert(self, source_uri: str) -> ConvertedArtifact:
        """Return a converted artifact URI derived from the source URI."""
        if not isinstance(source_uri, str) or not source_uri.strip():
            raise ValidationError("source_uri must be a non-blank string")
        path = source_uri.split("://", 1)[-1]
        return ConvertedArtifact(artifact_uri=f"{self._prefix}{path}", source_uri=source_uri)


class StaticReportEvaluator:
    """Fake evaluator returning precomputed, contract-valid reports by artifact URI."""

    def __init__(self, reports: Mapping[str, Mapping[str, Any]]) -> None:
        """Index reports by the exact artifact URI they were produced from."""
        self._reports: dict[str, Mapping[str, Any]] = dict(reports)

    def evaluate(self, artifact_uri: str, eval_sets: Sequence[str]) -> Mapping[str, Any]:
        """Return a validated copy of the report for ``artifact_uri``.

        ``eval_sets`` is accepted for interface parity with the evaluation
        contract; the fakes carry reports that were already built for those sets.
        """
        report = self._reports.get(artifact_uri)
        if report is None:
            raise GateError(f"no evaluation report for artifact {artifact_uri!r}")
        validate(report, EVAL_REPORT_SCHEMA)
        return deepcopy(dict(report))


class Gate:
    """Evaluate a converted challenger and promote only when the rules pass."""

    def __init__(
        self,
        *,
        settings: GateSettings,
        converter: ArtifactConverter,
        evaluator: ArtifactEvaluator,
        alias_store: AliasStore,
        audit_log: AuditLog,
        shadow_source: ShadowEvidenceSource | None = None,
        shadow_settings: ShadowEvidenceSettings | None = None,
        refusal_log: RefusalSink | None = None,
        clock: Clock | None = None,
    ) -> None:
        """Inject evaluation hooks, alias store, audits and shadow seams.

        The shadow args are optional so the P2-09 parser path is unchanged. STT
        runs must supply ``shadow_source``; evidence is read-only.
        """
        self._settings = settings
        self._converter = converter
        self._evaluator = evaluator
        self._alias_store = alias_store
        self._audit_log = audit_log
        self._shadow_source = shadow_source
        self._shadow_settings = shadow_settings or ShadowEvidenceSettings()
        self._refusal_log = refusal_log or PromotionRefusalLog(
            self._shadow_settings.refusal_log_path
        )
        self._clock = clock

    def evaluate(
        self, source_artifact_uri: str, champion_report: Mapping[str, Any]
    ) -> tuple[GateDecision, Mapping[str, Any]]:
        """Convert, evaluate and decide without changing aliases.

        Raises:
            GateError: If the returned report describes a different artifact.

        """
        _, decision, report = self._evaluate(source_artifact_uri, champion_report)
        return decision, report

    def run(
        self,
        source_artifact_uri: str,
        champion_report: Mapping[str, Any],
        *,
        challenger_version: str,
        actor_id: str,
        reason: str,
        change_kind: ChangeKind = "parser",
    ) -> GateResult:
        """Evaluate the converted artifact and flip aliases only on a pass.

        Args:
            source_artifact_uri: Checkpoint or trainer artifact URI to convert.
            champion_report: Report for the current champion artifact.
            challenger_version: MLflow version whose artifact was evaluated.
            actor_id: Human or service identity authorising a promotion.
            reason: Human-readable justification for the promotion.
            change_kind: ``"stt"`` requires valid shadow evidence; ``"parser"``
                keeps the P2-09 behaviour. Callers must label STT changes.

        Returns:
            A :class:`GateResult`; ``promotion`` is None when the gate fails.

        Raises:
            GateError: If ``change_kind`` is unknown or refusal auditing fails.

        """
        if change_kind not in CHANGE_KINDS:
            raise GateError(f"unknown change kind: {change_kind!r}")

        converted, decision, challenger_report = self._evaluate(
            source_artifact_uri, champion_report
        )
        champion_artifact_uri = champion_report.get("artifact_uri")
        if not isinstance(champion_artifact_uri, str) or not champion_artifact_uri.strip():
            raise GateError("champion report has no artifact_uri")

        summary: ShadowEvidenceSummary | None = None
        refusal: GateRefusal | None = None
        if change_kind == "stt":
            request = ShadowEvidenceRequest(
                candidate_version=challenger_version,
                champion_version=str(self._alias_store.get(CHAMPION_ALIAS) or ""),
                candidate_artifact_uri=converted.artifact_uri,
                champion_artifact_uri=champion_artifact_uri,
                evaluated_at=str(challenger_report.get("generated_at") or ""),
            )
            summary, refusal = self._check_shadow(request, actor_id=actor_id)

        if not decision.passed or refusal is not None:
            reasons = list(decision.reasons)
            if refusal is not None:
                reasons.extend(refusal.reasons)
            return GateResult(
                decision=replace(decision, passed=False, reasons=tuple(reasons)),
                converted_artifact_uri=converted.artifact_uri,
                refusal=refusal,
                shadow_evidence=summary,
            )

        promotion = promote(
            self._alias_store,
            challenger_version,
            actor_id=actor_id,
            reason=reason,
            audit_log=self._audit_log,
        )
        return GateResult(
            decision=decision,
            converted_artifact_uri=converted.artifact_uri,
            promotion=promotion,
            shadow_evidence=summary,
        )

    def _check_shadow(
        self, request: ShadowEvidenceRequest, *, actor_id: str
    ) -> tuple[ShadowEvidenceSummary | None, GateRefusal | None]:
        """Load and verify read-only shadow evidence, auditing any refusal."""
        report: Mapping[str, Any] | None = None
        try:
            if self._shadow_source is None:
                raise ShadowEvidenceError(
                    "shadow_evidence_missing", "no shadow evidence source is configured for STT"
                )
            report = self._shadow_source.load(request)
            summary = verify_shadow_evidence(
                report,
                request,
                self._shadow_settings,
                now=self._clock() if self._clock is not None else None,
            )
        except ShadowEvidenceError as exc:
            refusal = GateRefusal(code=exc.code, reasons=exc.reasons, task_id=TASK_ID)
        except Exception:
            refusal = GateRefusal(
                code="shadow_evidence_invalid",
                reasons=("shadow evidence source failed; no promotion is allowed",),
                task_id=TASK_ID,
            )
        else:
            return summary, None

        self._refusal_log.record_refusal(
            actor_id=actor_id,
            code=refusal.code,
            reasons=refusal.reasons,
            champion_version=request.champion_version,
            challenger_version=request.candidate_version,
            candidate_artifact_uri=request.candidate_artifact_uri,
            task_id=refusal.task_id,
        )
        return None, refusal

    def rollback(
        self, promotion: PromotionRecord, *, actor_id: str, reason: str
    ) -> PromotionRecord:
        """Restore the previous champion in one step and audit the rollback."""
        return rollback_promotion(
            self._alias_store,
            promotion,
            actor_id=actor_id,
            reason=reason,
            audit_log=self._audit_log,
        )

    def _evaluate(
        self, source_artifact_uri: str, champion_report: Mapping[str, Any]
    ) -> tuple[ConvertedArtifact, GateDecision, Mapping[str, Any]]:
        """Convert, evaluate and check that the report covers the converted artifact."""
        converted = self._converter.convert(source_artifact_uri)
        report = self._evaluator.evaluate(converted.artifact_uri, self._settings.eval_sets)
        validate(report, EVAL_REPORT_SCHEMA)
        reported_uri = report.get("artifact_uri")
        if reported_uri != converted.artifact_uri:
            raise GateError(
                f"report artifact_uri {reported_uri!r} is not the converted artifact "
                f"{converted.artifact_uri!r}"
            )
        return converted, decide_gate(champion_report, report, self._settings), report
