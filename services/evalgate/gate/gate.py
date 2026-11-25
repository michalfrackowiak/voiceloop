"""Converted-artifact evaluation and promotion orchestration for P2-09.

The gate never evaluates a training checkpoint: an injected
:class:`ArtifactConverter` produces the deployable artifact, and an injected
:class:`ArtifactEvaluator` returns the eval report for that exact artifact URI
(``context.md`` section 2 invariant 3). A report whose ``artifact_uri`` differs
is refused. P3-04 will supply the real converted-artifact runner; this module
ships only fakes and protocol hooks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

from voiceloop_common import ValidationError, validate

from .aliases import AliasStore, promote, rollback_promotion
from .audit import AuditLog, PromotionRecord
from .rules import GateDecision, GateError, decide_gate
from .settings import GateSettings

EVAL_REPORT_SCHEMA = "eval_report"


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
    ) -> None:
        """Inject settings, conversion/evaluation hooks, alias store and audit log."""
        self._settings = settings
        self._converter = converter
        self._evaluator = evaluator
        self._alias_store = alias_store
        self._audit_log = audit_log

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
    ) -> GateResult:
        """Evaluate the converted artifact and flip aliases only on a pass.

        Args:
            source_artifact_uri: Checkpoint or trainer artifact URI to convert.
            champion_report: Report for the current champion artifact.
            challenger_version: MLflow version whose artifact was evaluated.
            actor_id: Human or service identity authorising a promotion.
            reason: Human-readable justification for the promotion.

        Returns:
            A :class:`GateResult`; ``promotion`` is None when the gate fails.

        """
        converted, decision, _ = self._evaluate(source_artifact_uri, champion_report)
        if not decision.passed:
            return GateResult(decision=decision, converted_artifact_uri=converted.artifact_uri)
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
        )

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
