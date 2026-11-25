"""P3-12.2 STT failure injections: fail closed before any promotion.

Each injection uses only public P3-04/P3-07/P3-08 seams and in-memory fakes.
Refusals carry stable reason codes, are hash-chained and are labelled with the
mandatory STT stand-in marker; no fixture PII or raw transcript reaches a log.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from services.evalgate.artifact import (
    EXPECTED_FIXTURE_SHA256,
    EXPECTED_GOLDEN_CONTENT_HASH,
    ArtifactBehaviour,
    evaluate_checkpoint_uris_do_not_change_result,
)
from services.evalgate.gate import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    CollectingShadowEvidenceSource,
    Gate,
    GateSettings,
    InMemoryAliasStore,
    PrefixArtifactConverter,
    PromotionAuditLog,
    ShadowEvidenceRequest,
    ShadowEvidenceSettings,
    StaticReportEvaluator,
    StaticShadowEvidenceSource,
    decide_gate,
)

from .helpers import (
    STT_CHALLENGER_VERSION,
    STT_CHAMPION_VERSION,
    STT_RELEASE_LABEL,
    SYNTHETIC_REVIEWER_ID,
    assert_labelled_outputs,
    assert_no_pii_text,
    build_synthetic_shadow_evidence,
    evaluate_stt_artifact,
    evaluate_stt_promotion_reports,
    make_labelled_refusal_log,
    stt_artifact_settings,
    write_derived_eval_fixture,
)

CHECKPOINT_URI = "fake://trainer-artifacts/whisper-ctranslate2/synthetic-checkpoint"
CONVERTED_URI = "converted://trainer-artifacts/whisper-ctranslate2/synthetic-checkpoint"
CHAMPION_URI = "fake://evalgate/champion-synthetic-fixture-standin"


ReportMutation = Callable[[dict[str, Any]], None]


class MutableShadowSource:
    """Public-protocol fake whose evidence can be replaced between gate runs."""

    def __init__(self, report: dict[str, Any]) -> None:
        """Store one deep copy of the initial synthetic-labelled report."""
        self.report = deepcopy(report)
        self.requests: list[ShadowEvidenceRequest] = []

    def load(self, request: ShadowEvidenceRequest) -> dict[str, Any] | None:
        """Record one read-only lookup and return a copy of current evidence."""
        self.requests.append(request)
        return deepcopy(self.report)


def _mutate_stale(report: dict[str, Any]) -> None:
    """Back-date the comparison so it predates the evaluated artifact."""
    report["generated_at"] = "2020-01-01T00:00:00Z"


def _mutate_not_synthetic(report: dict[str, Any]) -> None:
    """Strip the synthetic/fixture status label and provenance claim."""
    report["status"] = "PRODUCTION-CLAIM"
    report["provenance"]["synthetic"] = False
    report["provenance"]["fixture_standin"] = False


def _mutate_promotion_claim(report: dict[str, Any]) -> None:
    """Make the evidence claim an alias write it is forbidden to perform."""
    report["promotion"]["alias_writes"] = 1


def _mutate_mismatch(report: dict[str, Any]) -> None:
    """Bind the evidence to a different candidate artifact URI."""
    report["provenance"]["models"][0]["artifact_uri"] = "converted://other/synthetic"


def _mutate_incomplete(report: dict[str, Any]) -> None:
    """Drop all scored calls from the evidence counts."""
    report["sample_counts"]["calls_scored"] = 0


def _mutate_invalid(report: dict[str, Any]) -> None:
    """Replace the artifact type with an unsupported value."""
    report["artifact_type"] = "not-a-shadow-comparison"


def _make_gate(
    tmp_path: Path,
    *,
    challenger_report: dict[str, Any],
    champion_report: dict[str, Any],
    shadow_report: dict[str, Any] | None,
    shadow_source_configured: bool = True,
) -> tuple[Gate, InMemoryAliasStore, PromotionAuditLog, Any, Any]:
    """Wire a consumer-only STT gate with labelled refusal auditing."""
    aliases = InMemoryAliasStore(
        {
            CHAMPION_ALIAS: STT_CHAMPION_VERSION,
            CHALLENGER_ALIAS: STT_CHALLENGER_VERSION,
        }
    )
    audit = PromotionAuditLog(tmp_path / "audit.jsonl")
    refusal_log = make_labelled_refusal_log(tmp_path / "refusals.jsonl")
    source = (
        CollectingShadowEvidenceSource(
            StaticShadowEvidenceSource(
                {(STT_CHALLENGER_VERSION, STT_CHAMPION_VERSION): shadow_report}
                if shadow_report is not None
                else {}
            )
        )
        if shadow_source_configured
        else None
    )
    gate = Gate(
        settings=GateSettings(),
        converter=PrefixArtifactConverter(),
        evaluator=StaticReportEvaluator({CONVERTED_URI: challenger_report}),
        alias_store=aliases,
        audit_log=audit,
        shadow_source=source,
        shadow_settings=ShadowEvidenceSettings(refusal_log_path=str(tmp_path / "refusals.jsonl")),
        refusal_log=refusal_log,
    )
    return gate, aliases, audit, source, refusal_log


def _assert_no_promotion(
    aliases: InMemoryAliasStore,
    audit: PromotionAuditLog,
    *,
    entries_before: tuple[Any, ...],
    head_before: str,
) -> None:
    """Assert no alias flip or audit write escaped a refusal."""
    assert aliases.get(CHAMPION_ALIAS) == STT_CHAMPION_VERSION
    assert aliases.get(CHALLENGER_ALIAS) == STT_CHALLENGER_VERSION
    assert audit.entries() == entries_before
    assert audit.head_hash() == head_before
    assert audit.verify_chain() == ()


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        pytest.param(_mutate_stale, "shadow_evidence_stale", id="stale"),
        pytest.param(_mutate_not_synthetic, "shadow_evidence_not_synthetic", id="non-synthetic"),
        pytest.param(_mutate_promotion_claim, "shadow_evidence_promotion_claim", id="claim"),
        pytest.param(_mutate_mismatch, "shadow_evidence_mismatch", id="mismatch"),
        pytest.param(_mutate_incomplete, "shadow_evidence_incomplete", id="incomplete"),
        pytest.param(_mutate_invalid, "shadow_evidence_invalid", id="invalid"),
    ],
)
def test_shadow_evidence_injections_refuse_promotion(
    tmp_path: Path,
    fixture_rows: list[dict[str, Any]],
    mutate: ReportMutation,
    expected_code: str,
) -> None:
    """Every malformed shadow report produces a structured, chained refusal."""
    derived = write_derived_eval_fixture(tmp_path / "derived.jsonl", fixture_rows)
    champion_report, challenger_report = evaluate_stt_promotion_reports(
        derived, champion_uri=CHAMPION_URI, challenger_uri=CONVERTED_URI
    )
    shadow = build_synthetic_shadow_evidence(
        tmp_path / "shadow-valid",
        fixture_rows=fixture_rows,
        candidate_artifact_uri=CONVERTED_URI,
        champion_artifact_uri=CHAMPION_URI,
    )
    bad_report = deepcopy(shadow.report)
    mutate(bad_report)
    gate, aliases, audit, source, refusal_log = _make_gate(
        tmp_path,
        challenger_report=challenger_report,
        champion_report=champion_report,
        shadow_report=bad_report,
    )
    entries_before = audit.entries()
    head_before = audit.head_hash()

    result = gate.run(
        CHECKPOINT_URI,
        champion_report,
        challenger_version=STT_CHALLENGER_VERSION,
        actor_id=SYNTHETIC_REVIEWER_ID,
        reason=f"{STT_RELEASE_LABEL}; refusal rehearsal",
        change_kind="stt",
    )

    assert not result.decision.passed
    assert result.promotion is None
    assert result.refusal is not None
    assert result.refusal.code == expected_code
    assert result.shadow_evidence is None
    assert source is not None and len(source.requests) == 1
    _assert_no_promotion(aliases, audit, entries_before=entries_before, head_before=head_before)
    records = refusal_log.delegate.entries()
    assert len(records) == 1
    record = records[0]
    assert record.code == expected_code
    assert record.reasons[0] == STT_RELEASE_LABEL
    assert refusal_log.delegate.verify_chain() == ()
    assert STT_RELEASE_LABEL in refusal_log.delegate.path.read_text(encoding="utf-8")
    assert_no_pii_text(refusal_log.delegate.path.read_text(encoding="utf-8"))


def test_missing_shadow_evidence_and_no_source_both_refuse(
    tmp_path: Path,
    fixture_rows: list[dict[str, Any]],
) -> None:
    """A missing report and an unconfigured source are equally fail-closed."""
    derived = write_derived_eval_fixture(tmp_path / "derived.jsonl", fixture_rows)
    champion_report, challenger_report = evaluate_stt_promotion_reports(
        derived, champion_uri=CHAMPION_URI, challenger_uri=CONVERTED_URI
    )
    for index, configured in enumerate((True, False)):
        case_dir = tmp_path / f"case-{index}"
        gate, aliases, audit, _, refusal_log = _make_gate(
            case_dir,
            challenger_report=challenger_report,
            champion_report=champion_report,
            shadow_report=None,
            shadow_source_configured=configured,
        )
        entries_before = audit.entries()
        head_before = audit.head_hash()
        result = gate.run(
            CHECKPOINT_URI,
            champion_report,
            challenger_version=STT_CHALLENGER_VERSION,
            actor_id=SYNTHETIC_REVIEWER_ID,
            reason=f"{STT_RELEASE_LABEL}; missing-evidence rehearsal",
            change_kind="stt",
        )
        assert result.refusal is not None
        assert result.refusal.code == "shadow_evidence_missing"
        assert result.promotion is None
        _assert_no_promotion(aliases, audit, entries_before=entries_before, head_before=head_before)
        assert [record.code for record in refusal_log.delegate.entries()] == [
            "shadow_evidence_missing"
        ]
        assert refusal_log.delegate.verify_chain() == ()


def test_refusal_log_chain_spans_multiple_injections(
    tmp_path: Path,
    fixture_rows: list[dict[str, Any]],
) -> None:
    """Three refusals form one verified hash chain with labelled reason codes."""
    derived = write_derived_eval_fixture(tmp_path / "derived.jsonl", fixture_rows)
    champion_report, challenger_report = evaluate_stt_promotion_reports(
        derived, champion_uri=CHAMPION_URI, challenger_uri=CONVERTED_URI
    )
    shadow = build_synthetic_shadow_evidence(
        tmp_path / "shadow-valid",
        fixture_rows=fixture_rows,
        candidate_artifact_uri=CONVERTED_URI,
        champion_artifact_uri=CHAMPION_URI,
    )
    source = MutableShadowSource(shadow.report)
    aliases = InMemoryAliasStore(
        {
            CHAMPION_ALIAS: STT_CHAMPION_VERSION,
            CHALLENGER_ALIAS: STT_CHALLENGER_VERSION,
        }
    )
    audit = PromotionAuditLog(tmp_path / "chain" / "audit.jsonl")
    refusal_log = make_labelled_refusal_log(tmp_path / "chain" / "refusals.jsonl")
    gate = Gate(
        settings=GateSettings(),
        converter=PrefixArtifactConverter(),
        evaluator=StaticReportEvaluator({CONVERTED_URI: challenger_report}),
        alias_store=aliases,
        audit_log=audit,
        shadow_source=source,
        shadow_settings=ShadowEvidenceSettings(
            refusal_log_path=str(tmp_path / "chain" / "refusals.jsonl")
        ),
        refusal_log=refusal_log,
    )
    for index, mutate in enumerate((_mutate_stale, _mutate_promotion_claim, _mutate_invalid)):
        report = deepcopy(shadow.report)
        mutate(report)
        source.report = report
        result = gate.run(
            CHECKPOINT_URI,
            champion_report,
            challenger_version=STT_CHALLENGER_VERSION,
            actor_id=SYNTHETIC_REVIEWER_ID,
            reason=f"{STT_RELEASE_LABEL}; chained refusal {index}",
            change_kind="stt",
        )
        assert result.refusal is not None
        assert result.promotion is None
    assert len(source.requests) == 3
    _assert_no_promotion(aliases, audit, entries_before=(), head_before="")
    records = refusal_log.delegate.entries()
    assert [record.code for record in records] == [
        "shadow_evidence_stale",
        "shadow_evidence_promotion_claim",
        "shadow_evidence_invalid",
    ]
    assert [record.previous_hash for record in records[1:]] == [
        records[0].entry_hash,
        records[1].entry_hash,
    ]
    assert refusal_log.delegate.verify_chain() == ()
    assert all(STT_RELEASE_LABEL in record.reasons for record in records)


def test_valid_shadow_alone_cannot_promote_a_regressing_report(
    tmp_path: Path,
    fixture_rows: list[dict[str, Any]],
) -> None:
    """Valid shadow evidence never overrides a failing P2-09 e2e decision."""
    derived = write_derived_eval_fixture(tmp_path / "derived.jsonl", fixture_rows)
    champion_report, _ = evaluate_stt_promotion_reports(
        derived, champion_uri=CHAMPION_URI, challenger_uri=CONVERTED_URI
    )
    _, regressing = evaluate_stt_promotion_reports(
        derived,
        champion_uri=CHAMPION_URI,
        challenger_uri=CONVERTED_URI,
        challenger_artifact_kind="identity",
    )
    assert not decide_gate(champion_report, regressing, GateSettings()).passed
    shadow = build_synthetic_shadow_evidence(
        tmp_path / "shadow-valid",
        fixture_rows=fixture_rows,
        candidate_artifact_uri=CONVERTED_URI,
        champion_artifact_uri=CHAMPION_URI,
    )
    gate, aliases, audit, _, refusal_log = _make_gate(
        tmp_path / "gate",
        challenger_report=regressing,
        champion_report=champion_report,
        shadow_report=shadow.report,
    )
    entries_before = audit.entries()
    head_before = audit.head_hash()
    result = gate.run(
        CHECKPOINT_URI,
        champion_report,
        challenger_version=STT_CHALLENGER_VERSION,
        actor_id=SYNTHETIC_REVIEWER_ID,
        reason=f"{STT_RELEASE_LABEL}; regression must not promote",
        change_kind="stt",
    )
    assert not result.decision.passed
    assert result.promotion is None
    assert result.refusal is None
    assert result.shadow_evidence is not None
    _assert_no_promotion(aliases, audit, entries_before=entries_before, head_before=head_before)
    assert refusal_log.delegate.entries() == ()


@pytest.mark.parametrize(
    ("behaviour", "expected_code"),
    [
        ("empty", "empty_output"),
        ("truncated", "truncated_output"),
        ("hallucinating", "silence_hallucination"),
        ("forgetting", "multilingual_forgetting"),
    ],
)
def test_artifact_injections_fail_closed_before_promotion(
    tmp_path: Path,
    behaviour: ArtifactBehaviour,
    expected_code: str,
) -> None:
    """P3-04 probe failures are labelled, structured and never promote."""
    settings = stt_artifact_settings(
        tmp_path / behaviour,
        checkpoint_uri=CHECKPOINT_URI,
        behaviour=behaviour,
    )
    _, evaluation = evaluate_stt_artifact(settings, behaviour)
    assert evaluation.probe_suite.status == "fail"
    assert expected_code in evaluation.probe_suite.summary()["errors_by_code"]
    assert evaluation.gate_summary["outcome"] == "fail"
    assert evaluation.report["gate_decision"]["outcome"] == "fail"
    assert STT_RELEASE_LABEL in str(evaluation.report["gate_decision"]["reason"])
    if behaviour == "hallucinating":
        assert evaluation.hallucination["rate"] > 0.0
    if behaviour == "forgetting":
        assert evaluation.multilingual["forgetting_rate"] > 0.0
    assert_labelled_outputs(
        (
            Path(settings.output_path),
            Path(settings.output_path).with_name("stt_champion_report.json"),
        )
    )


def test_checkpoint_uri_substitution_cannot_change_evaluated_artifact(
    tmp_path: Path,
) -> None:
    """A different checkpoint URI still evaluates the same content-addressed artifact."""
    settings = stt_artifact_settings(tmp_path, checkpoint_uri=CHECKPOINT_URI)
    artifact, _ = evaluate_stt_artifact(settings, "healthy")
    assert EXPECTED_FIXTURE_SHA256 == settings.expected_fixture_sha256
    assert EXPECTED_GOLDEN_CONTENT_HASH == settings.expected_golden_content_hash
    assert evaluate_checkpoint_uris_do_not_change_result(
        settings, artifact, "fake://trainer-artifacts/whisper-ctranslate2/other-checkpoint"
    )


def test_parser_path_keeps_p2_09_behaviour_without_shadow_source(
    tmp_path: Path,
    fixture_rows: list[dict[str, Any]],
) -> None:
    """P3-08 is additive: parser promotions never require shadow evidence."""
    derived = write_derived_eval_fixture(tmp_path / "derived.jsonl", fixture_rows)
    champion_report, challenger_report = evaluate_stt_promotion_reports(
        derived, champion_uri=CHAMPION_URI, challenger_uri=CONVERTED_URI
    )
    aliases = InMemoryAliasStore({CHAMPION_ALIAS: "v1-parser", CHALLENGER_ALIAS: "v2-parser"})
    audit = PromotionAuditLog(tmp_path / "audit.jsonl")
    gate = Gate(
        settings=GateSettings(),
        converter=PrefixArtifactConverter(),
        evaluator=StaticReportEvaluator({CONVERTED_URI: challenger_report}),
        alias_store=aliases,
        audit_log=audit,
    )
    result = gate.run(
        CHECKPOINT_URI,
        champion_report,
        challenger_version="v2-parser",
        actor_id=SYNTHETIC_REVIEWER_ID,
        reason=f"{STT_RELEASE_LABEL}; parser path unchanged",
    )
    assert result.decision.passed
    assert result.promotion is not None
    assert result.refusal is None
    assert result.shadow_evidence is None
    assert aliases.get(CHAMPION_ALIAS) == "v2-parser"
    assert audit.verify_chain() == ()
