"""Contract-valid STT reports, exact-artifact gate and fail-closed probes."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from services.evalgate.artifact.artifact import artifact_hash_from_uri, create_fake_artifact
from services.evalgate.artifact.decoding import decoding_hash_from_fields
from services.evalgate.artifact.report import (
    EXTENSION_KEY,
    EXTRA_METRICS,
    ArtifactEvaluation,
    ExactArtifactConverter,
    HashedReportEvaluator,
    evaluate_artifact,
    evaluate_checkpoint_uris_do_not_change_result,
    run_exact_artifact_gate,
    validate_stt_report,
    write_stt_report,
)
from services.evalgate.artifact.settings import ArtifactBehaviour, ArtifactSettings
from services.evalgate.gate import GateError
from voiceloop_common import ValidationError, validate


def _evaluate(
    settings: ArtifactSettings, behaviour: ArtifactBehaviour, languages: str = "en"
) -> ArtifactEvaluation:
    """Evaluate one injected fake artifact with deterministic bootstrap settings."""
    run_settings = settings.model_copy(update={"served_languages_csv": languages})
    return evaluate_artifact(run_settings, create_fake_artifact(behaviour))


@pytest.mark.parametrize(
    ("behaviour", "languages"),
    [
        ("healthy", "en"),
        ("empty", "en"),
        ("truncated", "en"),
        ("hallucinating", "en"),
        ("forgetting", "en,es"),
    ],
)
def test_every_stt_report_carries_hallucination_and_multilingual_evidence(
    settings: ArtifactSettings, behaviour: ArtifactBehaviour, languages: str
) -> None:
    """Both mandatory diagnostics appear in every exposed report, for every behaviour."""
    artifact = create_fake_artifact(behaviour)
    run_settings = settings.model_copy(update={"served_languages_csv": languages})
    evaluation = evaluate_artifact(run_settings, artifact)

    for candidate_report in (evaluation.report, evaluation.champion_report):
        validate(candidate_report, "eval_report")
        validate_stt_report(candidate_report)
        assert candidate_report["eval_sets"] == ["general_capability"]
        for metric in EXTRA_METRICS:
            value = candidate_report["metrics"][metric]
            assert isinstance(value, (int, float)) and not isinstance(value, bool)
            interval = candidate_report["confidence_intervals"][metric]
            assert interval["level"] == run_settings.bootstrap_level
        assert "hallucination_rate" in candidate_report["slice_metrics"]["silence"]
        assert "forgetting_rate" in candidate_report["slice_metrics"]["other_languages"]
        assert "turns" in candidate_report["slice_metrics"]["silence"]
        assert "turns" in candidate_report["slice_metrics"]["other_languages"]

        extension = candidate_report["decoding_settings"][EXTENSION_KEY]
        assert extension["hallucination"]["promotion_metric"] is False
        assert extension["multilingual_forgetting"]["gate_evidence"] is False
        assert extension["decoding"]["sha256"] == decoding_hash_from_fields(
            candidate_report["decoding_settings"]
        )
        assert candidate_report["gate_decision"]["outcome"] in {"fail", "not_evaluated"}

    assert evaluation.report["artifact_uri"] == artifact.uri
    assert evaluation.champion_report["artifact_uri"] == create_fake_artifact("healthy").uri
    assert evaluation.gate_summary["status"] == "evaluated"
    assert evaluation.gate_summary["evaluated_artifact_uri"] == artifact.uri
    assert evaluation.gate_summary["source_checkpoint_uri"] == run_settings.checkpoint_uri


def test_champion_report_is_a_fully_validated_stt_report(settings: ArtifactSettings) -> None:
    """The returned healthy baseline is not a diagnostics-free second-class report."""
    run_settings = settings.model_copy(update={"served_languages_csv": "en,es"})
    evaluation = evaluate_artifact(run_settings, create_fake_artifact("forgetting"))
    champion = evaluation.champion_report
    healthy = create_fake_artifact("healthy")

    validate(champion, "eval_report")
    validate_stt_report(champion)
    assert champion["artifact_uri"] == healthy.uri
    assert champion["metrics"]["silence_hallucination_rate"] == 0.0
    assert champion["metrics"]["multilingual_forgetting_rate"] == 0.0
    assert champion["slice_metrics"]["silence"]["hallucination_rate"] == 0.0
    assert champion["slice_metrics"]["other_languages"]["forgetting_rate"] == 0.0
    assert champion["gate_decision"]["outcome"] == "not_evaluated"

    extension = champion["decoding_settings"][EXTENSION_KEY]
    assert extension["artifact"]["sha256"] == healthy.sha256
    assert extension["probes"]["status"] == "pass"
    assert extension["hallucination"]["promotion_metric"] is False
    assert extension["multilingual_forgetting"]["gate_evidence"] is False
    assert extension["gate"]["status"] == "not_applicable_healthy_fake_baseline"


def test_hashed_evaluator_rejects_tampered_decoding_fields(
    settings: ArtifactSettings,
) -> None:
    """A clean artifact URI cannot rescue tampered decoding settings at the seam."""
    clean = _evaluate(settings, "healthy").report
    tampered: dict[str, Any] = deepcopy(clean)
    tampered["decoding_settings"]["initial_prompt"] = "changed"
    artifact = create_fake_artifact("healthy")

    assert artifact_hash_from_uri(tampered["artifact_uri"]) == artifact.sha256
    evaluator = HashedReportEvaluator(artifact, tampered)
    with pytest.raises(ValidationError, match="decoding settings hash"):
        evaluator.evaluate(tampered["artifact_uri"], ("general_capability",))


def test_exact_gate_rejects_tampered_decoding_fields(settings: ArtifactSettings) -> None:
    """The public gate seam validates the challenger before any P2-09 rule runs."""
    evaluation = _evaluate(settings, "healthy")
    tampered: dict[str, Any] = deepcopy(evaluation.report)
    tampered["decoding_settings"]["initial_prompt"] = "changed"

    assert (
        artifact_hash_from_uri(tampered["artifact_uri"]) == create_fake_artifact("healthy").sha256
    )
    with pytest.raises(ValidationError, match="decoding settings hash"):
        run_exact_artifact_gate(
            create_fake_artifact("healthy"), tampered, evaluation.champion_report
        )


def test_injected_empty_output_is_caught_end_to_end(settings: ArtifactSettings) -> None:
    """A fixture model that emits no text is caught with a failing report decision."""
    evaluation = _evaluate(settings, "empty")
    artifact = create_fake_artifact("empty")

    assert evaluation.probe_suite.status == "fail"
    codes = evaluation.probe_suite.summary()["errors_by_code"]
    assert codes.get("empty_output", 0) > 0
    assert evaluation.report["gate_decision"]["outcome"] == "fail"
    assert "empty_output" in evaluation.report["gate_decision"]["reason"]
    assert evaluation.gate_summary["outcome"] == "fail"
    assert evaluation.champion_report["artifact_uri"] != artifact.uri


def test_exact_artifact_uri_is_enforced(settings: ArtifactSettings) -> None:
    """The converter substitutes the artifact; the evaluator rejects any other URI."""
    artifact = create_fake_artifact("healthy")
    evaluation = _evaluate(settings, "healthy")

    converted = ExactArtifactConverter(artifact).convert(settings.checkpoint_uri)
    assert converted.artifact_uri == artifact.uri
    assert converted.source_uri == settings.checkpoint_uri
    assert evaluate_checkpoint_uris_do_not_change_result(
        settings, artifact, "fake://other/checkpoint"
    )

    evaluator = HashedReportEvaluator(artifact, evaluation.report)
    with pytest.raises(GateError, match="non-deployable URI"):
        evaluator.evaluate(f"artifact+sha256:{'0' * 64}://fake/empty", ("general_capability",))
    assert evaluator.evaluate(artifact.uri, ("general_capability",)) == evaluation.report


def test_report_validation_rejects_tampering(settings: ArtifactSettings) -> None:
    """Changed decoding settings, artifact hashes or transcripts fail validation."""
    report = _evaluate(settings, "healthy").report

    changed_settings: dict[str, Any] = deepcopy(report)
    changed_settings["decoding_settings"]["initial_prompt"] = "changed"
    with pytest.raises(ValidationError, match="decoding settings hash"):
        validate_stt_report(changed_settings)

    changed_artifact: dict[str, Any] = deepcopy(report)
    changed_artifact["decoding_settings"][EXTENSION_KEY]["artifact"]["sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="reported artifact hash"):
        validate_stt_report(changed_artifact)

    leaked: dict[str, Any] = deepcopy(report)
    leaked["decoding_settings"][EXTENSION_KEY]["expected_transcript"] = "synthetic text"
    with pytest.raises(ValidationError, match="must not carry transcripts"):
        validate_stt_report(leaked)


@pytest.mark.parametrize("tamper_side", ("challenger", "champion"))
def test_nested_decoding_hash_tamper_fails_closed_on_every_public_seam(
    settings: ArtifactSettings, tmp_path: Path, tamper_side: str
) -> None:
    """Only the nested hash changes; every public report seam refuses the report."""
    evaluation = _evaluate(settings, "healthy")
    clean = evaluation.report if tamper_side == "challenger" else evaluation.champion_report
    other = evaluation.champion_report if tamper_side == "challenger" else evaluation.report
    original_hash = clean["decoding_settings"][EXTENSION_KEY]["decoding"]["sha256"]

    tampered: dict[str, Any] = deepcopy(clean)
    tampered["decoding_settings"][EXTENSION_KEY]["decoding"]["sha256"] = "0" * 64

    # Prove the regression fixture changes only the nested hash: the top-level
    # hash still recomputes from the decoding fields, and restoring that one
    # value reproduces the original report byte-for-byte.
    assert original_hash == decoding_hash_from_fields(tampered["decoding_settings"])
    assert tampered["decoding_settings"]["decoding_settings_sha256"] == original_hash
    restored: dict[str, Any] = deepcopy(tampered)
    restored["decoding_settings"][EXTENSION_KEY]["decoding"]["sha256"] = original_hash
    assert restored == clean

    with pytest.raises(ValidationError, match="nested decoding hash"):
        validate_stt_report(tampered)

    output = tmp_path / "tampered-stt-report.json"
    with pytest.raises(ValidationError, match="nested decoding hash"):
        write_stt_report(tampered, output)
    assert not output.exists()

    artifact = create_fake_artifact("healthy")
    with pytest.raises(ValidationError, match="nested decoding hash"):
        HashedReportEvaluator(artifact, tampered).evaluate(
            tampered["artifact_uri"], ("general_capability",)
        )

    with pytest.raises(ValidationError, match="nested decoding hash"):
        run_exact_artifact_gate(
            artifact,
            tampered if tamper_side == "challenger" else clean,
            other if tamper_side == "challenger" else tampered,
        )


def test_report_has_no_audio_bytes_or_pii_shaped_text(settings: ArtifactSettings) -> None:
    """Report JSON contains counts and hashes only, not fixture text or audio."""
    report = _evaluate(settings, "hallucinating").report
    serialised = json.dumps(report, sort_keys=True)

    assert "sam sample" not in serialised.casefold()
    assert "@example.com" not in serialised.casefold()
    assert "+1 555 010" not in serialised.casefold()
    assert settings.hallucination_text.casefold() not in serialised.casefold()
    assert "RIFF" not in serialised
