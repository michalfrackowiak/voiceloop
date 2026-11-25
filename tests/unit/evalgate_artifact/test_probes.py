"""Read-only fixture probes and deterministic failure injection."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from services.evalgate.artifact.artifact import create_fake_artifact
from services.evalgate.artifact.decoding import decode_combinations, production_combination
from services.evalgate.artifact.probes import (
    SILENCE_KINDS,
    ProbeInputs,
    fake_transcribe,
    hallucination_summary,
    load_probe_inputs,
    multilingual_summary,
    run_probe_suite,
    verify_inputs,
)
from services.evalgate.artifact.settings import (
    EXPECTED_FIXTURE_SHA256,
    EXPECTED_GOLDEN_CONTENT_HASH,
    ArtifactSettings,
)
from services.evalgate.metrics import tokenize


def _index_where(inputs: ProbeInputs, predicate: Callable[[Mapping[str, Any]], bool]) -> int:
    """Return the first matching fixture row index or fail the test clearly."""
    for index, row in enumerate(inputs.rows):
        if predicate(row):
            return index
    raise AssertionError("committed fixtures do not contain the required probe case")


def _event_text(inputs: ProbeInputs, index: int) -> str:
    """Return the baseline STT text recorded in a fixture event."""
    stt = inputs.events[index].get("stt")
    return str(stt.get("text", "")) if isinstance(stt, Mapping) else ""


def test_verify_inputs_checks_committed_hashes_without_writes(
    settings: ArtifactSettings,
) -> None:
    """Fixture and golden hashes are pinned and provenance forbids training use."""
    provenance = verify_inputs(settings)

    assert provenance["fixture_sha256"] == EXPECTED_FIXTURE_SHA256
    assert provenance["golden_content_hash"] == EXPECTED_GOLDEN_CONTENT_HASH
    assert provenance["fixture_hash_verified"] is True
    assert provenance["golden_seal_ok"] is True
    assert provenance["golden_synthetic"] is True
    assert provenance["golden_workflow_stand_in"] is True
    assert provenance["golden_used_for_training_or_tuning"] is False
    assert provenance["real_audio_or_human_labels_used"] is False
    assert "SYNTHETIC/FIXTURE STAND-IN" in str(provenance["limitation"])


def test_fake_transcribe_injects_each_failure(settings: ArtifactSettings) -> None:
    """The fake seams produce empty, truncated, hallucinated and forgotten text."""
    inputs = load_probe_inputs(settings)
    primary = production_combination(settings)
    speaking = _index_where(
        inputs, lambda row: len(tokenize(str(row.get("expected_transcript", "")))) >= 4
    )
    silent = _index_where(
        inputs,
        lambda row: (
            str(row.get("audio_kind", "")) in SILENCE_KINDS
            and not tokenize(str(row.get("expected_transcript", "")))
        ),
    )

    empty_text, empty_slots = fake_transcribe(
        create_fake_artifact("empty"),
        inputs.rows[speaking],
        inputs.events[speaking],
        primary.settings,
        settings,
        inputs.prior_turn_counts[speaking],
    )
    assert empty_text == ""
    assert empty_slots == {}

    truncated_text, _ = fake_transcribe(
        create_fake_artifact("truncated"),
        inputs.rows[speaking],
        inputs.events[speaking],
        primary.settings,
        settings,
        inputs.prior_turn_counts[speaking],
    )
    baseline_tokens = tokenize(_event_text(inputs, speaking))
    truncated_tokens = tokenize(truncated_text)
    assert 0 < len(truncated_tokens) < len(baseline_tokens)
    assert truncated_tokens == baseline_tokens[: len(truncated_tokens)]

    hallucinated_text, _ = fake_transcribe(
        create_fake_artifact("hallucinating"),
        inputs.rows[silent],
        inputs.events[silent],
        primary.settings,
        settings,
        inputs.prior_turn_counts[silent],
    )
    assert hallucinated_text == settings.hallucination_text
    assert tokenize(hallucinated_text)

    spanish_settings = production_combination(settings.model_copy(update={"language": "es"}))
    forgotten_text, _ = fake_transcribe(
        create_fake_artifact("forgetting"),
        inputs.rows[speaking],
        inputs.events[speaking],
        spanish_settings.settings,
        settings,
        inputs.prior_turn_counts[speaking],
    )
    assert forgotten_text == ""


def test_probe_suite_catches_empty_and_truncated_outputs(
    settings: ArtifactSettings,
) -> None:
    """Injected empty/prefix-truncated output fails the suite with structured codes."""
    inputs = load_probe_inputs(settings)
    combinations = decode_combinations(settings)[:1]

    healthy = run_probe_suite(inputs, create_fake_artifact("healthy"), combinations, settings)
    assert healthy.status == "pass"
    assert healthy.summary()["error_count"] == 0

    empty = run_probe_suite(inputs, create_fake_artifact("empty"), combinations, settings)
    assert empty.status == "fail"
    assert empty.summary()["errors_by_code"]["empty_output"] > 0

    truncated = run_probe_suite(inputs, create_fake_artifact("truncated"), combinations, settings)
    assert truncated.status == "fail"
    assert truncated.summary()["errors_by_code"]["truncated_output"] > 0


def test_hallucination_and_multilingual_summaries(
    settings: ArtifactSettings,
) -> None:
    """Silence hallucination measures firing cases; multilingual tags stay diagnostic."""
    inputs = load_probe_inputs(settings)
    primary = production_combination(settings)

    healthy_summary = hallucination_summary(
        inputs, create_fake_artifact("healthy"), primary, settings
    )
    assert healthy_summary["turns"] > 0
    assert healthy_summary["hallucinated_turns"] == 0
    assert healthy_summary["rate"] == 0.0
    assert healthy_summary["promotion_metric"] is False

    hallucinating_summary = hallucination_summary(
        inputs, create_fake_artifact("hallucinating"), primary, settings
    )
    assert hallucinating_summary["hallucinated_turns"] == hallucinating_summary["turns"]
    assert hallucinating_summary["rate"] == 1.0
    assert hallucinating_summary["firing_cases"]

    default_multilingual = multilingual_summary(
        inputs, create_fake_artifact("forgetting"), primary, settings
    )
    assert default_multilingual["turns"] == 0
    assert default_multilingual["gate_evidence"] is False
    assert default_multilingual["status"].startswith("not_evaluated")

    multilingual_settings = settings.model_copy(update={"served_languages_csv": "en,es"})
    multilingual_combination = production_combination(multilingual_settings)
    tagged = multilingual_summary(
        inputs,
        create_fake_artifact("forgetting"),
        multilingual_combination,
        multilingual_settings,
    )
    assert tagged["turns"] > 0
    assert tagged["forgetting_rate"] > 0.0
    assert tagged["surrogate_only"] is True
    assert tagged["human_labels"] is False
    assert tagged["gate_evidence"] is False
    assert tagged["status"] == "diagnostic_only_synthetic_language_tag"

    # A non-English language in the base combination of the suite is classified
    # as forgetting rather than an ordinary empty output.
    tagged_combination = replace(primary.settings, language="es")
    assert tagged_combination.sha256 != primary.settings.sha256
