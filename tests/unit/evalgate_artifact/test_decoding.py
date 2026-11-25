"""Deterministic decoding combinations and their exact-settings hashes."""

from __future__ import annotations

import pytest

from services.evalgate.artifact.decoding import (
    DecodingSettings,
    decode_combinations,
    decoding_hash_from_fields,
    language_combinations,
    production_combination,
)
from services.evalgate.artifact.settings import ArtifactSettings
from voiceloop_common import ConfigError


def test_base_combinations_cover_all_three_switches(settings: ArtifactSettings) -> None:
    """The eight hotword/prompt/context combinations are ordered and unique."""
    combinations = decode_combinations(settings)

    assert [combination.name for combination in combinations] == [
        "hotwords_off_prompt_off_context_off",
        "hotwords_off_prompt_off_context_on",
        "hotwords_off_prompt_on_context_off",
        "hotwords_off_prompt_on_context_on",
        "hotwords_on_prompt_off_context_off",
        "hotwords_on_prompt_off_context_on",
        "hotwords_on_prompt_on_context_off",
        "hotwords_on_prompt_on_context_on",
    ]
    hashes = [combination.settings.sha256 for combination in combinations]
    assert len(set(hashes)) == 8
    for combination in combinations:
        fields = combination.settings.to_report_fields()
        assert decoding_hash_from_fields(fields) == combination.settings.sha256


def test_production_and_language_combinations(settings: ArtifactSettings) -> None:
    """Production uses the configured values; extra languages clone the same matrix."""
    production = production_combination(settings)
    fields = production.settings.to_report_fields()

    assert fields["hotwords"] == list(settings.hotwords)
    assert fields["initial_prompt"] == settings.initial_prompt
    assert fields["condition_on_previous_text"] == settings.condition_on_previous_text
    assert fields["language"] == settings.language

    base = decode_combinations(settings)
    extra = language_combinations(base, ("es", "fr-ca"))
    assert len(extra) == len(base) * 2
    assert extra[0].name.endswith("_lang_es")
    assert extra[0].settings.sha256 != base[0].settings.sha256
    assert (
        decoding_hash_from_fields(extra[-1].settings.to_report_fields())
        == extra[-1].settings.sha256
    )


def test_blank_settings_are_rejected() -> None:
    """Blank hotwords or language never receive a hash."""
    with pytest.raises(ConfigError, match="hotwords"):
        DecodingSettings(("",), "prompt", False, "en")
    with pytest.raises(ConfigError, match="language"):
        DecodingSettings((), "prompt", False, "  ")
