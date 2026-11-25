"""Typed settings and provenance anchor tests for P3-03."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from services.datasets.augment.settings import (
    EXPECTED_FIXTURE_SHA256,
    EXPECTED_GOLDEN_CONTENT_HASH,
    STAND_IN_LABEL,
    AugmentSettings,
    parse_csv,
)
from voiceloop_common import ConfigError


def test_defaults_use_typed_prefix_and_committed_anchors() -> None:
    settings = AugmentSettings()
    assert settings.model_config.get("env_prefix") == "VOICELOOP_AUGMENT_"
    assert EXPECTED_GOLDEN_CONTENT_HASH == (
        "662cdaf43ee2d5fde03faf5e69fde196114d4f312406d94c00bf53714d03da8e"
    )
    assert EXPECTED_FIXTURE_SHA256 == (
        "aca23a303954d035245a97f78319df3694d63611b717a2ff2131bc2e0a4453fd"
    )
    assert STAND_IN_LABEL == "SYNTHETIC/FIXTURE STAND-IN - NOT REAL AUDIO"
    assert settings.expected_golden_content_hash == EXPECTED_GOLDEN_CONTENT_HASH
    assert settings.expected_fixture_sha256 == EXPECTED_FIXTURE_SHA256
    assert "misheard_surname" in settings.failure_classes()
    assert all(token.startswith("surname-token-") for token in settings.name_tokens())


def test_environment_overrides_are_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICELOOP_AUGMENT_SEED", "11")
    monkeypatch.setenv("VOICELOOP_AUGMENT_DURATION_MS", "120")
    monkeypatch.setenv("VOICELOOP_DATASETS_SPLIT_SEED", "99")
    settings = AugmentSettings()
    assert settings.seed == 11
    assert settings.duration_ms == 120
    assert settings.failure_classes_csv == ",".join(
        ("misheard_surname", "new_surname_tokens", "misheard_guests", "line_noise")
    )


def test_csv_parser_rejects_empty_and_duplicates() -> None:
    assert parse_csv(" a, b ,,c ", field="classes") == ("a", "b", "c")
    with pytest.raises(ConfigError):
        parse_csv(" , ", field="classes")
    with pytest.raises(ConfigError):
        parse_csv("a,a", field="classes")


def test_invalid_bands_and_rates_fail_closed() -> None:
    with pytest.raises(PydanticValidationError):
        AugmentSettings(telephony_low_hz=4000, telephony_high_hz=300)
    with pytest.raises(PydanticValidationError):
        AugmentSettings(duration_ms=0)
    with pytest.raises(PydanticValidationError):
        AugmentSettings(telephony_sample_rate_hz=8000, telephony_high_hz=5000)
