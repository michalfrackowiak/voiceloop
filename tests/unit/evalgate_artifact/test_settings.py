"""Typed settings and committed hash pins for P3-04."""

from __future__ import annotations

import pytest

from services.evalgate.artifact.settings import (
    EXPECTED_FIXTURE_SHA256,
    EXPECTED_GOLDEN_CONTENT_HASH,
    ArtifactSettings,
)
from voiceloop_common import ConfigError


def test_default_expected_hashes_are_the_committed_pins() -> None:
    """The package refuses any fixture/golden hash other than the committed ones."""
    settings = ArtifactSettings()

    assert settings.expected_fixture_sha256 == EXPECTED_FIXTURE_SHA256
    assert settings.expected_golden_content_hash == EXPECTED_GOLDEN_CONTENT_HASH
    assert settings.language == "en"
    assert settings.served_languages() == ("en",)
    assert settings.non_english_languages() == ()


def test_served_language_codes_are_validated() -> None:
    """Language configuration is lowercased and cannot contain blank/invalid tags."""
    settings = ArtifactSettings(served_languages_csv="EN, es ,fr-CA")

    assert settings.served_languages() == ("en", "es", "fr-ca")
    assert settings.non_english_languages() == ("es", "fr-ca")

    with pytest.raises(ConfigError, match="invalid served language codes"):
        ArtifactSettings(served_languages_csv="en,english").served_languages()
    with pytest.raises(ConfigError, match="at least one served language"):
        ArtifactSettings(served_languages_csv=" , ").served_languages()
