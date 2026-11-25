"""Typed settings with the VOICELOOP_REVIEW_ prefix."""

from __future__ import annotations

import pytest

from services.review_adapter import ReviewAdapterSettings


def test_env_prefix_is_scoped() -> None:
    """Settings never read another component's variables."""
    assert ReviewAdapterSettings.model_config["env_prefix"] == "VOICELOOP_REVIEW_"


def test_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tool, schema path and reviewer ID are all environment-backed."""
    monkeypatch.setenv("VOICELOOP_REVIEW_TOOL", "fake_alternate")
    monkeypatch.setenv("VOICELOOP_REVIEW_BOOKING_SCHEMA_PATH", "tests/custom.json")
    monkeypatch.setenv("VOICELOOP_REVIEW_DEFAULT_REVIEWER_ID", "reviewer-77")
    settings = ReviewAdapterSettings()
    assert settings.tool == "fake_alternate"
    assert settings.booking_schema_path == "tests/custom.json"
    assert settings.default_reviewer_id == "reviewer-77"


def test_defaults_are_fake_and_offline() -> None:
    """Defaults select a fake tool and the bundled booking schema."""
    settings = ReviewAdapterSettings()
    assert settings.tool == "fake_in_memory"
    assert settings.booking_schema_path is None
    assert settings.default_reviewer_id == "fixture-reviewer"
