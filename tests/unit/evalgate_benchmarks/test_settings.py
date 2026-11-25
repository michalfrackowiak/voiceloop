"""Typed settings tests for the benchmark harness."""

from __future__ import annotations

import pytest

from services.evalgate.benchmarks.settings import (
    EXPECTED_GOLDEN_CONTENT_HASH,
    BenchmarkSettings,
)
from voiceloop_common import ConfigError


def test_env_prefix_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings read VOICELOOP_BENCH_ variables and parse candidate order."""
    monkeypatch.setenv("VOICELOOP_BENCH_BOOTSTRAP_SEED", "99")
    monkeypatch.setenv("VOICELOOP_BENCH_CANDIDATES_CSV", "degraded, identity")
    settings = BenchmarkSettings()
    assert settings.bootstrap_seed == 99
    assert settings.candidate_names() == ("degraded", "identity")
    assert settings.expected_golden_content_hash == EXPECTED_GOLDEN_CONTENT_HASH
    assert settings.decoding_settings()["condition_on_previous_text"] is False


def test_candidate_names_reject_duplicates() -> None:
    """Duplicate candidate names are rejected before any scoring."""
    with pytest.raises(ConfigError, match="unique"):
        BenchmarkSettings(candidates_csv="identity,identity").candidate_names()
    with pytest.raises(ConfigError, match="at least one"):
        BenchmarkSettings(candidates_csv=" , ").candidate_names()


def test_invalid_bootstrap_level_rejected() -> None:
    """A confidence level of 1.0 is outside the contract's valid range."""
    with pytest.raises(ValueError):
        BenchmarkSettings(bootstrap_level=1.0)
