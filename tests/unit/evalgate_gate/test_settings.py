"""P2-09: typed settings read only the VOICELOOP_GATE_ prefix."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from services.evalgate.gate import GateSettings


def test_defaults() -> None:
    settings = GateSettings()

    assert settings.min_e2e_margin == pytest.approx(0.01)
    assert settings.min_ci_level == pytest.approx(0.95)
    assert settings.slice_regression_tolerance == pytest.approx(0.0)
    assert settings.eval_sets == ["general_capability", "regression"]
    assert settings.audit_log_path == "reports/evalgate/promotion_audit.jsonl"


def test_env_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICELOOP_GATE_MIN_E2E_MARGIN", "0.04")
    monkeypatch.setenv("VOICELOOP_GATE_MIN_CI_LEVEL", "0.9")
    monkeypatch.setenv("VOICELOOP_GATE_SLICE_REGRESSION_TOLERANCE", "0.02")
    monkeypatch.setenv("VOICELOOP_GATE_EVAL_SETS", '["regression"]')
    monkeypatch.setenv("VOICELOOP_GATE_AUDIT_LOG_PATH", "tmp/audit.jsonl")

    settings = GateSettings()

    assert settings.min_e2e_margin == pytest.approx(0.04)
    assert settings.min_ci_level == pytest.approx(0.9)
    assert settings.slice_regression_tolerance == pytest.approx(0.02)
    assert settings.eval_sets == ["regression"]
    assert settings.audit_log_path == "tmp/audit.jsonl"


def test_unrelated_prefix_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICELOOP_EVALGATE_MIN_E2E_MARGIN", "0.5")

    assert GateSettings().min_e2e_margin == pytest.approx(0.01)


def test_invalid_ranges_are_rejected() -> None:
    with pytest.raises(ValidationError):
        GateSettings(min_e2e_margin=-0.1)
    with pytest.raises(ValidationError):
        GateSettings(min_ci_level=1.0)
    with pytest.raises(ValidationError):
        GateSettings(slice_regression_tolerance=-0.1)
