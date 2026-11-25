"""P2-09.3/P2-09.4: converted-artifact hook, promotion and rollback orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from services.evalgate.gate import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    Gate,
    GateError,
    GateSettings,
    InMemoryAliasStore,
    PrefixArtifactConverter,
    PromotionAuditLog,
    StaticReportEvaluator,
)
from tests.unit.evalgate_gate.conftest import make_report
from voiceloop_common import ValidationError

SOURCE_URI = "checkpoint://parser-v4"
CONVERTED_URI = "converted://parser-v4"


def _gate(
    tmp_path: Path,
    reports: dict[str, dict[str, Any]],
    *,
    settings: GateSettings | None = None,
) -> tuple[Gate, InMemoryAliasStore, PromotionAuditLog]:
    """Assemble a gate with fakes and a champion v3 / challenger v4 alias pair."""
    store = InMemoryAliasStore({CHAMPION_ALIAS: "v3", CHALLENGER_ALIAS: "v4"})
    log = PromotionAuditLog(tmp_path / "audit.jsonl")
    gate = Gate(
        settings=settings or GateSettings(min_e2e_margin=0.01),
        converter=PrefixArtifactConverter(),
        evaluator=StaticReportEvaluator(reports),
        alias_store=store,
        audit_log=log,
    )
    return gate, store, log


def _champion() -> dict[str, Any]:
    """Return a champion report with full slice coverage."""
    return make_report(artifact_uri="converted://parser-v3", e2e_accuracy=0.70)


def _better() -> dict[str, Any]:
    """Return a challenger report that clears the gate."""
    return make_report(artifact_uri=CONVERTED_URI, e2e_accuracy=0.80, e2e_delta_lower=0.05)


def test_evaluate_uses_converted_artifact(tmp_path: Path, champion_report: dict[str, Any]) -> None:
    gate, store, _ = _gate(tmp_path, {CONVERTED_URI: _better()})

    decision, report = gate.evaluate(SOURCE_URI, champion_report)

    assert decision.passed is True
    assert report["artifact_uri"] == CONVERTED_URI
    assert store.snapshot() == {CHAMPION_ALIAS: "v3", CHALLENGER_ALIAS: "v4"}


def test_report_for_another_artifact_is_refused(
    tmp_path: Path, champion_report: dict[str, Any]
) -> None:
    mismatched = make_report(
        artifact_uri="converted://another-model", e2e_accuracy=0.80, e2e_delta_lower=0.05
    )
    gate, _, _ = _gate(tmp_path, {CONVERTED_URI: mismatched})

    with pytest.raises(GateError, match="is not the converted artifact"):
        gate.evaluate(SOURCE_URI, champion_report)


def test_gate_promotes_a_better_model(tmp_path: Path, champion_report: dict[str, Any]) -> None:
    gate, store, log = _gate(tmp_path, {CONVERTED_URI: _better()})

    result = gate.run(
        SOURCE_URI,
        champion_report,
        challenger_version="v4",
        actor_id="ml-engineer",
        reason="fixture gate pass",
    )

    assert result.decision.passed is True
    assert result.converted_artifact_uri == CONVERTED_URI
    assert result.promotion is not None
    assert result.promotion.before_version == "v3"
    assert result.promotion.after_version == "v4"
    assert store.get(CHAMPION_ALIAS) == "v4"
    assert [entry.action for entry in log.entries()] == ["promote"]
    assert log.verify_chain() == ()


def test_gate_refuses_a_worse_model_without_side_effects(
    tmp_path: Path, champion_report: dict[str, Any]
) -> None:
    worse = make_report(artifact_uri=CONVERTED_URI, e2e_accuracy=0.60, e2e_delta_lower=-0.05)
    gate, store, log = _gate(tmp_path, {CONVERTED_URI: worse})

    result = gate.run(
        SOURCE_URI,
        champion_report,
        challenger_version="v4",
        actor_id="ml-engineer",
        reason="worse model",
    )

    assert result.decision.passed is False
    assert result.promotion is None
    assert store.get(CHAMPION_ALIAS) == "v3"
    assert log.entries() == ()


def test_gate_rollback_restores_previous_champion(
    tmp_path: Path, champion_report: dict[str, Any]
) -> None:
    gate, store, log = _gate(tmp_path, {CONVERTED_URI: _better()})
    result = gate.run(
        SOURCE_URI,
        champion_report,
        challenger_version="v4",
        actor_id="ml-engineer",
        reason="promote",
    )
    assert result.promotion is not None

    rollback_record = gate.rollback(
        result.promotion, actor_id="oncall", reason="post-promotion regression"
    )

    assert store.get(CHAMPION_ALIAS) == "v3"
    assert rollback_record.before_version == "v4"
    assert rollback_record.after_version == "v3"
    assert [entry.action for entry in log.entries()] == ["promote", "rollback"]
    assert log.verify_chain() == ()


def test_no_report_for_converted_artifact_is_an_error(
    tmp_path: Path, champion_report: dict[str, Any]
) -> None:
    gate, _, _ = _gate(tmp_path, {})

    with pytest.raises(GateError, match="no evaluation report"):
        gate.evaluate(SOURCE_URI, champion_report)


def test_static_evaluator_validates_reports(tmp_path: Path) -> None:
    invalid = make_report(artifact_uri=CONVERTED_URI, e2e_accuracy=0.8)
    del invalid["metrics"]["e2e_booking_accuracy"]
    evaluator = StaticReportEvaluator({CONVERTED_URI: invalid})

    with pytest.raises(ValidationError):
        evaluator.evaluate(CONVERTED_URI, ["general_capability"])


def test_gate_fails_closed_on_zero_turn_language_slice(
    tmp_path: Path, champion_report: dict[str, Any]
) -> None:
    """Real fixture coverage must exist before a slice can count as safe."""
    challenger = make_report(
        artifact_uri=CONVERTED_URI,
        e2e_accuracy=0.95,
        e2e_delta_lower=0.20,
        slice_values={"other_languages": (0.0, 1.0)},
    )
    gate, store, log = _gate(tmp_path, {CONVERTED_URI: challenger})

    result = gate.run(
        SOURCE_URI,
        champion_report,
        challenger_version="v4",
        actor_id="ml-engineer",
        reason="multilingual coverage missing",
    )

    assert result.decision.passed is False
    assert "other_languages: zero turns" in " ".join(result.decision.reasons)
    assert result.promotion is None
    assert store.get(CHAMPION_ALIAS) == "v3"
    assert log.entries() == ()
