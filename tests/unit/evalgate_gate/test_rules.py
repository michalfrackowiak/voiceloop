"""P2-09.1/P2-09.2: pass rule, CI deltas, slices and evidence sets."""

from __future__ import annotations

from typing import Any

import pytest

from services.evalgate.gate import GateSettings, decide_gate
from tests.unit.evalgate_gate.conftest import make_report
from voiceloop_common import ValidationError


def _joined(decision: Any) -> str:
    """Join decision reasons for substring assertions."""
    return " ".join(decision.reasons)


def test_better_challenger_passes(
    gate_settings: GateSettings, champion_report: dict[str, Any], better_report: dict[str, Any]
) -> None:
    decision = decide_gate(champion_report, better_report, gate_settings)

    assert decision.passed is True
    assert decision.e2e_delta_lower == pytest.approx(0.05)
    assert decision.required_margin == pytest.approx(0.01)
    assert "exceeds required margin" in _joined(decision)
    assert len(decision.slice_deltas) == 5


def test_worse_challenger_fails(
    gate_settings: GateSettings, champion_report: dict[str, Any], worse_report: dict[str, Any]
) -> None:
    decision = decide_gate(champion_report, worse_report, gate_settings)

    assert decision.passed is False
    assert "does not exceed required margin" in _joined(decision)


def test_better_wer_with_worse_e2e_is_refused(gate_settings: GateSettings) -> None:
    """WER improvements must never promote a model (context.md §2 invariant 4)."""
    champion = make_report(artifact_uri="converted://champion", e2e_accuracy=0.70, wer=0.50)
    challenger = make_report(
        artifact_uri="converted://challenger",
        e2e_accuracy=0.60,
        e2e_delta_lower=-0.06,
        wer=0.10,
        wer_delta_lower=-0.40,
        slice_values={name: (10.0, 0.60) for name in ("names", "dates", "numbers", "silence")}
        | {"other_languages": (10.0, 0.60)},
    )

    decision = decide_gate(champion, challenger, gate_settings)

    assert decision.passed is False
    assert "e2e delta lower bound" in _joined(decision)
    assert "WER" not in _joined(decision)


def test_absolute_e2e_value_is_not_read_as_delta(
    gate_settings: GateSettings, champion_report: dict[str, Any]
) -> None:
    """A high absolute accuracy cannot compensate for a negative paired delta."""
    challenger = make_report(
        artifact_uri="converted://challenger",
        e2e_accuracy=0.95,
        e2e_delta_lower=-0.02,
        slice_values={
            name: (10.0, 0.95)
            for name in ("names", "dates", "numbers", "silence", "other_languages")
        },
    )

    decision = decide_gate(champion_report, challenger, gate_settings)

    assert decision.passed is False
    assert decision.e2e_delta_lower == pytest.approx(-0.02)


def test_margin_is_strict(gate_settings: GateSettings, champion_report: dict[str, Any]) -> None:
    """An e2e lower bound equal to the margin does not pass."""
    challenger = make_report(
        artifact_uri="converted://challenger", e2e_accuracy=0.75, e2e_delta_lower=0.01
    )

    decision = decide_gate(champion_report, challenger, gate_settings)

    assert decision.passed is False


def test_low_confidence_level_fails(
    champion_report: dict[str, Any], better_report: dict[str, Any]
) -> None:
    settings = GateSettings(min_e2e_margin=0.01, min_ci_level=0.9)
    weak = make_report(
        artifact_uri="converted://challenger-v4",
        e2e_accuracy=0.80,
        e2e_delta_lower=0.05,
        ci_level=0.8,
    )

    decision = decide_gate(champion_report, weak, settings)

    assert decision.passed is False
    assert "confidence level" in _joined(decision)


def test_zero_turn_slice_fails_closed(
    gate_settings: GateSettings, champion_report: dict[str, Any]
) -> None:
    """Zero-turn critical slices must not silently count as no regression."""
    challenger = make_report(
        artifact_uri="converted://challenger-v4",
        e2e_accuracy=0.95,
        e2e_delta_lower=0.20,
        slice_values={"other_languages": (0.0, 1.0)},
    )

    decision = decide_gate(champion_report, challenger, gate_settings)

    assert decision.passed is False
    assert "other_languages: zero turns" in _joined(decision)
    assert "other_languages" not in decision.slice_deltas


def test_zero_turn_champion_slice_fails_closed(
    gate_settings: GateSettings, better_report: dict[str, Any]
) -> None:
    champion = make_report(
        artifact_uri="converted://champion-v3",
        e2e_accuracy=0.70,
        slice_values={"silence": (0.0, 1.0)},
    )

    decision = decide_gate(champion, better_report, gate_settings)

    assert decision.passed is False
    assert "silence: zero turns" in _joined(decision)


def test_missing_slice_fails_closed(
    gate_settings: GateSettings, champion_report: dict[str, Any], better_report: dict[str, Any]
) -> None:
    del better_report["slice_metrics"]["names"]

    decision = decide_gate(champion_report, better_report, gate_settings)

    assert decision.passed is False
    assert "names: missing slice metrics" in _joined(decision)


def test_critical_slice_regression_fails(
    gate_settings: GateSettings, champion_report: dict[str, Any]
) -> None:
    challenger = make_report(
        artifact_uri="converted://challenger-v4",
        e2e_accuracy=0.90,
        e2e_delta_lower=0.15,
        slice_values={"names": (12.0, 0.55)},
    )

    decision = decide_gate(champion_report, challenger, gate_settings)

    assert decision.passed is False
    assert "names: e2e regression" in _joined(decision)
    assert decision.slice_deltas["names"] == pytest.approx(-0.15)


def test_slice_tolerance_allows_configured_small_drop(
    champion_report: dict[str, Any], better_report: dict[str, Any]
) -> None:
    settings = GateSettings(min_e2e_margin=0.01, slice_regression_tolerance=0.2)
    better_report["slice_metrics"]["dates"]["e2e_booking_accuracy"] = 0.65

    decision = decide_gate(champion_report, better_report, settings)

    assert decision.passed is True
    assert decision.slice_deltas["dates"] == pytest.approx(-0.05)


def test_golden_label_alone_is_not_evidence(gate_settings: GateSettings) -> None:
    champion = make_report(
        artifact_uri="converted://champion", e2e_accuracy=0.7, eval_sets=("golden",)
    )
    challenger = make_report(
        artifact_uri="converted://challenger",
        e2e_accuracy=0.8,
        e2e_delta_lower=0.05,
        eval_sets=("golden",),
    )

    decision = decide_gate(champion, challenger, gate_settings)

    assert decision.passed is False
    assert "golden" in _joined(decision)
    assert "no accepted eval set" in _joined(decision)


def test_golden_label_mixed_with_regression_is_refused(
    gate_settings: GateSettings, champion_report: dict[str, Any]
) -> None:
    challenger = make_report(
        artifact_uri="converted://challenger",
        e2e_accuracy=0.8,
        e2e_delta_lower=0.05,
        eval_sets=("golden", "regression"),
    )

    decision = decide_gate(champion_report, challenger, gate_settings)

    assert decision.passed is False
    assert "not sealed-set evidence" in _joined(decision)


def test_contract_violation_raises(
    gate_settings: GateSettings, champion_report: dict[str, Any], better_report: dict[str, Any]
) -> None:
    del better_report["confidence_intervals"]["wer"]["upper"]

    with pytest.raises(ValidationError, match="confidence_intervals"):
        decide_gate(champion_report, better_report, gate_settings)


def test_invalid_champion_report_raises(
    gate_settings: GateSettings, better_report: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        decide_gate({"schema_version": "1.0.0"}, better_report, gate_settings)


def test_report_field_is_schema_shaped(
    gate_settings: GateSettings, champion_report: dict[str, Any], better_report: dict[str, Any]
) -> None:
    field = decide_gate(champion_report, better_report, gate_settings).to_report_field()

    assert set(field) == {"outcome", "reason", "evaluated_at"}
    assert field["outcome"] == "pass"
    assert field["evaluated_at"].endswith("Z")
