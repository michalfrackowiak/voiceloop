"""Hand-computed slot F1, schema-valid rate and E2E accuracy cases."""

from __future__ import annotations

from typing import Any

import pytest

from services.evalgate.metrics import (
    EvaluationError,
    Prediction,
    aggregate_metrics,
    aggregate_slot_f1,
    e2e_booking_accuracy,
    e2e_field_accuracies,
    e2e_turn_result,
    evaluate_turn,
    is_schema_valid,
    schema_valid_rate,
    slot_f1,
)


def valid_slots(**overrides: Any) -> dict[str, Any]:
    """Return a complete, schema-valid booking slot object."""
    slots: dict[str, Any] = {
        "check_in": "2026-12-12",
        "check_out": "2026-12-14",
        "guests": 2,
        "room_type": "standard",
        "guest_name": "Ada Fixture",
        "contact": {"email": "ada.fixture@example.com", "phone": "+1 555 010 0001"},
    }
    slots.update(overrides)
    return slots


def test_slot_f1_hand_computed_exact_and_partial() -> None:
    """Exact match is F1 1.0; one wrong field gives 5/6 on all three."""
    gold = valid_slots()
    exact = slot_f1(gold, gold)
    assert (exact.tp, exact.fp, exact.fn) == (6, 0, 0)
    assert exact.f1 == 1.0

    predicted = valid_slots(room_type="suite")
    partial = slot_f1(gold, predicted)
    assert (partial.tp, partial.fp, partial.fn) == (5, 1, 1)
    assert partial.precision == pytest.approx(5 / 6)
    assert partial.recall == pytest.approx(5 / 6)
    assert partial.f1 == pytest.approx(5 / 6)


def test_slot_f1_empty_and_missing_fields() -> None:
    """Two empty slot dicts score 1.0; a dropped field scores 0.0."""
    assert slot_f1({}, {}).f1 == 1.0
    assert slot_f1({"guests": 2}, {}).f1 == 0.0
    assert aggregate_slot_f1(tp=9, fp=1, fn=0) == pytest.approx(2 * 0.9 / 1.9)


def test_schema_valid_rate_hand_computed() -> None:
    """Valid and invalid objects are counted, with a no-slot exemption."""
    valid = valid_slots()
    invalid_dates = valid_slots(check_out="2026-12-01")
    assert is_schema_valid(valid)
    assert not is_schema_valid(invalid_dates)
    assert not is_schema_valid({})
    assert is_schema_valid({}, no_slot_expected=True)
    assert schema_valid_rate([valid, invalid_dates]) == 0.5
    assert schema_valid_rate([{}, {}], [{}, {}]) == 1.0
    assert schema_valid_rate([]) == 0.0
    with pytest.raises(EvaluationError, match="must align"):
        schema_valid_rate([valid], [{}, {}])


def test_e2e_turn_rule_and_field_breakdown() -> None:
    """A turn is correct only when every slot field matches exactly."""
    gold = valid_slots()
    assert e2e_turn_result(gold, gold).turn_correct is True
    assert e2e_turn_result({}, {}).turn_correct is True
    assert e2e_turn_result({}, {"guests": 2}).turn_correct is False

    wrong = e2e_turn_result(gold, valid_slots(guests=3))
    assert wrong.turn_correct is False
    assert wrong.field_accuracy["guests"] is False
    assert wrong.field_accuracy["room_type"] is True
    assert e2e_field_accuracies([wrong])["guests"] == 0.0
    assert e2e_field_accuracies([wrong])["room_type"] == 1.0
    assert e2e_booking_accuracy([e2e_turn_result(gold, gold), wrong]) == 0.5


def test_evaluate_turn_and_aggregate_metrics() -> None:
    """One perfect fixture turn propagates through all five core metrics."""
    gold = valid_slots()
    transcript = "Check in on the twelfth of December for Ada Fixture, two guests."
    turn = evaluate_turn(
        transcript,
        gold,
        Prediction(transcript=transcript, slots=gold),
    )
    assert turn.wer == 0.0
    assert turn.entity_error == 0.0
    assert turn.slot_f1 == 1.0
    assert turn.schema_valid == 1.0
    assert turn.e2e_correct == 1.0

    aggregate = aggregate_metrics([turn])
    assert aggregate["wer"] == 0.0
    assert aggregate["entity_error"] == 0.0
    assert aggregate["slot_f1"] == 1.0
    assert aggregate["schema_valid_rate"] == 1.0
    assert aggregate["e2e_booking_accuracy"] == 1.0
    assert aggregate["e2e_field_guest_name"] == 1.0
    assert isinstance(aggregate["entity_error_names"], float)

    with pytest.raises(EvaluationError, match="zero turns"):
        aggregate_metrics([])
