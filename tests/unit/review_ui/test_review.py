"""Synthetic review item, blind mode, routing and second-opinion tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ui.constants import (
    FIXTURE_ITEM_PATH,
    ROUTING_LABELS,
    SYNTHETIC_LABEL,
    USABILITY_STATUS,
)
from ui.errors import UiValidationError
from ui.form import FormDescriptor
from ui.privacy import assert_no_sensitive_text
from ui.review import (
    REASON_CODES,
    ReviewSubmission,
    SecondOpinionRequest,
    build_review_state,
    evaluate_second_opinion,
    load_review_items,
    record_diff,
)


def test_load_synthetic_items_is_screened_and_pseudonymous() -> None:
    """Committed items are pseudonymous, unique and carry no fixture content."""
    items = load_review_items()

    assert len(items) == 3
    assert len({item.item_id for item in items}) == 3
    assert [item.item_id for item in items] == [
        "item-token-0001",
        "item-token-0002",
        "item-token-0003",
    ]
    assert [item.blind for item in items] == [False, True, False]
    assert_no_sensitive_text([item.to_dict() for item in items])


def test_load_review_items_rejects_pii_in_fixture(tmp_path: Path) -> None:
    """A fixture containing an email fails closed."""
    payload: list[dict[str, Any]] = json.loads(FIXTURE_ITEM_PATH.read_text(encoding="utf-8"))
    payload[0]["guest_name"] = "ada.fixture@example.com"
    path = tmp_path / "synthetic_items.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UiValidationError):
        load_review_items(path)


def test_build_review_state_has_all_ui_contract_sections(
    booking_descriptor: FormDescriptor,
) -> None:
    """Each state exposes the deterministic sections the browser renders."""
    item = load_review_items()[0]
    state = build_review_state(item, booking_descriptor)

    assert state["contract_version"] == "internal-1.0.0"
    assert state["label"] == SYNTHETIC_LABEL
    assert state["usability_status"] == USABILITY_STATUS
    assert set(state["routing"]["labels"]) == set(ROUTING_LABELS)
    assert state["routing"]["selected"] is None
    assert state["second_opinion"]["allowed"] is False
    assert state["player"]["audio_playback"] is False
    assert isinstance(state["form"]["fields"], list)
    assert isinstance(state["record_diff"], list)
    assert_no_sensitive_text(state)


def test_blind_item_withholds_prefill(booking_descriptor: FormDescriptor) -> None:
    """Blind mode suppresses all parser prefill values."""
    blind_item = load_review_items()[1]
    state = build_review_state(blind_item, booking_descriptor)

    assert state["blind_mode"]["active"] is True
    assert state["blind_mode"]["prefill_withheld"] is True
    assert state["item"]["prefill_slots"] == {}
    assert any(issue["code"] == "required" for issue in state["slot_issues"])


def test_record_diff_reports_status_without_values() -> None:
    """The side-panel diff reports only field-level status codes."""
    item = load_review_items()[0]

    rows = record_diff(item.prefill_slots, item.final_record)

    by_field = {row["field"]: row["status"] for row in rows}
    assert by_field["check_out"] == "different"
    assert by_field["room_type"] == "same"
    assert all(set(row) == {"field", "status"} for row in rows)


def test_second_opinion_fails_closed_on_missing_ids() -> None:
    """Missing item, reviewer and approval IDs block the flow together."""
    state = evaluate_second_opinion(SecondOpinionRequest("", "", "", None), {}, frozenset())

    assert state.allowed is False
    assert state.audit_event is None
    assert set(state.reasons) == {
        "missing_item_id",
        "missing_primary_reviewer_id",
        "missing_second_reviewer_id",
        "missing_approval_id",
        "unknown_item_id",
    }
    assert set(state.reasons) <= REASON_CODES


def test_second_opinion_fails_closed_on_same_reviewer_and_bad_approval(
    booking_descriptor: FormDescriptor,
) -> None:
    """A repeated reviewer and an unknown approval token block the flow."""
    item = load_review_items()[0]
    submission = ReviewSubmission(
        item_id=item.item_id,
        reviewer_id=item.reviewer_id,
        routing="both_fine",
        slots=dict(item.final_record),
        completed=True,
    )
    request = SecondOpinionRequest(
        item.item_id,
        item.reviewer_id,
        item.reviewer_id,
        "approval-token-9999",
    )

    state = evaluate_second_opinion(
        request,
        {item.item_id: submission},
        frozenset({"approval-token-0002"}),
        booking_descriptor,
    )

    assert state.allowed is False
    assert state.audit_event is None
    assert "same_reviewer" in state.reasons
    assert "unrecognised_approval" in state.reasons


def test_second_opinion_allows_only_with_valid_completed_submission(
    booking_descriptor: FormDescriptor,
) -> None:
    """A completed, schema-valid primary review with distinct IDs and approval passes."""
    item = load_review_items()[0]
    submission = ReviewSubmission(
        item_id=item.item_id,
        reviewer_id=item.reviewer_id,
        routing="parser_wrong",
        slots=dict(item.final_record),
        completed=True,
    )
    request = SecondOpinionRequest(
        item.item_id,
        item.reviewer_id,
        "reviewer-token-02",
        "approval-token-0002",
    )

    state = evaluate_second_opinion(
        request,
        {item.item_id: submission},
        frozenset({"approval-token-0002"}),
        booking_descriptor,
    )

    assert state.allowed is True
    assert state.reasons == ()
    assert state.audit_event is not None
    assert state.audit_event["status"] == "second_opinion_requested"
    assert state.audit_event["approval_id"] == "approval-token-0002"
    assert state.audit_event["label"] == SYNTHETIC_LABEL
    assert_no_sensitive_text(state.to_dict())


def test_second_opinion_rejects_unvalidated_slots(
    booking_descriptor: FormDescriptor,
) -> None:
    """A submission with invalid slots cannot unlock a second opinion."""
    item = load_review_items()[0]
    invalid_slots = dict(item.final_record)
    invalid_slots["check_out"] = "2026-12-01"
    submission = ReviewSubmission(
        item_id=item.item_id,
        reviewer_id=item.reviewer_id,
        routing="both_fine",
        slots=invalid_slots,
        completed=True,
    )
    request = SecondOpinionRequest(
        item.item_id,
        item.reviewer_id,
        "reviewer-token-02",
        "approval-token-0002",
    )

    state = evaluate_second_opinion(
        request,
        {item.item_id: submission},
        frozenset({"approval-token-0002"}),
        booking_descriptor,
    )

    assert state.allowed is False
    assert "slot_validation_failed" in state.reasons
