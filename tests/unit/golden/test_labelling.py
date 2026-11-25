"""P1-06.2: tool-agnostic export, import and distinct second reviewer."""

from __future__ import annotations

import pytest

from services.datasets.golden import (
    GoldenSettings,
    InMemoryLabelTool,
    LabelResult,
    SourceTurn,
    build_label_tool,
    compute_coverage,
    export_to_tool,
    import_labels,
    import_second_reviews,
    record_second_review,
    synthetic_labels_from_turns,
)
from voiceloop_common import ConfigError, ValidationError


def _result(source_id: str, reviewer_id: str, transcript: str = "hello") -> LabelResult:
    """Return a label with empty slots (no bookable slot established)."""
    return LabelResult(
        source_id=source_id,
        reviewer_id=reviewer_id,
        corrected_transcript=transcript,
        slots={},
    )


def test_export_contract_fields(fixture_turns: tuple[SourceTurn, ...]) -> None:
    tool = InMemoryLabelTool()
    item_ids = export_to_tool(fixture_turns[:2], tool)
    assert len(item_ids) == 2
    pulled = tool.pull(item_ids)
    assert set(pulled) == set(item_ids)


def test_export_and_import_labels(fixture_turns: tuple[SourceTurn, ...]) -> None:
    tool = InMemoryLabelTool()
    item_ids = export_to_tool(fixture_turns[:2], tool)
    tool.put_result(
        item_ids[0],
        reviewer_id="reviewer-a",
        corrected_transcript="corrected one",
        slots={},
        routing="both_fine",
    )
    tool.put_result(
        item_ids[1], reviewer_id="reviewer-a", corrected_transcript="corrected two", slots={}
    )
    imported = import_labels(tool, item_ids)
    assert imported[item_ids[0]].corrected_transcript == "corrected one"
    assert imported[item_ids[1]].reviewer_id == "reviewer-a"


def test_import_requires_a_label_for_every_item(fixture_turns: tuple[SourceTurn, ...]) -> None:
    tool = InMemoryLabelTool()
    item_ids = export_to_tool(fixture_turns[:2], tool)
    tool.put_result(item_ids[0], reviewer_id="reviewer-a", corrected_transcript="x", slots={})
    with pytest.raises(ValidationError, match="no label imported"):
        import_labels(tool, item_ids)


def test_invalid_slots_and_unknown_routing_are_rejected() -> None:
    with pytest.raises(ValidationError):
        LabelResult(
            source_id="call-1:0", reviewer_id="r", corrected_transcript="x", slots={"guests": 2}
        )
    with pytest.raises(ValidationError, match="routing"):
        LabelResult(
            source_id="call-1:0",
            reviewer_id="r",
            corrected_transcript="x",
            slots={},
            routing="unknown",
        )


def test_second_review_requires_a_distinct_reviewer() -> None:
    primary = _result("call-1:0", "reviewer-a")
    with pytest.raises(ValidationError, match="distinct"):
        record_second_review(primary, _result("call-1:0", "reviewer-a"))


def test_agreement_and_disagreement_are_recorded() -> None:
    primary = _result("call-1:0", "reviewer-a", "the room is deluxe")
    agree = record_second_review(primary, _result("call-1:0", "reviewer-b", "the room is deluxe"))
    assert agree.agreed and agree.transcript_agreed and agree.slots_agreed
    different = record_second_review(
        primary, _result("call-1:0", "reviewer-b", "the room is standard")
    )
    assert not different.agreed
    assert not different.transcript_agreed
    assert different.slots_agreed


def test_import_second_reviews_requires_two_submissions(
    fixture_turns: tuple[SourceTurn, ...],
) -> None:
    tool = InMemoryLabelTool()
    item_ids = export_to_tool(fixture_turns[:1], tool)
    tool.put_result(item_ids[0], reviewer_id="reviewer-a", corrected_transcript="x", slots={})
    primaries = import_labels(tool, item_ids)
    with pytest.raises(ValidationError, match="missing second review"):
        import_second_reviews(tool, item_ids, primaries)
    tool.put_result(item_ids[0], reviewer_id="reviewer-b", corrected_transcript="x", slots={})
    assert import_second_reviews(tool, item_ids, primaries)[item_ids[0]].agreed


def test_coverage_counts_reviews_and_disagreements(
    fixture_turns: tuple[SourceTurn, ...],
) -> None:
    labels = synthetic_labels_from_turns(
        fixture_turns[:3], primary_reviewer_id="a", second_reviewer_id="b"
    )
    seconds = {
        label.source_id: record_second_review(
            _result(label.source_id, "a", label.corrected_transcript),
            _result(
                label.source_id,
                "b",
                label.corrected_transcript if label.turn_index < 2 else "disagree",
            ),
        )
        for label in labels
    }
    coverage = compute_coverage(labels, seconds)
    assert coverage.total == 3
    assert coverage.second_review_count == 3
    assert coverage.agreement_count == 2
    assert coverage.disagreement_count == 1
    assert coverage.ratio == 1.0


def test_fake_tool_is_selected_by_settings() -> None:
    tool = build_label_tool(GoldenSettings(label_tool="in_memory"))
    assert isinstance(tool, InMemoryLabelTool)
    with pytest.raises(ConfigError, match="unsupported label tool"):
        build_label_tool(GoldenSettings().model_copy(update={"label_tool": "label_studio"}))
