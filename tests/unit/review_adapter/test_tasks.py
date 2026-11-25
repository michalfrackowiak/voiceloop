"""Two-stage orchestration, schema-driven forms, prefill, warnings and idempotency."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from services.review_adapter import (
    ExplanationSpanSource,
    InMemoryReviewStore,
    InMemoryReviewTool,
    LowConfidenceSpan,
    QueueItemSpanSource,
    ReviewAdapterError,
    ReviewAdapterSettings,
    ReviewCoordinator,
    ReviewTool,
    SimpleQueueItem,
    Stage1Answer,
    Stage2Answer,
    build_review_task,
    build_slot_form,
)
from tests.unit.review_adapter.conftest import (
    CLOCK,
    queue_item,
    scripted_review,
)


def as_dicts(spans: Iterable[Any]) -> list[dict[str, Any]]:
    """Normalise P2-02 and adapter span types to contract dictionaries."""
    return [span.to_dict() for span in spans]


class RecordingTool:
    """Delegate to a fake tool while recording stage 2 prefill arguments."""

    def __init__(self, delegate: ReviewTool) -> None:
        self._delegate = delegate
        self.prefills: list[Mapping[str, Any] | None] = []

    def tool_name(self) -> str:
        return self._delegate.tool_name()

    def create_task(self, task: Any) -> str:
        return self._delegate.create_task(task)

    def pull_stage1(self, tool_task_id: str) -> Stage1Answer | None:
        return self._delegate.pull_stage1(tool_task_id)

    def prepare_stage2(self, tool_task_id: str, prefill: Mapping[str, Any] | None) -> None:
        self.prefills.append(prefill)
        self._delegate.prepare_stage2(tool_task_id, prefill)

    def pull_stage2(self, tool_task_id: str) -> Stage2Answer | None:
        return self._delegate.pull_stage2(tool_task_id)


class RecordingParser:
    """Deterministic parser stand-in recording the transcripts it was given."""

    def __init__(self, by_transcript: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        self._by_transcript = dict(by_transcript or {})
        self.transcripts: list[str] = []

    def parse(
        self, transcript: str, *, context: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        self.transcripts.append(transcript)
        return self._by_transcript.get(transcript, {})


class ExplodingParser:
    """Parser stand-in that always fails, with PII-like exception text."""

    def parse(
        self, transcript: str, *, context: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        raise RuntimeError("secret guest name in parser error")


class ExplanationStore:
    """Minimal P2-01-style store with an optional span-bearing explanation."""

    def __init__(self, explanation: object | None) -> None:
        self._explanation = explanation

    def get(self, event_id: str) -> object | None:
        return self._explanation


class PendingStage2Tool(RecordingTool):
    """Fake tool whose stage 2 answer stays pending."""

    def pull_stage2(self, tool_task_id: str) -> Stage2Answer | None:
        return None


def test_slot_form_is_generated_from_booking_schema(booking_schema: dict[str, Any]) -> None:
    """Every schema property, required flag, enum and nested field appears."""
    form = build_slot_form()
    assert {field.name for field in form.fields} == set(booking_schema["properties"])
    assert set(form.required) == set(booking_schema["required"])
    by_name = {field.name: field for field in form.fields}
    assert by_name["room_type"].enum == tuple(booking_schema["properties"]["room_type"]["enum"])
    assert by_name["guests"].minimum == 1
    assert by_name["guests"].required is True
    assert {field.name for field in by_name["contact"].properties} == {"email", "phone"}
    assert form.to_dict()["schema_version"] == booking_schema["schema_version"]


def test_slot_form_follows_schema_override(tmp_path: Path) -> None:
    """A project-specific schema path changes the generated form, not code."""
    path = tmp_path / "booking.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "9.9.9",
                "type": "object",
                "required": ["custom_slot"],
                "properties": {"custom_slot": {"type": "string", "enum": ["a", "b"]}},
            }
        ),
        encoding="utf-8",
    )
    coordinator = ReviewCoordinator(
        InMemoryReviewTool(), settings=ReviewAdapterSettings(booking_schema_path=str(path))
    )
    form = coordinator.slot_form
    assert [field.name for field in form.fields] == ["custom_slot"]
    assert form.required == ("custom_slot",)
    assert form.fields[0].enum == ("a", "b")
    assert form.schema_version == "9.9.9"


def test_highlights_are_copied_from_queue_item_not_fabricated() -> None:
    """A queue item's explicit spans are used verbatim; empty stays empty."""
    item = queue_item("misheard_surname")
    task = build_review_task(item, build_slot_form())
    assert item.low_confidence_spans
    assert as_dicts(task.low_confidence_spans) == as_dicts(item.low_confidence_spans)
    assert all(span.confidence < 0.60 for span in task.low_confidence_spans)
    bare = SimpleQueueItem("event-x", "call-x", 0, "s3://bucket/x.wav", "hi")
    assert build_review_task(bare, build_slot_form()).low_confidence_spans == ()


def test_explanation_spans_are_used_only_when_explicit() -> None:
    """A span-bearing explanation overrides the queue item; P2-01 today cannot."""
    item = queue_item("misheard_surname")
    explicit = [
        {"start_ms": 10, "end_ms": 20, "text": "Smith", "confidence": 0.31},
    ]
    enhanced = ExplanationSpanSource(
        ExplanationStore(type("E", (), {"low_confidence_spans": explicit})())
    )
    assert enhanced.spans_for(item) == (LowConfidenceSpan(10, 20, "Smith", 0.31),)
    plain = ExplanationSpanSource(ExplanationStore(object()))
    assert as_dicts(plain.spans_for(item)) == as_dicts(item.low_confidence_spans)


def test_invalid_explicit_span_is_rejected() -> None:
    """Malformed span data fails loudly rather than being silently invented."""
    from types import SimpleNamespace

    store = ExplanationStore(
        SimpleNamespace(low_confidence_spans=[{"start_ms": -1, "end_ms": 5, "confidence": 0.4}])
    )
    with pytest.raises(ReviewAdapterError, match="non-negative start_ms"):
        ExplanationSpanSource(store).spans_for(queue_item("misheard_surname"))


def test_prefill_runs_on_corrected_transcript() -> None:
    """Stage 2 prefill uses stage 1 output, never the original hypothesis."""
    corrected = "the name is Mr. Smith"
    slots = {"guest_name": "Mr. Smith"}
    item = queue_item("misheard_surname")
    script = scripted_review("misheard_surname", corrected_transcript=corrected)
    parser = RecordingParser({corrected: slots})
    tool = RecordingTool(InMemoryReviewTool({item.event_id: script}, clock=CLOCK))
    coordinator = ReviewCoordinator(tool, InMemoryReviewStore(), parser=parser)
    result = coordinator.run_item(item)
    assert parser.transcripts == [corrected]
    assert tool.prefills == [slots]
    assert result.corrected_transcript == corrected
    assert "parser_prefill_empty" not in result.warnings


def test_blind_item_skips_parser_and_prefill() -> None:
    """No parser call and a ``None`` prefill for blind items."""
    item = queue_item("misheard_surname", blind=True)
    script = scripted_review("misheard_surname")
    parser = ExplodingParser()
    tool = RecordingTool(InMemoryReviewTool({item.event_id: script}, clock=CLOCK))
    coordinator = ReviewCoordinator(tool, InMemoryReviewStore(), parser=parser)
    result = coordinator.run_item(item)
    assert tool.prefills == [None]
    assert result.blind is True
    assert not any(warning.startswith("parser_") for warning in result.warnings)


def test_parser_failure_is_recorded_as_warning() -> None:
    """Parser exception text never enters warnings, but the failure is visible."""
    item = queue_item("misheard_surname")
    coordinator = ReviewCoordinator(
        InMemoryReviewTool({item.event_id: scripted_review("misheard_surname")}, clock=CLOCK),
        InMemoryReviewStore(),
        parser=ExplodingParser(),
    )
    result = coordinator.run_item(item)
    assert "parser_failed:RuntimeError" in result.warnings
    assert "parser_prefill_empty" in result.warnings
    assert all("secret guest name" not in warning for warning in result.warnings)


def test_invalid_routing_label_is_rejected_before_storage() -> None:
    """Only the four contract routing labels may be stored."""
    item = queue_item("misheard_surname")
    store = InMemoryReviewStore()
    coordinator = ReviewCoordinator(
        InMemoryReviewTool(
            {item.event_id: scripted_review("misheard_surname", routing="maybe")}, clock=CLOCK
        ),
        store,
    )
    with pytest.raises(ReviewAdapterError, match="unknown routing label"):
        coordinator.run_item(item)
    assert store.all() == ()


def test_invalid_slots_are_stored_with_pii_free_warnings() -> None:
    """Human slots are authoritative even when schema-invalid; warnings are field-only."""
    item = queue_item("misheard_surname")
    invalid = {
        "check_in": "2026-12-05",
        "check_out": "2026-12-01",
        "guests": 0,
        "room_type": "standard",
        "guest_name": "Private Guest Name",
    }
    coordinator = ReviewCoordinator(
        InMemoryReviewTool(
            {item.event_id: scripted_review("misheard_surname", slots=invalid)}, clock=CLOCK
        ),
        InMemoryReviewStore(),
    )
    result = coordinator.run_item(item)
    assert "slots_schema_invalid:guests:minimum" in result.warnings
    assert "slots_cross_field_invalid:check_out_after_check_in" in result.warnings
    assert all("Private Guest Name" not in warning for warning in result.warnings)


def test_duplicate_push_and_pull_create_one_result() -> None:
    """Idempotent push and duplicate stage 2 pulls cannot duplicate a result."""
    item = queue_item("misheard_surname")
    script = scripted_review("misheard_surname")
    tool = InMemoryReviewTool({item.event_id: script}, clock=CLOCK)
    store = InMemoryReviewStore()
    coordinator = ReviewCoordinator(tool, store)
    item_id = coordinator.push(item)
    first_tool_task = coordinator.task_for(item_id)
    assert coordinator.push(item) == item_id
    coordinator.stage1(item_id)
    first = coordinator.stage2(item_id)
    again = coordinator.push(item)
    coordinator.stage1(again)
    second = coordinator.stage2(again)
    assert again == item_id
    assert first_tool_task == coordinator.task_for(again)
    assert first is second
    assert coordinator.pull_result(item_id) is first
    assert len(store.all()) == 1


def test_pending_stage_answers_are_explicit() -> None:
    """An unscripted tool returns pending rather than invented reviewer data."""
    item = queue_item("misheard_surname")
    coordinator = ReviewCoordinator(InMemoryReviewTool(), InMemoryReviewStore())
    item_id = coordinator.push(item)
    with pytest.raises(ReviewAdapterError, match="stage 1 answer is pending"):
        coordinator.stage1(item_id)

    script = scripted_review("misheard_surname")
    pending = PendingStage2Tool(InMemoryReviewTool({item.event_id: script}, clock=CLOCK))
    second = ReviewCoordinator(pending, InMemoryReviewStore())
    other_id = second.push(item)
    second.stage1(other_id)
    with pytest.raises(ReviewAdapterError, match="stage 2 answer is pending"):
        second.stage2(other_id)


def test_inverted_timestamps_generate_warning() -> None:
    """Out-of-order reviewer timestamps are flagged, not silently fixed."""

    class InvertedTool(RecordingTool):
        def pull_stage2(self, tool_task_id: str) -> Stage2Answer | None:
            answer = self._delegate.pull_stage2(tool_task_id)
            assert answer is not None
            return replace(
                answer,
                started_at=datetime(2026, 11, 25, 12, tzinfo=UTC),
                completed_at=datetime(2026, 11, 25, 11, tzinfo=UTC),
            )

    item = queue_item("misheard_surname")
    tool = InvertedTool(
        InMemoryReviewTool({item.event_id: scripted_review("misheard_surname")}, clock=CLOCK)
    )
    result = ReviewCoordinator(tool, InMemoryReviewStore()).run_item(item)
    assert "review_timestamps_out_of_order" in result.warnings


def test_queue_item_span_source_is_the_default() -> None:
    """The default span source preserves the P2-02 item's highlights."""
    item = queue_item("misheard_surname")
    assert as_dicts(QueueItemSpanSource().spans_for(item)) == as_dicts(item.low_confidence_spans)
