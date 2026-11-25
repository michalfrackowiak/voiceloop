"""Signal extractor tests for P2-01.1 and P2-01.4."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from services.triage.scoring.settings import TriageScoringSettings
from services.triage.scoring.signals import (
    FakeTeacherModel,
    TeacherModel,
    TeacherVerdict,
    extract_parser_signals,
    extract_signals,
    teacher_observation,
)


class DisagreeingTeacher:
    """Stub teacher that disagrees on every turn."""

    def judge(self, event: dict[str, Any]) -> TeacherVerdict:
        """Always report a disagreement."""
        return TeacherVerdict(disagrees=True, reason="teacher heard a different surname")


def test_misheard_surname_fires_confidence_and_decoder_signals(
    load_labels: Callable[[], list[dict[str, Any]]],
    load_event: Callable[[str | Path], dict[str, Any]],
) -> None:
    """The named STT failure carries two independent low-level signals."""
    label = next(row for row in load_labels() if row["case_id"] == "misheard_surname")
    event = load_event(label["event_path"])
    observations = extract_signals(event, TriageScoringSettings())
    names = {item.name for item in observations}
    assert {"low_word_confidence", "decoder_stats"} <= names
    low_conf = next(item for item in observations if item.name == "low_word_confidence")
    assert "0.41" in low_conf.reason
    assert 0.0 < low_conf.severity <= 1.0


def test_no_slot_parse_is_exempt_from_schema_and_logprob_signals(
    load_labels: Callable[[], list[dict[str, Any]]],
    load_event: Callable[[str | Path], dict[str, Any]],
) -> None:
    """Greeting turns legitimately carry schema_valid=false with empty output."""
    label = next(
        row
        for row in load_labels()
        if row["failure_type"] == "clean" and row["turn_index"] == 0
    )
    event = load_event(label["event_path"])
    assert event["parser"]["schema_valid"] is False
    assert event["parser"]["output"] == {}
    assert extract_parser_signals(event, TriageScoringSettings()) == []


def test_schema_invalid_and_low_parse_probability_fire_on_named_failure(
    load_labels: Callable[[], list[dict[str, Any]]],
    load_event: Callable[[str | Path], dict[str, Any]],
) -> None:
    """The schema-invalid fixture produces both parser signals."""
    label = next(row for row in load_labels() if row["case_id"] == "schema_invalid_parse")
    observations = extract_parser_signals(load_event(label["event_path"]), TriageScoringSettings())
    assert {item.name for item in observations} == {"schema_invalid", "low_parse_probability"}


def test_missing_optional_fields_do_not_invent_signals() -> None:
    """A minimal valid event with no optional fields fires nothing."""
    event = {"event_id": "x", "call_id": "c", "turn_index": 0, "stt": {"text": "Hello."}}
    assert extract_signals(event, TriageScoringSettings()) == []


def test_thresholds_are_configurable() -> None:
    """Raising a threshold changes which signals fire."""
    event = {
        "stt": {
            "text": "Hello.",
            "words": [{"w": "Hello.", "conf": 0.92, "start_ms": 0, "end_ms": 10}],
        }
    }
    assert extract_signals(event, TriageScoringSettings()) == []
    strict = TriageScoringSettings(min_word_conf=0.95, mean_word_conf=0.99)
    observations = extract_signals(event, strict)
    assert [item.name for item in observations] == ["low_word_confidence"]


def test_teacher_signal_is_off_by_default_and_injectable() -> None:
    """The fake teacher agrees; an injected disagreeing teacher fires when enabled."""
    event = {"stt": {"text": "Mr. Smythe."}}
    teacher: TeacherModel = DisagreeingTeacher()
    assert teacher_observation(event, TriageScoringSettings(), teacher) == []
    enabled = TriageScoringSettings(teacher_enabled=True)
    observations = teacher_observation(event, enabled, teacher)
    assert [item.name for item in observations] == ["teacher_disagreement"]
    assert observations[0].weight == enabled.weight_teacher_disagreement
    assert teacher_observation(event, enabled, FakeTeacherModel()) == []
