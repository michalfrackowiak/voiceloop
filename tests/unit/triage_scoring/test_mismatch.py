"""Parser-vs-record and hypothesis-vs-parser mismatch tests for P2-01.3."""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

from services.triage.scoring.mismatch import (
    detect_final_record_mismatch,
    detect_mismatches,
)
from services.triage.scoring.settings import TriageScoringSettings


def _event_by_case(
    case_id: str,
    load_labels: Callable[[], list[dict[str, Any]]],
    load_event: Callable[[str | Path], dict[str, Any]],
) -> dict[str, Any]:
    """Load one named fixture event."""
    label = next(row for row in load_labels() if row["case_id"] == case_id)
    return load_event(label["event_path"])


def test_final_record_mismatch_has_the_highest_weight_and_is_detected(
    load_labels: Callable[[], list[dict[str, Any]]],
    load_event: Callable[[str | Path], dict[str, Any]],
) -> None:
    """A derived mismatch outranks every other configured signal.

    No fixture turn has a final record that disagrees with its parser output;
    this test starts from a clean fixture and mutates one field explicitly.
    """
    settings = TriageScoringSettings()
    event = copy.deepcopy(
        next(
            load_event(row["event_path"])
            for row in load_labels()
            if row["case_id"] is None
            and row["turn_index"] == 5
            and row["call_id"] == "call-0002"
        )
    )
    assert detect_final_record_mismatch(event, settings) == []
    event["outcome"]["final_record"]["guests"] = 3
    observations = detect_final_record_mismatch(event, settings)
    assert [item.name for item in observations] == ["final_record_mismatch"]
    assert observations[0].weight == settings.weight_final_record_mismatch
    assert observations[0].weight > settings.weight_hypothesis_parser_mismatch
    assert observations[0].weight > settings.weight_low_word_confidence
    assert "guests" in observations[0].reason

    all_signals = detect_mismatches(event, settings)
    assert "final_record_mismatch" in {item.name for item in all_signals}


def test_date_confusion_fires_guest_and_nights_mismatches(
    load_labels: Callable[[], list[dict[str, Any]]],
    load_event: Callable[[str | Path], dict[str, Any]],
) -> None:
    """The parser-only date error is visible without fixture labels."""
    event = _event_by_case("date_confusion", load_labels, load_event)
    names = {item.name for item in detect_mismatches(event, TriageScoringSettings())}
    assert names == {"hypothesis_guest_count_mismatch", "hypothesis_nights_mismatch"}


def test_nights_misparsed_fires_nights_mismatch(
    load_labels: Callable[[], list[dict[str, Any]]],
    load_event: Callable[[str | Path], dict[str, Any]],
) -> None:
    """A stated four-night stay parsed as three nights is detected."""
    event = _event_by_case("nights_misparsed", load_labels, load_event)
    names = {item.name for item in detect_mismatches(event, TriageScoringSettings())}
    assert names == {"hypothesis_nights_mismatch"}


def test_correction_turn_does_not_fire_a_false_mismatch(
    load_labels: Callable[[], list[dict[str, Any]]],
    load_event: Callable[[str | Path], dict[str, Any]],
) -> None:
    """'three nights, not two' must not be read as a two-night request."""
    event = next(
        load_event(row["event_path"])
        for row in load_labels()
        if row["call_id"] == "call-0001" and row["turn_index"] == 3
    )
    assert detect_mismatches(event, TriageScoringSettings()) == []


def test_no_final_record_means_no_mismatch_signal(
    load_labels: Callable[[], list[dict[str, Any]]],
    load_event: Callable[[str | Path], dict[str, Any]],
) -> None:
    """Late outcomes are never inferred from an absent final_record."""
    event = _event_by_case("misheard_surname", load_labels, load_event)
    assert "outcome" not in event
    assert detect_final_record_mismatch(event, TriageScoringSettings()) == []
