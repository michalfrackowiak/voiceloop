"""Repair-behaviour detector tests for P2-01.2."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from services.triage.scoring.repair import (
    build_repair_context,
    detect_repair,
    load_prior_turns,
)
from services.triage.scoring.settings import TriageScoringSettings


def _event(text: str, status: str | None = None) -> dict[str, Any]:
    """Build a minimal schema-shaped event for repair tests."""
    event: dict[str, Any] = {"stt": {"text": text}}
    if status is not None:
        event["outcome"] = {"status": status}
    return event


def test_guest_correction_phrase_is_detected() -> None:
    """An explicit self-correction fires the guest_correction signal."""
    observations = detect_repair(
        _event("No, I said Smith."), TriageScoringSettings()
    )
    assert [item.name for item in observations] == ["guest_correction"]
    assert "correction" in observations[0].reason


def test_agent_reask_is_detected_from_dialogue_context() -> None:
    """Agent re-ask phrases are read from supplied prior-turn records."""
    context = build_repair_context(
        _event("Smith."),
        [{"speaker": "agent", "agent_text": "Could you repeat the name?"}],
    )
    observations = detect_repair(_event("Smith."), TriageScoringSettings(), context)
    assert [item.name for item in observations] == ["agent_reask"]


def test_transfer_is_detected_from_outcome_status() -> None:
    """A late transferred outcome records a human handoff."""
    observations = detect_repair(_event("Hello.", "transferred"), TriageScoringSettings())
    assert [item.name for item in observations] == ["transfer"]


def test_prior_turns_ref_loads_fixture_context(
    repo_root: Path,
    load_labels: Callable[[], list[dict[str, Any]]],
    load_event: Callable[[str | Path], dict[str, Any]],
) -> None:
    """The read-only JSONL context is resolved relative to the repo root."""
    label = next(row for row in load_labels() if row["case_id"] == "misheard_surname")
    event = load_event(label["event_path"])
    prior_turns = load_prior_turns(event, root=repo_root)
    assert len(prior_turns) == 6
    context = build_repair_context(event, prior_turns)
    assert len(context.prior_guest_texts) == 6
    assert context.prior_guest_texts[2] == "Mr. Smythe."


def test_clean_fixture_turns_have_no_repair_signal(
    load_labels: Callable[[], list[dict[str, Any]]],
    load_event: Callable[[str | Path], dict[str, Any]],
) -> None:
    """Ordinary slot updates are not misclassified as repairs."""
    for label in load_labels():
        if label["failure_type"] not in {"clean", "both_fine"}:
            continue
        event = load_event(label["event_path"])
        assert detect_repair(event, TriageScoringSettings()) == [], label
