"""P1-06.1: deterministic, failure-blind, call-level sampling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.datasets.golden import (
    GoldenSettings,
    SamplingError,
    load_source_turns,
    sample_calls,
    select_entire_pool,
)


def test_production_default_target_is_300_to_500() -> None:
    settings = GoldenSettings()
    assert settings.target_min_utterances == 300
    assert settings.target_max_utterances == 500


def test_fixture_pool_has_36_turns_and_no_failure_field() -> None:
    turns = load_source_turns()
    assert len(turns) == 36
    assert len({turn.call_id for turn in turns}) == 6
    assert not hasattr(turns[0], "failure_type")


def test_sample_is_deterministic_and_call_level() -> None:
    turns = load_source_turns()
    first = sample_calls(turns, seed=7, target_min=12, target_max=18)
    second = sample_calls(turns, seed=7, target_min=12, target_max=18)
    assert first == second
    assert 12 <= len(first.turns) <= 18
    grouped: dict[str, int] = {}
    for turn in turns:
        grouped[turn.call_id] = grouped.get(turn.call_id, 0) + 1
    selected: dict[str, int] = {}
    for turn in first.turns:
        selected[turn.call_id] = selected.get(turn.call_id, 0) + 1
    assert selected == {call_id: grouped[call_id] for call_id in first.call_ids}


def test_different_seed_can_change_selection() -> None:
    turns = load_source_turns()
    selections = {
        sample_calls(turns, seed=seed, target_min=12, target_max=18).call_ids for seed in range(10)
    }
    assert len(selections) > 1


def test_failure_labels_cannot_influence_selection(tmp_path: Path) -> None:
    turns = load_source_turns()
    source = Path("tests/fixtures/turns.jsonl").read_text(encoding="utf-8").splitlines()
    rewritten = []
    for number, line in enumerate(source):
        row = json.loads(line)
        row["failure_type"] = "stt_wrong" if number % 2 else "clean"
        row["case_id"] = "injected"
        rewritten.append(json.dumps(row, sort_keys=True))
    path = tmp_path / "relabelled.jsonl"
    path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    assert load_source_turns(path) == turns
    assert sample_calls(turns, seed=3, target_min=12, target_max=18) == sample_calls(
        load_source_turns(path), seed=3, target_min=12, target_max=18
    )


def test_unreachable_target_raises_without_splitting_a_call() -> None:
    with pytest.raises(SamplingError, match="cannot reach"):
        sample_calls(load_source_turns(), seed=1, target_min=40, target_max=42)


def test_invalid_band_raises() -> None:
    with pytest.raises(SamplingError, match="target_min"):
        sample_calls(load_source_turns(), seed=1, target_min=20, target_max=10)


def test_select_entire_pool_is_the_documented_synthetic_standin() -> None:
    result = select_entire_pool(load_source_turns())
    assert len(result.turns) == 36
    assert result.call_ids == tuple(f"call-{number:04d}" for number in range(1, 7))
    assert result.turns[0].source_id == "call-0001:0"
