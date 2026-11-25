"""Property and unit tests for call-level splitting (context.md §2 invariant 7)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from services.datasets import SplitPlan, assert_no_call_in_multiple_splits, assign_calls
from voiceloop_common import ValidationError

TURNS_PATH = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "turns.jsonl"
ROWS = [json.loads(line) for line in TURNS_PATH.read_text(encoding="utf-8").splitlines() if line]
CALL_IDS = tuple(sorted({str(row["call_id"]) for row in ROWS}))


def test_fixture_pool_has_six_calls_and_36_turns() -> None:
    """The property test below draws its call IDs from the real fixture pool."""
    assert len(ROWS) == 36
    assert len(CALL_IDS) == 6


@given(
    golden=st.lists(st.sampled_from(CALL_IDS), unique=True, max_size=6),
    regression=st.lists(st.sampled_from(CALL_IDS), unique=True, max_size=6),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_no_call_appears_in_more_than_one_split(
    golden: list[str], regression: list[str], seed: int
) -> None:
    """Property: no call is in two of train/val/test/golden/regression."""
    plan = assign_calls(
        CALL_IDS,
        golden_call_ids=golden,
        regression_call_ids=regression,
        seed=seed,
    )
    assert_no_call_in_multiple_splits(plan)
    groups = [
        set(plan.train),
        set(plan.val),
        set(plan.test),
        set(plan.golden),
        set(plan.regression),
    ]
    assert sum(len(group) for group in groups) == len(set().union(*groups))
    assert plan.assigned_calls() == set(CALL_IDS) - set(golden) - set(regression)
    assert plan.to_manifest_splits().keys() == {"train", "val", "test"}
    if plan.train:
        bad = SplitPlan(
            train=plan.train,
            val=plan.train,
            test=(),
            golden=(),
            regression=(),
        )
        with pytest.raises(ValidationError, match="both train and val"):
            assert_no_call_in_multiple_splits(bad)


def test_split_assignment_is_seed_stable_and_order_independent() -> None:
    """Sorting plus a seeded RNG makes assignment deterministic."""
    first = assign_calls(CALL_IDS, seed=7)
    second = assign_calls(tuple(reversed(CALL_IDS)), seed=7)
    assert first == second
    assert len(first.train) == 4
    assert len(first.val) == 1
    assert len(first.test) == 1
    assert first.assigned_calls() == set(CALL_IDS)


def test_golden_takes_precedence_and_small_pools_are_handled() -> None:
    """Overlapping exclusions stay disjoint and tiny pools never split a call twice."""
    plan = assign_calls(
        CALL_IDS,
        golden_call_ids=["call-0001"],
        regression_call_ids=["call-0001", "call-0002"],
    )
    assert plan.golden == ("call-0001",)
    assert plan.regression == ("call-0002",)
    assert assign_calls([]).assigned_calls() == set()
    assert assign_calls(["only-call"]).train == ("only-call",)
    tiny = assign_calls(["a", "b"])
    assert len(tiny.val) == 1
    assert len(tiny.train) == 1
    assert tiny.test == ()


def test_invalid_fractions_are_rejected() -> None:
    """Negative or over-allocated fractions fail closed."""
    with pytest.raises(ValidationError, match="fractions"):
        assign_calls(CALL_IDS, val_fraction=-0.1)
    with pytest.raises(ValidationError, match="fractions"):
        assign_calls(CALL_IDS, val_fraction=0.8, test_fraction=0.4)
