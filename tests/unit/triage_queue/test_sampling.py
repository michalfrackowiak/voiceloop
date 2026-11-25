"""Random-slice and blind-item sampling tests for P2-02."""

from __future__ import annotations

import math

from services.triage.queue import (
    TriageQueueSettings,
    rng_for_seed,
    select_blind_items,
    select_random_slice,
    slice_target,
)


def test_slice_target_uses_fraction_for_large_pools() -> None:
    settings = TriageQueueSettings(random_slice_fraction=0.04)
    assert slice_target(200, settings) == 8
    assert slice_target(0, settings) == 0


def test_slice_target_never_drops_all_turns_from_a_small_pool() -> None:
    settings = TriageQueueSettings(random_slice_fraction=0.04)
    assert slice_target(10, settings) == 1


def test_small_pool_target_override() -> None:
    settings = TriageQueueSettings(random_slice_target=2)
    assert slice_target(36, settings) == 2
    assert slice_target(1, settings) == 1


def test_slice_is_deterministic_for_a_seed() -> None:
    ids = [f"event-{index:03d}" for index in range(50)]
    settings = TriageQueueSettings(random_slice_target=5, random_seed=11)
    first = select_random_slice(ids, settings)
    second = select_random_slice(list(reversed(ids)), settings)
    assert first == second
    assert len(first) == 5
    assert list(first) == sorted(first)


def test_slice_changes_with_seed_and_duplicates_are_ignored() -> None:
    ids = [f"event-{index:03d}" for index in range(50)]
    first = select_random_slice(ids, TriageQueueSettings(random_slice_target=5, random_seed=1))
    second = select_random_slice(ids, TriageQueueSettings(random_slice_target=5, random_seed=2))
    assert first != second
    assert len(select_random_slice(ids + ids[:3], TriageQueueSettings(random_slice_target=5))) == 5


def test_sampler_never_receives_scores_or_labels() -> None:
    # The public sampler signature only accepts IDs: there is no path for a
    # failure score or a fixture label to influence selection.
    ids = ("clean-0", "failure-0", "clean-1", "failure-1")
    selected = select_random_slice(
        ids,
        TriageQueueSettings(random_slice_target=2, random_seed=99),
        rng=rng_for_seed(99, "slice"),
    )
    assert set(selected) <= set(ids)
    assert len(selected) == 2


def test_blind_selection_uses_configured_share_and_seed() -> None:
    ids = [f"item-{index}" for index in range(12)]
    first = select_blind_items(ids, 0.25, rng=rng_for_seed(4, "blind"))
    second = select_blind_items(ids, 0.25, rng=rng_for_seed(4, "blind"))
    assert first == second
    assert len(first) == math.ceil(12 * 0.25)
    assert select_blind_items(ids, 0.0, rng=rng_for_seed(4, "blind")) == frozenset()


def test_blind_share_marks_at_least_one_item_when_positive() -> None:
    selected = select_blind_items(["only"], 0.01, rng=rng_for_seed(1, "blind"))
    assert selected == frozenset({"only"})
