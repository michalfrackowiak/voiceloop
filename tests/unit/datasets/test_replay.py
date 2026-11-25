"""Tests for deterministic replay sampling from earlier verified turns."""

from __future__ import annotations

from services.datasets import ReviewedItem, is_verified_replay_turn, sample_replay


def _verified_pool(reviewed_items: tuple[ReviewedItem, ...]) -> tuple[ReviewedItem, ...]:
    """Return only earlier verified fixture turns for the replay pool."""
    return tuple(
        item for item in reviewed_items if is_verified_replay_turn(item) and item.turn_index < 2
    )


def test_replay_is_seeded_deterministic_and_verified_only(
    reviewed_items: tuple[ReviewedItem, ...],
) -> None:
    """The same seed selects the same earlier verified turns, without repeats."""
    pool = _verified_pool(reviewed_items)
    first = sample_replay(pool, size=3, seed=7)
    second = sample_replay(tuple(reversed(pool)), size=3, seed=7)
    assert first.items == second.items
    assert first.seed == 7
    assert first.requested == 3
    assert first.candidate_count == len(pool)
    assert len({item.source_id for item in first.items}) == len(first.items)
    assert all(is_verified_replay_turn(item) for item in first.items)
    assert [item.source_id for item in first.items] == sorted(
        item.source_id for item in first.items
    )


def test_replay_respects_turn_bound_and_pool_size(
    reviewed_items: tuple[ReviewedItem, ...],
) -> None:
    """Asking for more turns than available returns the whole earlier pool."""
    pool = _verified_pool(reviewed_items)
    selection = sample_replay(pool, size=100, seed=1, before_turn_index=1)
    assert selection.items == tuple(item for item in pool if item.turn_index == 0)
    assert sample_replay(pool, size=0, seed=1).items == ()
