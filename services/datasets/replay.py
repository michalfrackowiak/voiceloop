"""Deterministic replay sampling from earlier verified turns (P2-05.4).

Replay turns are mixed into training to limit forgetting (context.md §7 tier
3). Only already-reviewed, non-failure turns are eligible, the pool is sorted
by ``(call_id, turn_index)`` before sampling, and a seed makes selection
reproducible. The sampler never invents new turns.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

from .examples import REVIEW_OK_LABELS, ReviewedItem


@dataclass(frozen=True, slots=True)
class ReplaySelection:
    """A seeded replay sample plus its audit metadata."""

    items: tuple[ReviewedItem, ...]
    seed: int
    requested: int
    candidate_count: int


def is_verified_replay_turn(item: ReviewedItem) -> bool:
    """Return whether a reviewed item is an earlier verified (non-failure) turn."""
    label = item.failure_type or item.routing
    return label in REVIEW_OK_LABELS


def sample_replay(
    items: Iterable[ReviewedItem],
    *,
    size: int,
    seed: int = 0,
    before_turn_index: int | None = None,
) -> ReplaySelection:
    """Select verified turns deterministically for replay mixing.

    Args:
        items: Earlier reviewed turns; duplicates by source ID are ignored.
        size: Requested number of turns; sampling returns fewer if the pool is small.
        seed: Seed for the deterministic RNG.
        before_turn_index: Optional upper bound on turn index (exclusive).

    Returns:
        The selected items sorted by ``(call_id, turn_index)``.

    """
    unique: dict[str, ReviewedItem] = {item.source_id: item for item in items}
    candidates = [
        item
        for item in unique.values()
        if is_verified_replay_turn(item)
        and (before_turn_index is None or item.turn_index < before_turn_index)
    ]
    candidates.sort(key=lambda item: (item.call_id, item.turn_index))
    chosen = random.Random(seed).sample(candidates, min(max(size, 0), len(candidates)))
    return ReplaySelection(
        items=tuple(sorted(chosen, key=lambda item: (item.call_id, item.turn_index))),
        seed=seed,
        requested=size,
        candidate_count=len(candidates),
    )
