"""Seeded random-slice and blind-item sampling for the review queue.

Both samplers operate only on stable identifier lists: they never see a failure
score, a fixture failure label or a transcript. This is what makes the random
slice unbiased (``context.md`` §2 invariant 6). Separate named RNG streams keep
slice selection independent of blind-item selection.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Literal

from .settings import TriageQueueSettings

RANDOM_SLICE: Literal["random"] = "random"
FAILURE_SLICE: Literal["failure"] = "failure"
SliceTag = Literal["random", "failure"]


def rng_for_seed(seed: int, stream: str) -> random.Random:
    """Return a seeded RNG for one named sampling stream.

    Args:
        seed: Base integer seed, normally from settings.
        stream: Stable name such as ``slice`` or ``blind``.

    Returns:
        A fresh, deterministic ``random.Random`` instance.

    """
    return random.Random(f"{seed}:{stream}")


def slice_target(count: int, settings: TriageQueueSettings) -> int:
    """Resolve the number of random-slice turns for a pool of ``count`` turns.

    The fraction path is used in production. For a non-empty pool it never
    returns zero, because at least one turn must bypass the failure filter;
    therefore an exact 3–5% share is only representable when ``count *
    fraction`` is an integer. Tests and tiny rehearsals can set an explicit
    ``random_slice_target`` and record the deviation in their handoff.

    Args:
        count: Number of eligible turns.
        settings: Queue settings holding the fraction or explicit target.

    Returns:
        A clamped, non-negative target count.

    """
    if count <= 0:
        return 0
    if settings.random_slice_target is not None:
        return min(count, settings.random_slice_target)
    return min(count, max(1, round(count * settings.random_slice_fraction)))


def select_random_slice(
    turn_ids: Sequence[str],
    settings: TriageQueueSettings,
    *,
    rng: random.Random | None = None,
) -> tuple[str, ...]:
    """Select the unbiased random slice from all eligible turn IDs.

    The input is deliberately just IDs: no scores, labels or text are consulted.
    Selection is by seeded RNG, and the returned IDs are sorted for stable
    downstream ordering regardless of RNG implementation details.

    Args:
        turn_ids: All eligible turn IDs, in any order.
        settings: Queue settings holding the fraction/target and seed.
        rng: Optional injected RNG; defaults to the named ``slice`` stream.

    Returns:
        Selected event IDs in ascending order.

    """
    unique_ids = sorted(set(turn_ids))
    target = slice_target(len(unique_ids), settings)
    if target == 0:
        return ()
    generator = rng if rng is not None else rng_for_seed(settings.random_seed, "slice")
    return tuple(sorted(generator.sample(unique_ids, target)))


def select_blind_items(
    item_ids: Sequence[str],
    blind_share: float,
    *,
    rng: random.Random,
) -> frozenset[str]:
    """Choose which queued items are blind (no parser prefill).

    Args:
        item_ids: Final, ordered queue item IDs.
        blind_share: Desired share in ``[0, 1]``; rounded up so any positive
            share marks at least one item when the queue is non-empty.
        rng: Injected RNG, normally the named ``blind`` stream.

    Returns:
        The selected event IDs as a frozenset.

    """
    ids = tuple(item_ids)
    if not ids or blind_share <= 0.0:
        return frozenset()
    count = min(len(ids), math.ceil(len(ids) * blind_share))
    return frozenset(rng.sample(list(ids), count))
