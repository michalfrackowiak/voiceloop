"""Call-level dataset splitting with golden and regression exclusion (P2-05.2).

The split unit is always the call ID (context.md §2 invariant 7): no utterance
from a call may appear in two splits. Golden and regression calls are removed
before assignment and recorded in :class:`SplitPlan` so callers can assert that
they never overlap with train/val/test.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

from voiceloop_common import ValidationError


@dataclass(frozen=True, slots=True)
class SplitPlan:
    """Deterministic call-ID assignment plus the excluded call sets."""

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    golden: tuple[str, ...]
    regression: tuple[str, ...]

    def to_manifest_splits(self) -> dict[str, list[str]]:
        """Return the train/val/test arrays required by the manifest contract."""
        return {
            "train": list(self.train),
            "val": list(self.val),
            "test": list(self.test),
        }

    def assigned_calls(self) -> set[str]:
        """Return every call assigned to train, val or test."""
        return set(self.train) | set(self.val) | set(self.test)


def assign_calls(
    call_ids: Iterable[str],
    *,
    golden_call_ids: Iterable[str] = (),
    regression_call_ids: Iterable[str] = (),
    val_fraction: float = 0.2,
    test_fraction: float = 0.1,
    seed: int = 0,
) -> SplitPlan:
    """Assign whole calls to train, val and test with a seeded RNG.

    Args:
        call_ids: Call IDs that contributed at least one example.
        golden_call_ids: Sealed golden calls; never assigned to a data split.
        regression_call_ids: Reviewed-failure calls; never assigned to a split.
        val_fraction: Fraction of eligible calls reserved for validation.
        test_fraction: Fraction of eligible calls reserved for test.
        seed: Deterministic seed; input order never affects the result.

    Returns:
        A plan whose five call sets are pairwise disjoint. With one eligible
        call only train is populated; with two, train and val; with three or
        more all three splits have at least one call.

    Raises:
        ValidationError: If fractions are negative or sum above one.

    """
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction > 1:
        raise ValidationError("split fractions must be non-negative and sum at most 1")
    candidates = set(call_ids)
    golden = candidates & set(golden_call_ids)
    regression = (candidates & set(regression_call_ids)) - golden
    eligible = sorted(candidates - golden - regression)

    order = list(eligible)
    random.Random(seed).shuffle(order)
    count = len(order)
    test_size = max(1, round(count * test_fraction)) if count >= 3 else 0
    val_size = max(1, round(count * val_fraction)) if count - test_size >= 2 else 0
    test = order[:test_size]
    val = order[test_size : test_size + val_size]
    train = order[test_size + val_size :]
    return SplitPlan(
        train=tuple(sorted(train)),
        val=tuple(sorted(val)),
        test=tuple(sorted(test)),
        golden=tuple(sorted(golden)),
        regression=tuple(sorted(regression)),
    )


def assert_no_call_in_multiple_splits(plan: SplitPlan) -> None:
    """Raise if any call is assigned to more than one of the five call sets.

    This is the runtime form of the property tested under
    ``tests/unit/datasets``; it enforces context.md §2 invariant 7 including
    the sealed golden and accumulated regression sets.
    """
    groups: dict[str, set[str]] = {
        "train": set(plan.train),
        "val": set(plan.val),
        "test": set(plan.test),
        "golden": set(plan.golden),
        "regression": set(plan.regression),
    }
    names = list(groups)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            overlap = groups[first] & groups[second]
            if overlap:
                raise ValidationError(
                    f"call IDs appear in both {first} and {second}: " + ", ".join(sorted(overlap))
                )
