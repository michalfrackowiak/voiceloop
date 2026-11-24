"""Deterministic paired bootstrap confidence intervals for evaluation metrics.

The harness resamples *turn indices*, using the same indices for champion and
challenger, so each replicate compares the two artifacts on the same turns.
Intervals therefore describe the paired difference ``challenger - champion``:
a negative WER delta and a positive E2E delta are improvements.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import TypeVar

from .metrics import EvaluationError

T = TypeVar("T")


@dataclass(frozen=True)
class ConfidenceInterval:
    """A bootstrap interval with both bounds and its confidence level."""

    lower: float
    upper: float
    level: float

    def __post_init__(self) -> None:
        """Reject impossible intervals before they reach the report writer."""
        if not 0 < self.level < 1:
            raise EvaluationError("confidence level must be strictly between 0 and 1")
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise EvaluationError("confidence bounds must be finite")
        if self.lower > self.upper:
            raise EvaluationError("confidence interval lower bound exceeds upper bound")

    def to_dict(self) -> dict[str, float]:
        """Return the contract shape for ``confidence_intervals`` entries."""
        return {"lower": self.lower, "upper": self.upper, "level": self.level}


def percentile_interval(values: Sequence[float], level: float = 0.95) -> ConfidenceInterval:
    """Nearest-rank percentile interval of a sample.

    With ``n`` sorted values and ``alpha = 1 - level``, the bounds are at
    indices ``floor(alpha / 2 * n)`` and ``ceil((1 - alpha / 2) * n) - 1``,
    clamped to the sample range.

    Args:
        values: Non-empty sample of scalar statistics.
        level: Confidence level in ``(0, 1)``.

    Returns:
        A :class:`ConfidenceInterval` whose lower bound never exceeds its upper.

    """
    if not values:
        raise EvaluationError("cannot build a confidence interval from zero samples")
    if not 0 < level < 1:
        raise EvaluationError("confidence level must be strictly between 0 and 1")
    ordered = sorted(values)
    alpha = 1.0 - level
    n = len(ordered)
    lower_index = max(0, min(n - 1, math.floor(alpha / 2 * n)))
    upper_index = max(0, min(n - 1, math.ceil((1 - alpha / 2) * n) - 1))
    return ConfidenceInterval(lower=ordered[lower_index], upper=ordered[upper_index], level=level)


def bootstrap_deltas(
    champion: Sequence[T],
    challenger: Sequence[T],
    statistic: Callable[[Sequence[T]], float] | None = None,
    *,
    iterations: int = 1000,
    seed: int = 0,
) -> list[float]:
    """Draw paired resamples and return the statistic differences.

    Args:
        champion: Champion per-turn values, in fixture order.
        challenger: Challenger per-turn values aligned by turn index.
        statistic: Function reducing a resample to a scalar; defaults to mean.
        iterations: Number of paired resamples, each of length ``len(champion)``.
        seed: Seed for a local ``random.Random`` instance (no global RNG state).

    Returns:
        ``statistic(challenger_resample) - statistic(champion_resample)`` for
        every iteration, in draw order.

    """
    if len(champion) != len(challenger) or not champion:
        raise EvaluationError("champion and challenger must be non-empty and equal length")
    if iterations < 1:
        raise EvaluationError("bootstrap iterations must be at least 1")

    reduce_values = statistic or _mean
    rng = random.Random(seed)
    size = len(champion)
    deltas: list[float] = []
    for _ in range(iterations):
        indices = [rng.randrange(size) for _ in range(size)]
        champion_sample = [champion[index] for index in indices]
        challenger_sample = [challenger[index] for index in indices]
        deltas.append(reduce_values(challenger_sample) - reduce_values(champion_sample))
    return deltas


def paired_bootstrap_ci(
    champion: Sequence[T],
    challenger: Sequence[T],
    statistic: Callable[[Sequence[T]], float] | None = None,
    *,
    level: float = 0.95,
    iterations: int = 1000,
    seed: int = 0,
) -> ConfidenceInterval:
    """Return a paired-bootstrap CI for ``challenger - champion``.

    The same turn indices are drawn for both sequences in every replicate, so
    turn-level difficulty cancels out. The interval is deterministic for a
    given seed, iteration count and input order.
    """
    deltas = bootstrap_deltas(champion, challenger, statistic, iterations=iterations, seed=seed)
    return percentile_interval(deltas, level)


def _mean(values: Sequence[T]) -> float:
    """Default statistic: arithmetic mean of a non-empty numeric sample."""
    return fmean(values)  # type: ignore[arg-type]
