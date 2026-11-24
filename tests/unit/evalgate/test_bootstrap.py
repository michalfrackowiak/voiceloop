"""Hand-computed deterministic paired bootstrap cases."""

from __future__ import annotations

import random
from statistics import fmean

import pytest

from services.evalgate.bootstrap import (
    ConfidenceInterval,
    bootstrap_deltas,
    paired_bootstrap_ci,
    percentile_interval,
)
from services.evalgate.metrics import EvaluationError


def test_percentile_interval_hand_computed() -> None:
    """With four samples at level 0.5 the middle two ranks are the bounds."""
    interval = percentile_interval([3.0, 1.0, 0.0, 2.0], level=0.5)
    assert interval == ConfidenceInterval(lower=1.0, upper=2.0, level=0.5)
    assert interval.to_dict() == {"lower": 1.0, "upper": 2.0, "level": 0.5}


def test_constant_delta_interval() -> None:
    """A uniform +1 shift has a degenerate +1 interval for any resample."""
    interval = paired_bootstrap_ci(
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        iterations=10,
        seed=5,
    )
    assert interval.lower == pytest.approx(1.0)
    assert interval.upper == pytest.approx(1.0)
    assert interval.level == 0.95


def test_paired_deltas_share_indices_and_match_hand_computation() -> None:
    """Both artifacts use the same drawn indices in every replicate."""
    champion = [0.0, 1.0]
    challenger = [1.0, 0.0]
    deltas = bootstrap_deltas(champion, challenger, iterations=6, seed=3)

    rng = random.Random(3)
    expected: list[float] = []
    for _ in range(6):
        indices = [rng.randrange(2) for _ in range(2)]
        expected.append(
            fmean(challenger[index] for index in indices)
            - fmean(champion[index] for index in indices)
        )
    assert deltas == expected
    assert set(deltas) <= {-1.0, 0.0, 1.0}


def test_same_seed_reproduces_interval_and_bounds_are_ordered() -> None:
    """Seeding is local and deterministic; changing seeds stays valid."""
    champion = [0.0, 0.0, 0.0, 1.0]
    challenger = [1.0, 0.0, 0.0, 1.0]
    first = paired_bootstrap_ci(champion, challenger, iterations=50, seed=1)
    second = paired_bootstrap_ci(champion, challenger, iterations=50, seed=1)
    other = paired_bootstrap_ci(champion, challenger, iterations=50, seed=2)
    assert first == second
    assert first.lower <= first.upper
    assert other.lower <= other.upper
    assert first.level == other.level == 0.95


def test_invalid_bootstrap_inputs() -> None:
    """Empty/misaligned samples, bad iteration counts and bad CIs are rejected."""
    with pytest.raises(EvaluationError, match="non-empty"):
        paired_bootstrap_ci([], [], iterations=1)
    with pytest.raises(EvaluationError, match="equal length"):
        paired_bootstrap_ci([1.0], [1.0, 2.0], iterations=1)
    with pytest.raises(EvaluationError, match="at least 1"):
        paired_bootstrap_ci([1.0], [1.0], iterations=0)
    with pytest.raises(EvaluationError, match="zero samples"):
        percentile_interval([], 0.95)
    with pytest.raises(EvaluationError, match="lower bound exceeds"):
        ConfidenceInterval(lower=1.0, upper=0.0, level=0.95)
    with pytest.raises(EvaluationError, match="strictly between"):
        ConfidenceInterval(lower=0.0, upper=1.0, level=1.0)
