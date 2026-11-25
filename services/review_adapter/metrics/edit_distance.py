"""Deterministic Levenshtein distance from the STT hypothesis to the fake correction."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import mean, median

from .events import ReviewResultLike


def levenshtein(left: str, right: str) -> int:
    """Return the unit-cost Levenshtein distance between two strings.

    Insertion, deletion and substitution each cost one. Comparison is over
    Unicode code points without case folding or normalisation, so equal fixture
    strings always yield zero and no external library or model is involved.

    Args:
        left: Source string (the STT hypothesis).
        right: Target string (the scripted corrected transcript).

    Returns:
        The minimum number of single-character edits.

    """
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for row_index, source_char in enumerate(left, start=1):
        current = [row_index]
        for column_index, target_char in enumerate(right, start=1):
            substitution = previous[column_index - 1] + (source_char != target_char)
            current.append(
                min(
                    previous[column_index] + 1,
                    current[column_index - 1] + 1,
                    substitution,
                )
            )
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class ItemEditDistance:
    """Edit distance for one fake review, with lengths but no transcript text."""

    event_id: str
    call_id: str
    turn_index: int
    reviewer_id: str
    blind: bool
    distance: int
    hypothesis_length: int
    corrected_length: int


@dataclass(frozen=True)
class EditDistanceSummary:
    """Aggregate edit-distance statistics for fixture reviews."""

    count: int
    total_distance: int
    mean_distance: float | None
    median_distance: float | None
    min_distance: int | None
    max_distance: int | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable summary with no free text."""
        return {
            "count": self.count,
            "total_distance": self.total_distance,
            "mean_distance": self.mean_distance,
            "median_distance": self.median_distance,
            "min_distance": self.min_distance,
            "max_distance": self.max_distance,
        }


def compute_edit_distances(results: Iterable[ReviewResultLike]) -> tuple[ItemEditDistance, ...]:
    """Compute hypothesis-to-correction distances for every fake result.

    Args:
        results: Committed fixture review results in stable order.

    Returns:
        Per-item distances in input order.

    """
    return tuple(
        ItemEditDistance(
            event_id=result.event_id,
            call_id=result.call_id,
            turn_index=result.turn_index,
            reviewer_id=result.reviewer_id,
            blind=bool(result.blind),
            distance=levenshtein(result.hypothesis, result.corrected_transcript),
            hypothesis_length=len(result.hypothesis),
            corrected_length=len(result.corrected_transcript),
        )
        for result in results
    )


def summarize_edit_distances(items: Sequence[ItemEditDistance]) -> EditDistanceSummary:
    """Summarise per-item distances; summary fields are ``None`` when empty.

    Args:
        items: Per-item distances from :func:`compute_edit_distances`.

    Returns:
        Descriptive distance summary.

    """
    values = [item.distance for item in items]
    if not values:
        return EditDistanceSummary(0, 0, None, None, None, None)
    return EditDistanceSummary(
        count=len(values),
        total_distance=sum(values),
        mean_distance=mean(values),
        median_distance=median(values),
        min_distance=min(values),
        max_distance=max(values),
    )
