"""Per-item review time computed only from recorded fake timestamps."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import mean, median

from voiceloop_common import ValidationError

from .events import ReviewResultLike, parse_utc


@dataclass(frozen=True)
class ItemTiming:
    """Elapsed seconds for one scripted fake review, with pseudonymous lineage only."""

    event_id: str
    call_id: str
    turn_index: int
    reviewer_id: str
    blind: bool
    seconds: float


@dataclass(frozen=True)
class TimingSummary:
    """Aggregate statistics over per-item fake review times."""

    count: int
    total_seconds: float
    mean_seconds: float | None
    median_seconds: float | None
    min_seconds: float | None
    max_seconds: float | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable summary with no free text."""
        return {
            "count": self.count,
            "total_seconds": self.total_seconds,
            "mean_seconds": self.mean_seconds,
            "median_seconds": self.median_seconds,
            "min_seconds": self.min_seconds,
            "max_seconds": self.max_seconds,
        }


def elapsed_seconds(result: ReviewResultLike) -> float:
    """Compute elapsed seconds from a result's recorded fake timestamps.

    Args:
        result: P2-03-style fake review result with timezone-aware timestamps.

    Returns:
        Non-negative elapsed seconds.

    Raises:
        ValidationError: If completion precedes start.

    """
    started = parse_utc(result.started_at)
    completed = parse_utc(result.completed_at)
    seconds = (completed - started).total_seconds()
    if seconds < 0:
        raise ValidationError("review completed_at precedes started_at")
    return seconds


def compute_item_times(results: Iterable[ReviewResultLike]) -> tuple[ItemTiming, ...]:
    """Compute one elapsed-time record per fake review result.

    Args:
        results: Committed fixture review results in stable order.

    Returns:
        Per-item timings in input order.

    """
    return tuple(
        ItemTiming(
            event_id=result.event_id,
            call_id=result.call_id,
            turn_index=result.turn_index,
            reviewer_id=result.reviewer_id,
            blind=bool(result.blind),
            seconds=elapsed_seconds(result),
        )
        for result in results
    )


def summarize_times(items: Sequence[ItemTiming]) -> TimingSummary:
    """Summarise per-item times without any wall-clock or sampling assumptions.

    Args:
        items: Per-item timings from :func:`compute_item_times`.

    Returns:
        Descriptive summary; summary statistics are ``None`` for an empty input.

    """
    values = [item.seconds for item in items]
    if not values:
        return TimingSummary(0, 0.0, None, None, None, None)
    return TimingSummary(
        count=len(values),
        total_seconds=sum(values),
        mean_seconds=mean(values),
        median_seconds=median(values),
        min_seconds=min(values),
        max_seconds=max(values),
    )
