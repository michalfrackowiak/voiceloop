"""Descriptive drift detectors for triage scores and outcome rates (P4-02.1).

Detectors compare baseline and current replay windows supplied by injected
sources. They report counts and signed deltas only: no cause, no promotion
decision and no threshold change. Scores and statuses never leave as raw text.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from voiceloop_common import ValidationError

from .settings import (
    ALLOWED_OUTCOME_STATUSES,
    SYNTHETIC_LABEL,
    DriftSettings,
)

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_LONG_DIGITS = re.compile(r"\d{13,19}")
_PHONE = re.compile(r"(?:\+?\d{1,3}[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}")
_AUDIO_FILE = re.compile(r"(?i)\.(?:wav|mp3|flac|ogg|m4a)\b")


class OutcomeStatus(StrEnum):
    """Late-outcome statuses from the turn-event contract (context.md section 5)."""

    CONFIRMED = "confirmed"
    EDITED_BY_STAFF = "edited_by_staff"
    ABANDONED = "abandoned"
    TRANSFERRED = "transferred"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class TriageScoreWindow:
    """Baseline and current P2-01-style failure scores from replay windows."""

    baseline: tuple[float, ...]
    current: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class OutcomeWindow:
    """Baseline and current observed outcome statuses from replay windows."""

    baseline: tuple[str, ...]
    current: tuple[str, ...]


@runtime_checkable
class ScoreWindowSource(Protocol):
    """Seam supplying two score windows; a real source is injected by the caller."""

    def load(self) -> TriageScoreWindow:
        """Return the baseline and current score windows."""


@runtime_checkable
class OutcomeWindowSource(Protocol):
    """Seam supplying two outcome windows; a real source is injected by the caller."""

    def load(self) -> OutcomeWindow:
        """Return the baseline and current outcome windows."""


@dataclass(frozen=True, slots=True)
class InMemoryScoreWindowSource:
    """Deterministic fake score-window source."""

    window: TriageScoreWindow

    def load(self) -> TriageScoreWindow:
        """Return the injected score window."""
        return self.window


@dataclass(frozen=True, slots=True)
class InMemoryOutcomeWindowSource:
    """Deterministic fake outcome-window source."""

    window: OutcomeWindow

    def load(self) -> OutcomeWindow:
        """Return the injected outcome window."""
        return self.window


@dataclass(frozen=True, slots=True)
class DetectorSignal:
    """One detector's sample counts, descriptive deltas and alarm reasons."""

    name: str
    drifted: bool
    reasons: tuple[str, ...]
    baseline_count: int
    current_count: int
    deltas: Mapping[str, float | None]
    enough_data: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable, PII-free detector view."""
        payload: dict[str, Any] = {
            "name": self.name,
            "drifted": self.drifted,
            "reasons": list(self.reasons),
            "enough_data": self.enough_data,
            "baseline_count": self.baseline_count,
            "current_count": self.current_count,
            "deltas": {key: value for key, value in self.deltas.items()},
            "semantics": "descriptive_only_no_causal_claim",
        }
        assert_no_sensitive_text(payload)
        return payload


@dataclass(frozen=True, slots=True)
class DetectorReport:
    """Combined detector output for one replayed pair of windows."""

    signals: tuple[DetectorSignal, ...]
    drifted: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a labelled, JSON-serialisable report with no raw observations."""
        payload: dict[str, Any] = {
            "label": SYNTHETIC_LABEL,
            "drifted": self.drifted,
            "signals": [signal.to_dict() for signal in self.signals],
            "causal_claim": "none",
            "promotion_decision": "not_evaluated",
            "promotion_threshold_source": "P2-09 unchanged; detectors set no promotion threshold",
        }
        assert_no_sensitive_text(payload)
        return payload


def assert_no_sensitive_text(payload: Any) -> None:
    """Reject bytes and obvious PII patterns in a report or audit payload.

    Names are excluded structurally: sources supply only numbers and allowlisted
    status/reason codes, never free text from turn events.
    """
    if isinstance(payload, bytes):
        raise ValidationError("byte payloads are not allowed in reports")
    if isinstance(payload, str):
        for pattern in (_EMAIL, _LONG_DIGITS, _PHONE, _AUDIO_FILE):
            if pattern.search(payload):
                raise ValidationError("report contains a disallowed text pattern")
        return
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            assert_no_sensitive_text(key)
            assert_no_sensitive_text(value)
        return
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        for item in payload:
            assert_no_sensitive_text(item)


def _rounded(value: float | None) -> float | None:
    """Round a descriptive delta for deterministic serialisation."""
    return None if value is None else round(float(value), 6)


def _validate_scores(values: Sequence[float], field: str) -> tuple[float, ...]:
    """Validate a score window: finite floats in [0, 1]."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValidationError(f"{field} must be a sequence of scores")
    scores: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValidationError(f"{field}[{index}] must be a number")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValidationError(f"{field}[{index}] must be a finite score in [0, 1]")
        scores.append(number)
    return tuple(scores)


def _validate_statuses(values: Sequence[str], field: str) -> tuple[str, ...]:
    """Validate an outcome window against the allowed status list."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValidationError(f"{field} must be a sequence of outcome statuses")
    statuses: list[str] = []
    for index, value in enumerate(values):
        text = value.value if isinstance(value, OutcomeStatus) else str(value)
        if text not in ALLOWED_OUTCOME_STATUSES:
            raise ValidationError(f"{field}[{index}] has unknown outcome status")
        statuses.append(text)
    return tuple(statuses)


def _mean(values: Sequence[float]) -> float | None:
    """Return the arithmetic mean, or None for an empty window."""
    return sum(values) / len(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    """Return the median of an already validated score window."""
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _delta(current: float | None, baseline: float | None) -> float | None:
    """Return current minus baseline, or None when either side is undefined."""
    if current is None or baseline is None:
        return None
    return current - baseline


def _histogram(values: Sequence[float], buckets: int) -> tuple[float, ...]:
    """Return per-bucket proportions for scores in [0, 1]."""
    counts = [0] * buckets
    for value in values:
        index = min(buckets - 1, max(0, int(value * buckets)))
        counts[index] += 1
    total = len(values)
    return tuple(count / total for count in counts) if total else tuple(0.0 for _ in counts)


def detect_score_drift(window: TriageScoreWindow, settings: DriftSettings) -> DetectorSignal:
    """Compare triage score distributions with counts and descriptive deltas.

    The alarm is descriptive only: a shift in scores is not evidence that any
    component caused a failure, and the signal never promotes a model.
    """
    baseline = _validate_scores(window.baseline, "score baseline")
    current = _validate_scores(window.current, "score current")
    baseline_mean = _mean(baseline)
    current_mean = _mean(current)
    baseline_median = _median(baseline)
    current_median = _median(current)
    mean_delta = _delta(current_mean, baseline_mean)
    median_delta = _delta(current_median, baseline_median)

    baseline_histogram = _histogram(baseline, settings.score_buckets)
    current_histogram = _histogram(current, settings.score_buckets)
    bucket_deltas = tuple(
        current_histogram[index] - baseline_histogram[index]
        for index in range(settings.score_buckets)
    )
    tv_distance = sum(abs(value) for value in bucket_deltas) / 2.0
    max_bucket_delta = max(bucket_deltas) if bucket_deltas else None

    enough_data = (
        len(baseline) >= settings.min_window_samples and len(current) >= settings.min_window_samples
    )
    reasons: list[str] = []
    if not enough_data:
        reasons.append("insufficient_window_samples")
    else:
        assert mean_delta is not None
        if abs(mean_delta) >= settings.score_mean_delta_threshold:
            reasons.append("score_mean_delta")
        if tv_distance >= settings.score_tv_distance_threshold:
            reasons.append("score_distribution_shift")
    alarm_reasons = tuple(reason for reason in reasons if reason != "insufficient_window_samples")

    deltas: dict[str, float | None] = {
        "baseline_mean": _rounded(baseline_mean),
        "current_mean": _rounded(current_mean),
        "mean_delta": _rounded(mean_delta),
        "median_delta": _rounded(median_delta),
        "total_variation_distance": _rounded(tv_distance),
        "max_bucket_delta": _rounded(max_bucket_delta),
    }
    for index, value in enumerate(bucket_deltas):
        deltas[f"bucket_{index:02d}_delta"] = _rounded(value)

    return DetectorSignal(
        name="triage_score_distribution",
        drifted=bool(alarm_reasons),
        reasons=tuple(reasons),
        baseline_count=len(baseline),
        current_count=len(current),
        deltas=deltas,
        enough_data=enough_data,
    )


def detect_outcome_rate_drift(window: OutcomeWindow, settings: DriftSettings) -> DetectorSignal:
    """Compare observed outcome rates with counts and descriptive deltas.

    Outcome statuses are observations, not human labels or promotion evidence.
    """
    baseline = _validate_statuses(window.baseline, "outcome baseline")
    current = _validate_statuses(window.current, "outcome current")
    statuses = tuple(status.value for status in OutcomeStatus)
    baseline_counts = Counter(baseline)
    current_counts = Counter(current)

    deltas: dict[str, float | None] = {}
    for status in statuses:
        baseline_rate = baseline_counts[status] / len(baseline) if baseline else None
        current_rate = current_counts[status] / len(current) if current else None
        deltas[f"rate_{status}_baseline"] = _rounded(baseline_rate)
        deltas[f"rate_{status}_current"] = _rounded(current_rate)
        deltas[f"rate_{status}_delta"] = _rounded(_delta(current_rate, baseline_rate))

    degraded = set(settings.degraded_outcomes)
    baseline_degraded = sum(baseline_counts[status] for status in degraded)
    current_degraded = sum(current_counts[status] for status in degraded)
    baseline_degraded_rate = baseline_degraded / len(baseline) if baseline else None
    current_degraded_rate = current_degraded / len(current) if current else None
    degraded_delta = _delta(current_degraded_rate, baseline_degraded_rate)
    status_deltas = [
        abs(value)
        for status in statuses
        if (
            value := _delta(
                current_counts[status] / len(current) if current else None,
                baseline_counts[status] / len(baseline) if baseline else None,
            )
        )
        is not None
    ]
    max_abs_rate_delta = max(status_deltas) if status_deltas else None

    enough_data = (
        len(baseline) >= settings.min_window_samples and len(current) >= settings.min_window_samples
    )
    reasons: list[str] = []
    if not enough_data:
        reasons.append("insufficient_window_samples")
    else:
        assert max_abs_rate_delta is not None
        assert degraded_delta is not None
        if max_abs_rate_delta >= settings.outcome_rate_delta_threshold:
            reasons.append("outcome_status_rate_shift")
        if abs(degraded_delta) >= settings.outcome_rate_delta_threshold:
            reasons.append("degraded_outcome_rate_shift")
    alarm_reasons = tuple(reason for reason in reasons if reason != "insufficient_window_samples")

    deltas.update(
        {
            "degraded_rate_baseline": _rounded(baseline_degraded_rate),
            "degraded_rate_current": _rounded(current_degraded_rate),
            "degraded_rate_delta": _rounded(degraded_delta),
            "max_abs_rate_delta": _rounded(max_abs_rate_delta),
        }
    )
    return DetectorSignal(
        name="outcome_rates",
        drifted=bool(alarm_reasons),
        reasons=tuple(reasons),
        baseline_count=len(baseline),
        current_count=len(current),
        deltas=deltas,
        enough_data=enough_data,
    )


def detect_all(
    score_window: TriageScoreWindow,
    outcome_window: OutcomeWindow,
    settings: DriftSettings | None = None,
) -> DetectorReport:
    """Run both detectors and return one labelled, descriptive report."""
    config = settings or DriftSettings()
    signals = (
        detect_score_drift(score_window, config),
        detect_outcome_rate_drift(outcome_window, config),
    )
    return DetectorReport(signals=signals, drifted=any(signal.drifted for signal in signals))
