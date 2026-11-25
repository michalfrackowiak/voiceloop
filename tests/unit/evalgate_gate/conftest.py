"""Shared factories for P2-09 gate unit tests.

Reports are built through the P1-07 report writer so every test input is a
contract-valid ``eval_report`` 1.0.0 payload. Confidence intervals are always
constructed as paired challenger − champion deltas.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from services.evalgate import ConfidenceInterval, build_report
from services.evalgate.gate import GateSettings

CRITICAL_SLICES: tuple[str, ...] = (
    "names",
    "dates",
    "numbers",
    "silence",
    "other_languages",
)
DEFAULT_SETS: tuple[str, ...] = ("general_capability", "regression")


def make_report(
    *,
    artifact_uri: str,
    e2e_accuracy: float,
    e2e_delta_lower: float = 0.0,
    wer: float = 0.2,
    wer_delta_lower: float = 0.0,
    ci_level: float = 0.95,
    eval_sets: Sequence[str] = DEFAULT_SETS,
    slice_values: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Build a valid report with explicit delta CIs and critical-slice turns.

    ``slice_values`` maps a slice name to ``(turns, e2e_booking_accuracy)``;
    unspecified slices default to 10 turns at ``e2e_accuracy``.
    """
    supplied = dict(slice_values or {})

    def interval(lower: float) -> ConfidenceInterval:
        """Build a finite CI whose upper bound is above its lower bound."""
        return ConfidenceInterval(lower=lower, upper=lower + 0.02, level=ci_level)

    slices: dict[str, dict[str, float]] = {}
    for name in CRITICAL_SLICES:
        turns, slice_e2e = supplied.get(name, (10.0, e2e_accuracy))
        slices[name] = {
            "turns": float(turns),
            "wer": float(wer),
            "e2e_booking_accuracy": float(slice_e2e),
        }

    return build_report(
        artifact_uri=artifact_uri,
        eval_sets=eval_sets,
        metrics={
            "wer": float(wer),
            "entity_error": 0.1,
            "slot_f1": 0.8,
            "schema_valid_rate": 0.9,
            "e2e_booking_accuracy": float(e2e_accuracy),
        },
        confidence_intervals={
            "wer": interval(wer_delta_lower),
            "entity_error": interval(-0.01),
            "slot_f1": interval(-0.01),
            "schema_valid_rate": interval(-0.01),
            "e2e_booking_accuracy": interval(e2e_delta_lower),
        },
        slice_metrics=slices,
        decoding_settings={
            "hotwords": [],
            "initial_prompt": "",
            "condition_on_previous_text": False,
            "language": "en",
        },
    )


@pytest.fixture
def gate_settings() -> GateSettings:
    """Return deterministic gate settings with a small positive margin."""
    return GateSettings(min_e2e_margin=0.01)


@pytest.fixture
def champion_report() -> dict[str, Any]:
    """Return a champion report with full coverage and e2e accuracy 0.70."""
    return make_report(artifact_uri="converted://champion-v3", e2e_accuracy=0.70)


@pytest.fixture
def better_report() -> dict[str, Any]:
    """Return a challenger report whose paired e2e delta clears the margin."""
    return make_report(
        artifact_uri="converted://challenger-v4",
        e2e_accuracy=0.80,
        e2e_delta_lower=0.05,
    )


@pytest.fixture
def worse_report() -> dict[str, Any]:
    """Return a challenger report with a negative paired e2e delta."""
    return make_report(
        artifact_uri="converted://challenger-v5",
        e2e_accuracy=0.65,
        e2e_delta_lower=-0.04,
    )
