"""Pure promotion rules for the P2-09 gate.

The only promotion metric is end-to-end booking-field accuracy from audio
(``context.md`` section 2 invariant 4). A better WER is never sufficient.

``confidence_intervals.*`` in an eval report bound **paired challenger minus
champion deltas** (lead note in ``docs/LEAD_BRIEFS.md``), not absolute metric
values. The lower bound of the e2e delta is compared with the configured
margin; no interval is ever read as an absolute metric.

Critical-slice checks use the e2e values in each report's ``slice_metrics``,
but only when both reports have ``turns > 0`` for that slice. A zero-turn slice
is unevaluated evidence and fails closed instead of silently passing. The
eval report 1.0.0 schema has no per-slice CIs, so slice deltas are point
comparisons; adding slice CIs needs a schema version bump by the lead.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from voiceloop_common import ValidationError, utcnow, validate
from voiceloop_common.errors import VoiceLoopError

from .settings import GateSettings

EVAL_REPORT_SCHEMA = "eval_report"
PRIMARY_METRIC = "e2e_booking_accuracy"
CRITICAL_SLICES: tuple[str, ...] = (
    "names",
    "dates",
    "numbers",
    "silence",
    "other_languages",
)
ACCEPTED_EVAL_SETS: frozenset[str] = frozenset({"general_capability", "regression"})


class GateError(VoiceLoopError):
    """Raised when the gate cannot evaluate an artifact or an alias state."""


@dataclass(frozen=True)
class GateDecision:
    """Immutable pass/fail outcome with human-readable reasons."""

    passed: bool
    reasons: tuple[str, ...]
    e2e_delta_lower: float | None = None
    required_margin: float | None = None
    slice_deltas: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary, e.g. for logs or audit context."""
        return {
            "outcome": "pass" if self.passed else "fail",
            "reasons": list(self.reasons),
            "e2e_delta_lower": self.e2e_delta_lower,
            "required_margin": self.required_margin,
            "slice_deltas": dict(self.slice_deltas),
        }

    def to_report_field(self) -> dict[str, Any]:
        """Return the schema-compatible ``gate_decision`` object for a report."""
        return {
            "outcome": "pass" if self.passed else "fail",
            "reason": "; ".join(self.reasons),
            "evaluated_at": utcnow().isoformat(timespec="seconds").replace("+00:00", "Z"),
        }


def decide_gate(
    champion_report: Mapping[str, Any],
    challenger_report: Mapping[str, Any],
    settings: GateSettings,
) -> GateDecision:
    """Compare two contract-valid reports using the P2-09 pass rule.

    Args:
        champion_report: Report for the current ``@champion`` artifact.
        challenger_report: Report for the converted ``@challenger`` artifact; its
            CIs must be paired ``challenger - champion`` deltas.
        settings: Typed margins and evidence requirements.

    Returns:
        A :class:`GateDecision`. Policy failures are returned, not raised.

    Raises:
        voiceloop_common.ValidationError: If either report violates the
            evaluation report contract or a required number is not finite.

    """
    validate(champion_report, EVAL_REPORT_SCHEMA)
    validate(challenger_report, EVAL_REPORT_SCHEMA)

    failures: list[str] = []
    notes: list[str] = []
    _check_eval_sets(champion_report, "champion_report", failures)
    _check_eval_sets(challenger_report, "challenger_report", failures)

    interval = challenger_report["confidence_intervals"][PRIMARY_METRIC]
    level = _finite_number(interval["level"], "e2e confidence interval level")
    lower = _finite_number(interval["lower"], "e2e delta lower bound")
    upper = _finite_number(interval["upper"], "e2e delta upper bound")

    if level < settings.min_ci_level:
        failures.append(
            f"e2e confidence level {level:.3f} is below required {settings.min_ci_level:.3f}"
        )
    if lower > upper:
        failures.append("e2e delta CI has lower bound above upper bound")
    if lower > settings.min_e2e_margin:
        notes.append(
            f"e2e delta lower bound {lower:.4f} exceeds required margin "
            f"{settings.min_e2e_margin:.4f}"
        )
    else:
        failures.append(
            f"e2e delta lower bound {lower:.4f} does not exceed required margin "
            f"{settings.min_e2e_margin:.4f}"
        )

    slice_deltas = _check_slices(champion_report, challenger_report, settings, failures)
    return GateDecision(
        passed=not failures,
        reasons=tuple(failures + notes),
        e2e_delta_lower=lower,
        required_margin=settings.min_e2e_margin,
        slice_deltas=slice_deltas,
    )


def _check_eval_sets(report: Mapping[str, Any], label: str, reasons: list[str]) -> None:
    """Reject unsealed golden labels and reports without accepted evidence sets."""
    sets = set(report["eval_sets"])
    if "golden" in sets:
        reasons.append(
            f"{label}: a 'golden' label is not sealed-set evidence; no set manifest exists yet"
        )
    if not sets & ACCEPTED_EVAL_SETS:
        accepted = ", ".join(sorted(ACCEPTED_EVAL_SETS))
        reasons.append(f"{label}: no accepted eval set ({accepted})")


def _check_slices(
    champion_report: Mapping[str, Any],
    challenger_report: Mapping[str, Any],
    settings: GateSettings,
    failures: list[str],
) -> dict[str, float]:
    """Collect e2e slice deltas, failing closed on missing or zero-turn evidence."""
    champion_slices = champion_report["slice_metrics"]
    challenger_slices = challenger_report["slice_metrics"]
    deltas: dict[str, float] = {}
    for name in CRITICAL_SLICES:
        champion_slice = champion_slices.get(name)
        challenger_slice = challenger_slices.get(name)
        if not isinstance(champion_slice, Mapping) or not isinstance(challenger_slice, Mapping):
            failures.append(f"{name}: missing slice metrics on champion or challenger")
            continue

        champion_turns = _finite_number(champion_slice.get("turns"), f"{name} champion turns")
        challenger_turns = _finite_number(challenger_slice.get("turns"), f"{name} challenger turns")
        if champion_turns <= 0 or challenger_turns <= 0:
            failures.append(f"{name}: zero turns on champion or challenger (no evidence)")
            continue

        champion_e2e = _finite_number(champion_slice.get(PRIMARY_METRIC), f"{name} champion e2e")
        challenger_e2e = _finite_number(
            challenger_slice.get(PRIMARY_METRIC), f"{name} challenger e2e"
        )
        delta = challenger_e2e - champion_e2e
        deltas[name] = delta
        if delta < -settings.slice_regression_tolerance:
            failures.append(
                f"{name}: e2e regression {delta:.4f} "
                f"(tolerance {settings.slice_regression_tolerance:.4f})"
            )
    return deltas


def _finite_number(value: Any, label: str) -> float:
    """Return a finite float, rejecting booleans and non-numeric contract data."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{label} must be a finite number")
    return number
