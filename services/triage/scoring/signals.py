"""Deterministic STT, parser and optional teacher signal extractors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

from .settings import TriageScoringSettings

_LOGPROB_SEVERITY_SPAN = 0.5
_PARSE_SEVERITY_SPAN = 1.0
_COMPRESSION_SEVERITY_SPAN = 2.5


@dataclass(frozen=True)
class Observation:
    """A fired low-level signal with a configured weight and severity in [0, 1]."""

    name: str
    weight: float
    severity: float
    reason: str

    @property
    def contribution(self) -> float:
        """Return the weighted contribution to the raw failure score."""
        return self.weight * self.severity


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    """Return ``value`` as a string-keyed mapping when it is JSON-object-like."""
    return cast(Mapping[str, Any], value) if isinstance(value, Mapping) else None


def _number(value: Any) -> float | None:
    """Return a finite numeric value, or ``None`` for missing/non-numeric input."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _severity_below(value: float, ceiling: float, span: float) -> float:
    """Return 0 above ``ceiling`` and ramp linearly to 1 over ``span`` below it."""
    if value >= ceiling:
        return 0.0
    return min(1.0, (ceiling - value) / span)


def _severity_above(value: float, floor: float, span: float) -> float:
    """Return 0 below ``floor`` and ramp linearly to 1 over ``span`` above it."""
    if value <= floor:
        return 0.0
    return min(1.0, (value - floor) / span)


def extract_stt_signals(
    event: Mapping[str, Any], settings: TriageScoringSettings
) -> list[Observation]:
    """Extract word-confidence and decoder-stat signals from one turn event.

    Args:
        event: A schema-valid ``turn_event`` 1.0.0 dictionary.
        settings: Thresholds and weights for this run.

    Returns:
        Zero or more observations; missing optional fields never fire a signal.

    """
    observations: list[Observation] = []
    stt = _as_mapping(event.get("stt"))
    if stt is None:
        return observations

    text = stt.get("text", "")
    words = stt.get("words")
    confidences: list[float] = []
    if isinstance(words, list):
        for word in words:
            word_map = _as_mapping(word)
            conf = _number(word_map.get("conf")) if word_map is not None else None
            if conf is not None:
                confidences.append(conf)

    if confidences:
        min_conf = min(confidences)
        mean_conf = sum(confidences) / len(confidences)
        severity = max(
            _severity_below(min_conf, settings.min_word_conf, max(settings.min_word_conf, 1e-6)),
            _severity_below(mean_conf, settings.mean_word_conf, max(settings.mean_word_conf, 1e-6)),
        )
        if severity > 0.0:
            observations.append(
                Observation(
                    name="low_word_confidence",
                    weight=settings.weight_low_word_confidence,
                    severity=severity,
                    reason=(
                        f"min word confidence {min_conf:.2f} / mean {mean_conf:.2f} "
                        f"below thresholds {settings.min_word_conf:.2f} / "
                        f"{settings.mean_word_conf:.2f}"
                    ),
                )
            )
    elif isinstance(text, str) and not text.strip():
        # Empty hypothesis with no word confidences: no confidence evidence, but
        # the decoder produced nothing. Used by the exact-silence fixture turns.
        observations.append(
            Observation(
                name="low_word_confidence",
                weight=settings.weight_low_word_confidence,
                severity=1.0,
                reason="empty hypothesis with no word confidences",
            )
        )

    stats = _as_mapping(stt.get("decoder_stats"))
    if stats is None:
        return observations

    parts: list[str] = []
    severity = 0.0
    no_speech = _number(stats.get("no_speech_prob"))
    if no_speech is not None:
        candidate = _severity_above(
            no_speech,
            settings.no_speech_prob_ceiling,
            max(1.0 - settings.no_speech_prob_ceiling, 1e-6),
        )
        if candidate > 0.0:
            severity = max(severity, candidate)
            parts.append(f"no_speech_prob={no_speech:.2f}")

    avg_logprob = _number(stats.get("avg_logprob"))
    if avg_logprob is not None:
        candidate = _severity_below(avg_logprob, settings.avg_logprob_floor, _LOGPROB_SEVERITY_SPAN)
        if candidate > 0.0:
            severity = max(severity, candidate)
            parts.append(f"avg_logprob={avg_logprob:.2f}")

    compression = _number(stats.get("compression_ratio"))
    if compression is not None:
        candidate = _severity_above(
            compression,
            settings.compression_ratio_ceiling,
            _COMPRESSION_SEVERITY_SPAN,
        )
        if candidate > 0.0:
            severity = max(severity, candidate)
            parts.append(f"compression_ratio={compression:.2f}")

    if parts:
        observations.append(
            Observation(
                name="decoder_stats",
                weight=settings.weight_decoder_stats,
                severity=severity,
                reason="decoder signals over threshold: " + ", ".join(parts),
            )
        )
    return observations


def _expects_slots(output: Mapping[str, Any]) -> bool:
    """Return whether the parser actually emitted booking slots.

    An empty ``parser.output`` is treated as a no-slot parse even when
    ``schema_valid`` is false or ``expected_slot`` is ``"intent"``: the synthetic
    greeting turns deliberately carry those values while establishing no slots.
    A real dropped-all-slots failure is left to the final-record mismatch and
    repair-behaviour signals when those fields arrive.
    """
    return bool(output)


def extract_parser_signals(
    event: Mapping[str, Any], settings: TriageScoringSettings
) -> list[Observation]:
    """Extract schema-validity and parse-probability signals.

    Turns with no parser output (greetings, silence) are exempt: those fixtures
    intentionally carry ``schema_valid: false`` and a low logprob from a parser
    that was asked to emit nothing.

    Args:
        event: A schema-valid turn event.
        settings: Thresholds and weights for this run.

    Returns:
        Zero or more parser observations.

    """
    observations: list[Observation] = []
    parser = _as_mapping(event.get("parser"))
    if parser is None:
        return observations
    output = _as_mapping(parser.get("output")) or {}
    if not _expects_slots(output):
        return observations

    if parser.get("schema_valid") is False:
        observations.append(
            Observation(
                name="schema_invalid",
                weight=settings.weight_schema_invalid,
                severity=1.0,
                reason="parser reported schema_valid=false for a slot-bearing turn",
            )
        )

    logprob = _number(parser.get("logprob"))
    if logprob is not None and logprob < settings.parse_logprob_floor:
        severity = _severity_below(logprob, settings.parse_logprob_floor, _PARSE_SEVERITY_SPAN)
        observations.append(
            Observation(
                name="low_parse_probability",
                weight=settings.weight_low_parse_probability,
                severity=severity,
                reason=f"parser logprob {logprob:.2f} below {settings.parse_logprob_floor:.2f}",
            )
        )
    return observations


def extract_signals(event: Mapping[str, Any], settings: TriageScoringSettings) -> list[Observation]:
    """Extract all low-level signals for one turn event.

    Repair behaviour, mismatches and teacher disagreement are extracted by
    their own modules and combined in :mod:`services.triage.scoring.scoring`.
    """
    return extract_stt_signals(event, settings) + extract_parser_signals(event, settings)


@dataclass(frozen=True)
class TeacherVerdict:
    """Result of an offline teacher-model comparison for one turn."""

    disagrees: bool
    reason: str


@runtime_checkable
class TeacherModel(Protocol):
    """Injectable offline teacher adapter; never used in the call path."""

    def judge(self, event: Mapping[str, Any]) -> TeacherVerdict | None:
        """Compare the teacher transcript with the deployed hypothesis."""
        ...


class FakeTeacherModel:
    """Default teacher that always agrees; inject a real adapter for P4-04."""

    def judge(self, event: Mapping[str, Any]) -> TeacherVerdict:
        """Return an agreeing verdict without touching audio or network."""
        return TeacherVerdict(
            disagrees=False, reason="fake teacher: no offline transcript available"
        )


def teacher_observation(
    event: Mapping[str, Any],
    settings: TriageScoringSettings,
    teacher: TeacherModel,
) -> list[Observation]:
    """Return a teacher-disagreement observation when enabled and detected."""
    if not settings.teacher_enabled:
        return []
    verdict = teacher.judge(event)
    if verdict is None or not verdict.disagrees:
        return []
    return [
        Observation(
            name="teacher_disagreement",
            weight=settings.weight_teacher_disagreement,
            severity=1.0,
            reason=f"offline teacher disagrees: {verdict.reason}",
        )
    ]
