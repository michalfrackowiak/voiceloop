"""Pure metric primitives for the P1-07 evaluation harness.

No ML runtime, network or external metric library is required: WER uses a small
Levenshtein dynamic program, entity extraction uses deterministic configurable
regular expressions, and slot/E2E scores compare canonicalised JSON values.
All functions are pure so that per-turn results can be paired and resampled by
:mod:`services.evalgate.bootstrap`.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from jsonschema import ValidationError as JsonSchemaValidationError

from schemas.booking import BookingValidationError, validate_booking
from voiceloop_common.errors import VoiceLoopError


class EvaluationError(VoiceLoopError):
    """Raised when evaluation input or configuration is invalid."""


# --------------------------------------------------------------------------- #
# Word error rate
# --------------------------------------------------------------------------- #

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> tuple[str, ...]:
    r"""Case-fold text and return word tokens (``\w+``) for WER alignment."""
    return tuple(_WORD_RE.findall(text.casefold()))


@dataclass(frozen=True)
class AlignmentCounts:
    """Substitution, deletion and insertion counts from one token alignment."""

    substitutions: int
    deletions: int
    insertions: int


def levenshtein_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> AlignmentCounts:
    """Count edit operations with a Levenshtein DP and deterministic backtrace.

    Ties are resolved in the order substitution/match, deletion, insertion so
    that the split between error types is reproducible.

    Args:
        reference: Reference token sequence.
        hypothesis: Hypothesis token sequence.

    Returns:
        Counts of substitutions, deletions and insertions.

    """
    n, m = len(reference), len(hypothesis)
    distance = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        distance[i][0] = i
    for j in range(1, m + 1):
        distance[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            distance[i][j] = min(
                distance[i - 1][j - 1] + cost,
                distance[i - 1][j] + 1,
                distance[i][j - 1] + 1,
            )

    substitutions = deletions = insertions = 0
    i, j = n, m
    while i > 0 or j > 0:
        cost = 0 if i > 0 and j > 0 and reference[i - 1] == hypothesis[j - 1] else 1
        if i > 0 and j > 0 and distance[i][j] == distance[i - 1][j - 1] + cost:
            if cost:
                substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and distance[i][j] == distance[i - 1][j] + 1:
            deletions += 1
            i -= 1
        elif j > 0 and distance[i][j] == distance[i][j - 1] + 1:
            insertions += 1
            j -= 1
        else:  # pragma: no cover - unreachable for a consistent DP table
            raise EvaluationError("inconsistent Levenshtein table")
    return AlignmentCounts(substitutions, deletions, insertions)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Return ``(S + D + I) / max(reference_tokens, 1)``.

    Empty references are scored against one virtual word: a hallucinated
    four-token utterance on silence yields 4.0. This matches the report
    contract, which allows WER to exceed 1.
    """
    reference_tokens = tokenize(reference)
    counts = levenshtein_counts(reference_tokens, tokenize(hypothesis))
    return (counts.substitutions + counts.deletions + counts.insertions) / max(
        len(reference_tokens), 1
    )


# --------------------------------------------------------------------------- #
# Entity-level error (names, dates, numbers)
# --------------------------------------------------------------------------- #

_MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
_ORDINALS = (
    "first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|"
    "thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|"
    "twenty-first|twenty-second|twenty-third|thirtieth"
)
_CARDINALS = (
    "zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    "fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    "eighty|ninety|hundred"
)

DEFAULT_NAME_PATTERN = (
    r"\b(?:Mr|Mrs|Ms|Dr|mr|mrs|ms|dr)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"
    r"|\b[A-Z][a-z]+\s+[A-Z][a-z]+\b"
)
DEFAULT_DATE_PATTERN = (
    r"\b\d{4}-\d{2}-\d{2}\b"
    rf"|\b(?:\d{{1,2}}(?:st|nd|rd|th)|{_ORDINALS})(?:\s+of)?\s+(?:{_MONTHS})\b"
    rf"|\b(?:{_MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?\b"
    rf"|\b(?:{_MONTHS})\b"
)
DEFAULT_NUMBER_PATTERN = rf"\b\d+\b|\b(?:{_CARDINALS})\b"

ENTITY_TYPES: tuple[str, str, str] = ("names", "dates", "numbers")


@dataclass(frozen=True)
class EntityPatterns:
    """Deterministic regexes used to extract named entity types.

    ``names`` matches case-sensitively (capitalised names and titles);
    ``dates`` and ``numbers`` match case-insensitively. Overrides are checked
    against :data:`ENTITY_TYPES` and compiled once per extraction.
    """

    names: str = DEFAULT_NAME_PATTERN
    dates: str = DEFAULT_DATE_PATTERN
    numbers: str = DEFAULT_NUMBER_PATTERN

    def compiled(self) -> dict[str, re.Pattern[str]]:
        """Compile all patterns, raising :class:`EvaluationError` on bad regexes."""
        compiled: dict[str, re.Pattern[str]] = {}
        for entity_type in ENTITY_TYPES:
            source = getattr(self, entity_type)
            try:
                compiled[entity_type] = re.compile(source)
            except re.error as exc:
                raise EvaluationError(f"invalid {entity_type} entity pattern: {exc}") from exc
        return compiled

    def with_overrides(self, overrides: Mapping[str, str]) -> EntityPatterns:
        """Return a copy with selected patterns replaced by ``overrides``."""
        unknown = sorted(set(overrides) - set(ENTITY_TYPES))
        if unknown:
            raise EvaluationError(f"unknown entity pattern keys: {', '.join(unknown)}")
        values = {entity_type: getattr(self, entity_type) for entity_type in ENTITY_TYPES}
        values.update(overrides)
        return EntityPatterns(**values)


@dataclass(frozen=True)
class EntityResult:
    """Entity error for each type plus the occurrence-weighted aggregate."""

    by_type: dict[str, float]
    aggregate: float


def extract_entities(text: str, patterns: EntityPatterns | None = None) -> dict[str, Counter[str]]:
    """Return normalised multisets of named entities found in ``text``.

    Args:
        text: Transcript to scan.
        patterns: Optional regex overrides; defaults to :class:`EntityPatterns`.

    Returns:
        A multiset per entity type (``names``, ``dates``, ``numbers``).

    """
    patterns = patterns or EntityPatterns()
    extracted: dict[str, Counter[str]] = {}
    for entity_type, pattern in patterns.compiled().items():
        haystack = text if entity_type == "names" else text.casefold()
        counter: Counter[str] = Counter()
        for match in pattern.finditer(haystack):
            normalised = re.sub(r"[^\w\s]+", " ", match.group(0).casefold())
            normalised = " ".join(normalised.split())
            if normalised:
                counter[normalised] += 1
        extracted[entity_type] = counter
    return extracted


def entity_error(
    reference: str,
    hypothesis: str,
    patterns: EntityPatterns | None = None,
) -> EntityResult:
    """Compare multisets of names, dates and numbers in two strings.

    For each type, missing reference entities and extra hypothesis entities are
    both errors. The type error is ``(missing + extra) / max(ref, hyp, 1)``
    capped at 1; the aggregate is weighted by all reference entity occurrences.
    Empty references with no hypothesis entities score 0.
    """
    patterns = patterns or EntityPatterns()
    reference_entities = extract_entities(reference, patterns)
    hypothesis_entities = extract_entities(hypothesis, patterns)

    by_type: dict[str, float] = {}
    total_missing = total_extra = total_reference = 0
    for entity_type in ENTITY_TYPES:
        ref_counter = reference_entities[entity_type]
        hyp_counter = hypothesis_entities[entity_type]
        missing = sum((ref_counter - hyp_counter).values())
        extra = sum((hyp_counter - ref_counter).values())
        ref_count = sum(ref_counter.values())
        hyp_count = sum(hyp_counter.values())
        by_type[entity_type] = min(1.0, (missing + extra) / max(ref_count, hyp_count, 1))
        total_missing += missing
        total_extra += extra
        total_reference += ref_count
    aggregate = min(1.0, (total_missing + total_extra) / max(total_reference, 1))
    return EntityResult(by_type=by_type, aggregate=aggregate)


# --------------------------------------------------------------------------- #
# Parser slot F1 and schema validity
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SlotResult:
    """Exact-match precision, recall and F1 over canonical slot pairs."""

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def _canonical_values(slots: Mapping[str, Any]) -> dict[str, str]:
    """Serialise each slot value canonically so nested objects compare exactly."""
    try:
        return {
            str(key): json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            for key, value in slots.items()
        }
    except (TypeError, ValueError) as exc:
        raise EvaluationError("slot values must be JSON-serialisable") from exc


def slot_f1(reference_slots: Mapping[str, Any], predicted_slots: Mapping[str, Any]) -> SlotResult:
    """Compute exact-match slot F1 from two dicts of slots.

    Each ``(field, canonical value)`` pair is a true positive when present in
    both dicts, a false positive when only predicted, and a false negative when
    only referenced. Two empty dicts score F1 1.0.
    """
    reference_values = _canonical_values(reference_slots)
    predicted_values = _canonical_values(predicted_slots)
    tp = sum(1 for key, value in reference_values.items() if predicted_values.get(key) == value)
    fp = len(predicted_values) - tp
    fn = len(reference_values) - tp
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 1.0
    return SlotResult(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn)


def aggregate_slot_f1(tp: int, fp: int, fn: int) -> float:
    """Return micro-averaged F1 from summed slot counts across turns."""
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 1.0


def is_schema_valid(slots: Mapping[str, Any], *, no_slot_expected: bool = False) -> bool:
    """Validate parser slots with ``schemas.booking.validate_booking``.

    Args:
        slots: Parser output to check.
        no_slot_expected: When true, an empty output counts as valid because the
            labelled turn establishes no booking field (greetings, silence,
            noise). Otherwise ``{}`` fails the booking schema's required fields.

    Returns:
        True when the output is valid under the documented rule.

    """
    if no_slot_expected and not slots:
        return True
    try:
        validate_booking(dict(slots))
    except (BookingValidationError, JsonSchemaValidationError):
        return False
    return True


def schema_valid_rate(
    parser_outputs: Sequence[Mapping[str, Any]],
    expected_slots: Sequence[Mapping[str, Any]] | None = None,
) -> float:
    """Fraction of parser outputs that are valid booking objects.

    An empty sequence scores 0.0 (no evidence). When ``expected_slots`` is
    supplied, empty output on a no-booking turn is exempt from the required
    fields; see :func:`is_schema_valid`.
    """
    if not parser_outputs:
        return 0.0
    if expected_slots is not None and len(expected_slots) != len(parser_outputs):
        raise EvaluationError("expected_slots must align with parser_outputs")
    valid = 0
    for index, output in enumerate(parser_outputs):
        no_slot = expected_slots is not None and not expected_slots[index]
        if is_schema_valid(output, no_slot_expected=no_slot):
            valid += 1
    return valid / len(parser_outputs)


# --------------------------------------------------------------------------- #
# End-to-end booking-field accuracy from audio (primary promotion metric)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class E2EResult:
    """All-or-nothing turn result plus a per-field correctness breakdown."""

    turn_correct: bool
    field_accuracy: dict[str, bool]


def e2e_turn_result(
    reference_slots: Mapping[str, Any],
    predicted_slots: Mapping[str, Any],
) -> E2EResult:
    """Compare final expected slots against predicted slots for one turn.

    Rule: a field is correct only when it is present on both sides with an
    exactly equal canonical value; a turn is correct only when *every* field in
    the union of gold and predicted keys is correct. A no-booking turn is
    therefore correct only when both sides are empty.
    """
    reference_values = _canonical_values(reference_slots)
    predicted_values = _canonical_values(predicted_slots)
    field_accuracy = {
        key: key in reference_values
        and key in predicted_values
        and reference_values[key] == predicted_values[key]
        for key in set(reference_values) | set(predicted_values)
    }
    return E2EResult(turn_correct=all(field_accuracy.values()), field_accuracy=field_accuracy)


def e2e_booking_accuracy(results: Sequence[E2EResult]) -> float:
    """Return the fraction of turns whose complete booking slots match gold."""
    if not results:
        raise EvaluationError("cannot score e2e accuracy on zero turns")
    return fmean(1.0 if result.turn_correct else 0.0 for result in results)


def e2e_field_accuracies(results: Sequence[E2EResult]) -> dict[str, float]:
    """Return mean correctness per booking field over turns where it appears."""
    per_field: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for field, correct in result.field_accuracy.items():
            per_field[field].append(1.0 if correct else 0.0)
    return {field: fmean(values) for field, values in sorted(per_field.items())}


# --------------------------------------------------------------------------- #
# Per-turn container and aggregation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Prediction:
    """One fake artifact's STT transcript and parser slots for a fixture turn."""

    transcript: str
    slots: dict[str, Any]


@dataclass(frozen=True)
class TurnMetrics:
    """All per-turn values needed for aggregate metrics and paired bootstrap."""

    wer: float
    entity_error: float
    entity_error_by_type: Mapping[str, float]
    slot_tp: int
    slot_fp: int
    slot_fn: int
    slot_f1: float
    schema_valid: float
    e2e_correct: float
    e2e_field_accuracy: Mapping[str, bool]


def evaluate_turn(
    reference_transcript: str,
    reference_slots: Mapping[str, Any],
    prediction: Prediction,
    patterns: EntityPatterns | None = None,
) -> TurnMetrics:
    """Score one prediction against a labelled fixture turn."""
    entities = entity_error(reference_transcript, prediction.transcript, patterns)
    slots = slot_f1(reference_slots, prediction.slots)
    e2e = e2e_turn_result(reference_slots, prediction.slots)
    return TurnMetrics(
        wer=word_error_rate(reference_transcript, prediction.transcript),
        entity_error=entities.aggregate,
        entity_error_by_type=entities.by_type,
        slot_tp=slots.tp,
        slot_fp=slots.fp,
        slot_fn=slots.fn,
        slot_f1=slots.f1,
        schema_valid=1.0
        if is_schema_valid(prediction.slots, no_slot_expected=not reference_slots)
        else 0.0,
        e2e_correct=1.0 if e2e.turn_correct else 0.0,
        e2e_field_accuracy=e2e.field_accuracy,
    )


CORE_METRIC_NAMES: tuple[str, str, str, str, str] = (
    "wer",
    "entity_error",
    "slot_f1",
    "schema_valid_rate",
    "e2e_booking_accuracy",
)


def aggregate_metrics(turns: Sequence[TurnMetrics]) -> dict[str, float]:
    """Aggregate per-turn results into the five core metrics and diagnostics.

    WER and entity error are macro averages; slot F1 is micro-averaged from
    summed exact-match counts; the last two core metrics are macro averages of
    per-turn indicators. Extra numeric diagnostics carry per-entity-type error
    and per-field E2E accuracy, which the report schema permits.
    """
    if not turns:
        raise EvaluationError("cannot aggregate zero turns")
    result: dict[str, float] = {
        "wer": fmean(turn.wer for turn in turns),
        "entity_error": fmean(turn.entity_error for turn in turns),
        "slot_f1": aggregate_slot_f1(
            sum(turn.slot_tp for turn in turns),
            sum(turn.slot_fp for turn in turns),
            sum(turn.slot_fn for turn in turns),
        ),
        "schema_valid_rate": fmean(turn.schema_valid for turn in turns),
        "e2e_booking_accuracy": fmean(turn.e2e_correct for turn in turns),
    }
    for entity_type in ENTITY_TYPES:
        result[f"entity_error_{entity_type}"] = fmean(
            turn.entity_error_by_type.get(entity_type, 0.0) for turn in turns
        )
    per_field: dict[str, list[float]] = defaultdict(list)
    for turn in turns:
        for field, correct in turn.e2e_field_accuracy.items():
            per_field[field].append(1.0 if correct else 0.0)
    for field, values in per_field.items():
        result[f"e2e_field_{field}"] = fmean(values)
    return result
