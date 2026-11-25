"""PII span types, fail-closed validation and read-only P1-09 detection seams.

Character spans come from the committed P1-09 detector (or from caller-supplied
known names, e.g. reviewed ``guest_name`` slots). Timestamp spans are mapped
from word timings in the synthetic turn-event contract and are always validated
before any audio is modified. No matched text is stored in any span type.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from services.datasets.filters import PiiCategory, PiiFinding, detect_pii
from voiceloop_common import VoiceLoopError


# Category values mirror ``PiiCategory`` so findings convert without new logic.
class SpanCategory(StrEnum):
    """Category recorded on a span; never the matched value."""

    CARD_NUMBER = "card_number"
    EMAIL = "email"
    PHONE = "phone"
    PERSON_NAME = "person_name"
    PII = "pii"
    MIXED = "mixed"


@dataclass(frozen=True, slots=True)
class CharSpan:
    """Half-open character offsets plus a category; contains no text."""

    start: int
    end: int
    category: SpanCategory


@dataclass(frozen=True, slots=True)
class WordTiming:
    """Synthetic word timing with character offsets; contains no word text."""

    index: int
    start_ms: int
    end_ms: int
    start_char: int
    end_char: int


@dataclass(frozen=True, slots=True)
class TimestampSpan:
    """Half-open millisecond span to be zeroed in fake audio."""

    start_ms: int
    end_ms: int
    category: SpanCategory

    def to_dict(self) -> dict[str, str | int]:
        """Return offsets and category only, safe for reports and audits."""
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "category": str(self.category),
        }


class RedactionError(VoiceLoopError):
    """Structured, PII-free redaction failure with a stable machine code."""

    def __init__(
        self,
        code: str,
        message: str,
        details: Mapping[str, str | int] | None = None,
    ) -> None:
        """Store a stable code, human message and scalar-only details."""
        super().__init__(message)
        self.code = code
        self.details: dict[str, str | int] = dict(details or {})

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe error shape; details never carry matched PII."""
        return {
            "code": self.code,
            "message": str(self),
            "details": {key: self.details[key] for key in sorted(self.details)},
        }


class InvalidSpanError(RedactionError):
    """A supplied span or audio duration failed validation."""

    def __init__(
        self,
        code: str,
        message: str,
        details: Mapping[str, str | int] | None = None,
    ) -> None:
        """Store the structured failure with its specific code."""
        super().__init__(code, message, details)


class UnmappedSpanError(RedactionError):
    """A detected character span could not be mapped to word timings."""

    def __init__(self, message: str, details: Mapping[str, str | int] | None = None) -> None:
        """Store the structured failure with code ``unmapped_pii_span``."""
        super().__init__("unmapped_pii_span", message, details)


class UnverifiedRedactionError(RedactionError):
    """Post-redaction verification found a non-zero span or residual PII."""

    def __init__(
        self,
        code: str,
        message: str,
        details: Mapping[str, str | int] | None = None,
    ) -> None:
        """Store the structured failure with its specific code."""
        super().__init__(code, message, details)


class CardNumberRejected(RedactionError):
    """A card number was found; it must be hard-dropped, never pseudonymised."""

    def __init__(self, message: str, details: Mapping[str, str | int] | None = None) -> None:
        """Store the structured failure with code ``card_number_pii``."""
        super().__init__("card_number_pii", message, details)


class TranscriptPiiDetector(Protocol):
    """Read-only transcript PII detection seam; default delegates to P1-09."""

    def detect(self, text: str) -> tuple[PiiFinding, ...]:
        """Return findings with categories and offsets, no matched values."""
        ...


class P109TranscriptDetector:
    """Thin adapter over the committed, read-only P1-09 ``detect_pii`` API."""

    def detect(self, text: str) -> tuple[PiiFinding, ...]:
        """Detect transcript PII using P1-09; never modify the input text."""
        return detect_pii(text)


def validate_span(span: TimestampSpan, duration_ms: int) -> None:
    """Fail closed on negative, reversed or out-of-bounds timestamp spans.

    Args:
        span: Supplied span to validate before any padding or zeroing.
        duration_ms: Explicit fake-audio duration in milliseconds.

    Raises:
        InvalidSpanError: The duration or span is malformed. Details contain
            only offsets and duration, never audio or transcript values.

    """
    if duration_ms < 0:
        raise InvalidSpanError(
            "invalid_duration",
            "audio duration must not be negative",
            {"duration_ms": duration_ms},
        )
    if span.start_ms < 0 or span.end_ms < 0:
        raise InvalidSpanError(
            "negative_span",
            "PII audio spans must not have negative offsets",
            {"start_ms": span.start_ms, "end_ms": span.end_ms},
        )
    if span.start_ms >= span.end_ms:
        raise InvalidSpanError(
            "reversed_span",
            "PII audio span must have end_ms greater than start_ms",
            {"start_ms": span.start_ms, "end_ms": span.end_ms},
        )
    if span.end_ms > duration_ms:
        raise InvalidSpanError(
            "span_beyond_duration",
            "PII audio span extends beyond the audio duration",
            {"start_ms": span.start_ms, "end_ms": span.end_ms, "duration_ms": duration_ms},
        )


def merge_spans(
    spans: Sequence[TimestampSpan],
    *,
    padding_ms: int,
    duration_ms: int,
) -> tuple[TimestampSpan, ...]:
    """Validate, pad and deterministically merge overlapping/adjacent spans.

    Padding is clamped to the waveform bounds *after* strict validation, so it
    can never hide an invalid supplied span. Adjacent spans (``next.start <=
    current.end``) are merged to guarantee conservative, gap-free zeroing.

    Args:
        spans: Supplied timestamp spans, validated individually.
        padding_ms: Non-negative conservative padding on both sides.
        duration_ms: Explicit fake-audio duration in milliseconds.

    Returns:
        Sorted, non-overlapping, non-adjacent spans covering every supplied span.

    Raises:
        InvalidSpanError: Any span, padding value or duration is malformed.

    """
    if duration_ms < 0 or padding_ms < 0:
        raise InvalidSpanError(
            "invalid_duration",
            "duration and padding must not be negative",
            {"duration_ms": duration_ms, "padding_ms": padding_ms},
        )
    padded: list[TimestampSpan] = []
    for span in spans:
        validate_span(span, duration_ms)
        padded.append(
            TimestampSpan(
                start_ms=max(0, span.start_ms - padding_ms),
                end_ms=min(duration_ms, span.end_ms + padding_ms),
                category=span.category,
            )
        )
    ordered = sorted(padded, key=lambda item: (item.start_ms, item.end_ms, str(item.category)))
    merged: list[TimestampSpan] = []
    for span in ordered:
        if not merged or span.start_ms > merged[-1].end_ms:
            merged.append(span)
            continue
        previous = merged[-1]
        category = previous.category if previous.category == span.category else SpanCategory.MIXED
        merged[-1] = TimestampSpan(
            start_ms=previous.start_ms,
            end_ms=max(previous.end_ms, span.end_ms),
            category=category,
        )
    return tuple(merged)


def find_known_name_spans(text: str, names: Sequence[str]) -> tuple[CharSpan, ...]:
    """Locate caller-supplied known names without retaining their text.

    Longer names win over shorter ones, then earlier offsets, so ``"Ada
    Fixture"`` takes precedence over ``"Ada"`` deterministically. Overlaps are
    resolved before any replacement; only offsets and the person-name category
    are returned.

    Args:
        text: Transcript to scan read-only.
        names: Known names (e.g. reviewed ``guest_name`` slot values).

    Returns:
        Non-overlapping name spans ordered by start offset.

    """
    candidates: list[CharSpan] = []
    for raw_name in names:
        name = raw_name.strip()
        if not name:
            continue
        pattern = re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)", re.IGNORECASE)
        for match in pattern.finditer(text):
            candidates.append(CharSpan(match.start(), match.end(), SpanCategory.PERSON_NAME))
    accepted: list[CharSpan] = []
    for span in sorted(
        candidates,
        key=lambda item: (-(item.end - item.start), item.start, str(item.category)),
    ):
        if any(span.start < kept.end and kept.start < span.end for kept in accepted):
            continue
        accepted.append(span)
    return tuple(sorted(accepted, key=lambda item: (item.start, item.end)))


# P1-09's ``name is <token>`` branch can report only a bare title when the
# surname follows a dot, e.g. ``"The name is Mr. Smith."`` yields ``"Mr"``.
# P3-10 closes that gap on its own side without changing the read-only detector.
_TITLE_WORDS = frozenset({"mr", "mrs", "ms", "miss", "dr"})
_CUED_SURNAME_RE = re.compile(r"[.\u2010-\u2015]?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)")


def _is_bare_title(text: str, start: int, end: int) -> bool:
    """Return whether a finding covers exactly a known honorific."""
    return text[start:end].casefold().rstrip(".") in _TITLE_WORDS


def _full_cued_name_end(text: str, start: int, end: int) -> int | None:
    """Return the end of a bare title plus its capitalized continuation.

    Args:
        text: Exact transcript text being screened read-only.
        start: Inclusive start offset of the bare-title finding.
        end: Exclusive end offset of the bare-title finding.

    Returns:
        Exclusive offset covering the title and following capitalized token(s),
        or ``None`` when no safe continuation exists.

    """
    if not _is_bare_title(text, start, end):
        return None
    suffix = _CUED_SURNAME_RE.match(text, end)
    if suffix is None or suffix.end(1) <= end:
        return None
    return suffix.end(1)


def expand_cued_name_spans(
    findings: Sequence[PiiFinding],
    text: str,
    *,
    strict: bool = True,
) -> tuple[PiiFinding, ...]:
    """Expand bare-title P1-09 name findings to cover the full cued name.

    This is a P3-10-side safety net used by both :func:`resolve_char_spans`
    (text) and :meth:`P109WordSpanDetector.detect_spans` (audio). It only
    changes ``PERSON_NAME`` findings whose matched text is exactly a title;
    other categories and complete names pass through untouched.

    Args:
        findings: Read-only findings from P1-09 or an injected detector.
        text: Exact transcript text the offsets refer to.
        strict: When ``True``, a bare title with no capitalized continuation
            fails closed instead of leaving a possible surname unscreened.

    Returns:
        Findings with bare-title spans extended to the full cued name.

    Raises:
        UnverifiedRedactionError: ``strict=True`` and a bare title has no
            capitalized continuation; reviewed ``known_names`` are required.

    """
    expanded: list[PiiFinding] = []
    for finding in findings:
        if not _is_bare_title(text, finding.start, finding.end):
            expanded.append(finding)
            continue
        if finding.category is not PiiCategory.PERSON_NAME:
            expanded.append(finding)
            continue
        full_end = _full_cued_name_end(text, finding.start, finding.end)
        if full_end is None:
            if strict:
                raise UnverifiedRedactionError(
                    "unverified_cued_name",
                    "bare-title name has no capitalized continuation; supply reviewed known_names",
                    {"start": finding.start, "end": finding.end},
                )
            expanded.append(finding)
            continue
        expanded.append(PiiFinding(PiiCategory.PERSON_NAME, finding.start, full_end))
    return tuple(expanded)


def resolve_char_spans(
    findings: Sequence[PiiFinding],
    text: str,
    known_names: Sequence[str] = (),
) -> tuple[CharSpan, ...]:
    """Combine P1-09 findings and known-name spans into disjoint replacements.

    Bare-title findings are expanded to the full cued name first, so a title
    followed by a capitalized surname never leaves that surname behind when
    ``known_names`` is omitted. Longer spans still win; detector findings beat
    known-name spans of equal length; the final order is by offset.

    Args:
        findings: Findings from the read-only P1-09 detector.
        text: Exact transcript text used to locate caller-supplied names.
        known_names: Optional reviewed name values to cover undetected names.

    Returns:
        Disjoint character spans, longest-first resolution, offset-sorted output.

    """
    candidates: list[tuple[CharSpan, int]] = []
    for finding in expand_cued_name_spans(findings, text):
        candidates.append(
            (
                CharSpan(finding.start, finding.end, SpanCategory(str(finding.category))),
                0,
            )
        )
    candidates.extend((span, 1) for span in find_known_name_spans(text, known_names))
    return tuple(span for span, _priority in _resolve_candidates(candidates))


def _resolve_candidates(
    candidates: Sequence[tuple[CharSpan, int]],
) -> tuple[tuple[CharSpan, int], ...]:
    """Resolve candidate spans by length, offset and source priority."""
    accepted: list[tuple[CharSpan, int]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            -(item[0].end - item[0].start),
            item[0].start,
            item[1],
            str(item[0].category),
        ),
    ):
        span = candidate[0]
        if any(span.start < kept[0].end and kept[0].start < span.end for kept in accepted):
            continue
        accepted.append(candidate)
    return tuple(sorted(accepted, key=lambda item: (item[0].start, item[0].end)))


def word_timings_from_text(text: str, duration_ms: int) -> tuple[WordTiming, ...]:
    """Build deterministic synthetic word timings for a transcript.

    Words are located by whitespace in the exact supplied text; millisecond
    boundaries divide the explicit duration evenly. The result carries offsets
    only, never word text, so names never enter an intermediate structure.

    Args:
        text: Transcript text with the same whitespace as P1-09 screens.
        duration_ms: Explicit fake-audio duration in milliseconds.

    Returns:
        Word timings in transcript order.

    Raises:
        InvalidSpanError: Non-empty text was given a non-positive duration.

    """
    matches = tuple(re.finditer(r"\S+", text))
    if not matches:
        return ()
    if duration_ms <= 0:
        raise InvalidSpanError(
            "invalid_duration",
            "non-empty transcript requires positive audio duration",
            {"duration_ms": duration_ms, "word_count": len(matches)},
        )
    count = len(matches)
    timings: list[WordTiming] = []
    for index, match in enumerate(matches):
        start_ms = (index * duration_ms) // count
        end_ms = ((index + 1) * duration_ms) // count
        if end_ms <= start_ms:
            end_ms = min(duration_ms, start_ms + 1)
        if end_ms <= start_ms:
            start_ms = max(0, duration_ms - 1)
            end_ms = duration_ms
        timings.append(
            WordTiming(
                index=index,
                start_ms=start_ms,
                end_ms=end_ms,
                start_char=match.start(),
                end_char=match.end(),
            )
        )
    return tuple(timings)


class PiiSpanDetector(Protocol):
    """Injectable mapping from transcript PII to timestamped fake-audio spans."""

    def detect_spans(
        self,
        text: str,
        words: Sequence[WordTiming],
        *,
        known_names: Sequence[str] = (),
    ) -> tuple[TimestampSpan, ...]:
        """Return timestamp spans for PII found in ``text``."""
        ...


class P109WordSpanDetector:
    """Default detector: P1-09 findings plus known names, mapped to word times.

    A finding that cannot be mapped to any word timing is an error, not a
    skipped span. This keeps audio redaction fail-closed.
    """

    def __init__(self, detector: TranscriptPiiDetector | None = None) -> None:
        """Use the injected transcript detector, or P1-09 by default."""
        self._detector: TranscriptPiiDetector = detector or P109TranscriptDetector()

    def detect_spans(
        self,
        text: str,
        words: Sequence[WordTiming],
        *,
        known_names: Sequence[str] = (),
    ) -> tuple[TimestampSpan, ...]:
        """Map every detected character span to the covering word timing(s).

        Args:
            text: Transcript text screened read-only with P1-09.
            words: Synthetic word timings for the same text.
            known_names: Reviewed name values to include beyond P1-09 cues.

        Returns:
            One timestamp span per disjoint character span; merging is deferred
            to :func:`merge_spans` so padding is applied exactly once.

        Raises:
            UnmappedSpanError: A character span has no covering word timing.
            UnverifiedRedactionError: A bare title has no capitalized
                continuation and no reviewed name can be inferred.

        """
        detected = self._detector.detect(text)
        # Shared expansion also runs inside resolve_char_spans; reapplying it
        # here is idempotent because an expanded "Mr. Smith" is not a title.
        expanded = expand_cued_name_spans(detected, text)
        char_spans = resolve_char_spans(expanded, text, known_names)
        mapped: list[TimestampSpan] = []
        for span in char_spans:
            covering = [
                word for word in words if word.start_char < span.end and span.start < word.end_char
            ]
            if not covering:
                raise UnmappedSpanError(
                    "detected PII span has no synthetic word timing",
                    {"start": span.start, "end": span.end},
                )
            mapped.append(
                TimestampSpan(
                    start_ms=min(word.start_ms for word in covering),
                    end_ms=max(word.end_ms for word in covering),
                    category=span.category,
                )
            )
        return tuple(mapped)


__all__ = [
    "CardNumberRejected",
    "CharSpan",
    "InvalidSpanError",
    "P109TranscriptDetector",
    "P109WordSpanDetector",
    "PiiSpanDetector",
    "RedactionError",
    "SpanCategory",
    "TimestampSpan",
    "TranscriptPiiDetector",
    "UnmappedSpanError",
    "UnverifiedRedactionError",
    "WordTiming",
    "expand_cued_name_spans",
    "find_known_name_spans",
    "merge_spans",
    "resolve_char_spans",
    "validate_span",
    "word_timings_from_text",
]
