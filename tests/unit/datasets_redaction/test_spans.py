"""Span validation, deterministic merging and read-only P1-09 mapping tests."""

from __future__ import annotations

import re

import pytest

from services.datasets.filters import PiiCategory, PiiFinding, detect_pii
from services.datasets.redaction.spans import (
    InvalidSpanError,
    P109TranscriptDetector,
    P109WordSpanDetector,
    SpanCategory,
    TimestampSpan,
    UnmappedSpanError,
    UnverifiedRedactionError,
    expand_cued_name_spans,
    find_known_name_spans,
    merge_spans,
    resolve_char_spans,
    validate_span,
    word_timings_from_text,
)


def _span(start: int, end: int, category: SpanCategory = SpanCategory.PERSON_NAME) -> TimestampSpan:
    """Build a timestamp span for tests."""
    return TimestampSpan(start, end, category)


def test_validate_span_rejects_negative_reversed_and_out_of_bounds() -> None:
    cases = [
        (-1, 10, "negative_span"),
        (5, 1, "reversed_span"),
        (0, 101, "span_beyond_duration"),
        (20, 20, "reversed_span"),
    ]
    for start, end, expected_code in cases:
        with pytest.raises(InvalidSpanError) as exc:
            validate_span(_span(start, end), 100)
        assert exc.value.code == expected_code
        assert "start_ms" in exc.value.details or "duration_ms" in exc.value.details


def test_validate_span_rejects_negative_duration() -> None:
    with pytest.raises(InvalidSpanError) as exc:
        validate_span(_span(0, 1), -1)
    assert exc.value.code == "invalid_duration"


def test_merge_spans_pads_overlaps_and_adjacent_spans() -> None:
    merged = merge_spans(
        [_span(10, 30), _span(30, 50), _span(65, 75)],
        padding_ms=5,
        duration_ms=100,
    )
    assert merged == (
        _span(5, 55, SpanCategory.PERSON_NAME),
        _span(60, 80, SpanCategory.PERSON_NAME),
    )


def test_merge_spans_is_order_independent_and_marks_mixed_categories() -> None:
    first = merge_spans(
        [_span(10, 20, SpanCategory.EMAIL), _span(25, 35, SpanCategory.PHONE)],
        padding_ms=5,
        duration_ms=100,
    )
    second = merge_spans(
        [_span(25, 35, SpanCategory.PHONE), _span(10, 20, SpanCategory.EMAIL)],
        padding_ms=5,
        duration_ms=100,
    )
    assert first == second == (_span(5, 40, SpanCategory.MIXED),)


def test_merge_spans_clamps_padding_only_after_validation() -> None:
    merged = merge_spans([_span(0, 5)], padding_ms=50, duration_ms=10)
    assert merged == (_span(0, 10),)
    with pytest.raises(InvalidSpanError):
        merge_spans([_span(0, 11)], padding_ms=50, duration_ms=10)


def test_find_known_name_spans_prefers_longest_names_without_text() -> None:
    text = "Ada and Ada Fixture stayed."
    spans = find_known_name_spans(text, ["Ada", "Ada Fixture"])
    assert [(span.start, span.end) for span in spans] == [(0, 3), (8, 19)]
    assert all(span.category is SpanCategory.PERSON_NAME for span in spans)
    assert all(not hasattr(span, "name") for span in spans)


def test_word_timings_cover_exact_whitespace_offsets() -> None:
    text = "The name is Mr. Smith."
    timings = word_timings_from_text(text, 100)
    words = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
    assert [(item.start_char, item.end_char) for item in timings] == words
    assert timings[0].start_ms == 0
    assert timings[-1].end_ms == 100
    assert all(item.end_ms > item.start_ms for item in timings)


def test_word_timings_reject_nonpositive_duration_for_text() -> None:
    with pytest.raises(InvalidSpanError) as exc:
        word_timings_from_text("hello", 0)
    assert exc.value.code == "invalid_duration"
    assert word_timings_from_text("", 0) == ()


def test_p109_word_detector_maps_detected_name_to_word_timing() -> None:
    text = "The name is Mr. Smith."
    words = word_timings_from_text(text, 100)
    spans = P109WordSpanDetector().detect_spans(text, words)
    assert len(spans) == 1
    assert spans[0].category is SpanCategory.PERSON_NAME
    assert 0 <= spans[0].start_ms < spans[0].end_ms <= 100


def test_default_api_expands_bare_title_to_full_cued_name() -> None:
    """P1-09 reports only "Mr"; P3-10 must still cover "Smith"."""
    text = "The name is Mr. Smith."
    findings = detect_pii(text)
    assert [(finding.start, finding.end) for finding in findings] == [(12, 14)]
    assert [(finding.start, finding.end) for finding in expand_cued_name_spans(findings, text)] == [
        (12, 21)
    ]
    spans = resolve_char_spans(findings, text)
    assert [(span.start, span.end, span.category) for span in spans] == [
        (12, 21, SpanCategory.PERSON_NAME)
    ]


def test_default_audio_detector_covers_full_cued_name() -> None:
    """Without known_names, both the title and surname word spans are mapped."""
    text = "The name is Mr. Smith."
    words = word_timings_from_text(text, 100)
    default = P109WordSpanDetector().detect_spans(text, words)
    reviewed = P109WordSpanDetector().detect_spans(text, words, known_names=["Mr. Smith"])
    assert default == reviewed
    assert default[0].start_ms == words[3].start_ms
    assert default[0].end_ms == words[4].end_ms


def test_bare_title_without_capitalized_continuation_fails_closed() -> None:
    text = "The name is Mr."
    with pytest.raises(UnverifiedRedactionError) as exc:
        resolve_char_spans(detect_pii(text), text)
    assert exc.value.code == "unverified_cued_name"
    with pytest.raises(UnverifiedRedactionError):
        P109WordSpanDetector().detect_spans(text, word_timings_from_text(text, 100))


def test_known_name_span_matches_default_full_cued_name() -> None:
    text = "The name is Mr. Smith."
    words = word_timings_from_text(text, 100)
    spans = P109WordSpanDetector().detect_spans(text, words, known_names=["Mr. Smith"])
    assert spans == P109WordSpanDetector().detect_spans(text, words)
    assert len(spans) == 1


def test_unmapped_span_fails_closed() -> None:
    detector = P109WordSpanDetector()
    with pytest.raises(UnmappedSpanError) as exc:
        detector.detect_spans("The name is Ada.", ())
    assert exc.value.code == "unmapped_pii_span"


def test_card_findings_use_p109_detector_without_reimplementation() -> None:
    text = "Pay 4111 1111 1111 1111 now."
    findings = P109TranscriptDetector().detect(text)
    assert [finding.category for finding in findings] == [PiiCategory.CARD_NUMBER]
    spans = resolve_char_spans(findings, text)
    assert spans[0].category is SpanCategory.CARD_NUMBER
    assert (spans[0].start, spans[0].end) == (findings[0].start, findings[0].end)
    assert detect_pii(text)[0].category is PiiCategory.CARD_NUMBER


def test_synthetic_finding_adapter_can_be_injected() -> None:
    class StubDetector:
        """Return one synthetic phone finding without calling P1-09."""

        def detect(self, text: str) -> tuple[PiiFinding, ...]:
            """Return a fixed offset so mapping is fully deterministic."""
            assert text == "hello world"
            return (PiiFinding(PiiCategory.PHONE, 6, 11),)

    spans = P109WordSpanDetector(StubDetector()).detect_spans(
        "hello world", word_timings_from_text("hello world", 100)
    )
    assert len(spans) == 1
    assert spans[0].category is SpanCategory.PHONE
