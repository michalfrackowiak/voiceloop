"""Fake waveform redaction: conservative zeroing, merging and fail-closed proof."""

from __future__ import annotations

import pytest

from services.datasets.redaction.audio import (
    FakeAudio,
    InMemoryAudioStore,
    assert_no_unredacted_span,
    redact_audio,
    redact_audio_uri,
)
from services.datasets.redaction.settings import (
    RedactionSettings,
    SystemClock,
)
from services.datasets.redaction.spans import (
    InvalidSpanError,
    P109WordSpanDetector,
    SpanCategory,
    TimestampSpan,
    UnverifiedRedactionError,
    word_timings_from_text,
)
from voiceloop_common import StorageError


def _span(start: int, end: int) -> TimestampSpan:
    """Build a person-name timestamp span."""
    return TimestampSpan(start, end, SpanCategory.PERSON_NAME)


def test_fake_audio_is_deterministic_and_preserves_sample_rate() -> None:
    first = FakeAudio.synthetic("fake://tone", 8000, 300, seed=2)
    second = FakeAudio.synthetic("fake://tone", 8000, 300, seed=2)
    assert first == second
    assert first.sample_rate_hz == 8000
    assert first.sample_count == 2400
    assert first.envelope(0, 300) > 0
    assert not first.redacted


def test_fake_audio_rejects_sample_count_mismatch() -> None:
    with pytest.raises(InvalidSpanError) as exc:
        FakeAudio("fake://bad", 8000, 100, (1, 2, 3))
    assert exc.value.code == "invalid_audio"


def test_envelope_windows_are_deterministic() -> None:
    audio = FakeAudio.synthetic("fake://tone", 8000, 100, seed=1)
    assert audio.envelope_windows(20) == audio.envelope_windows(20)
    assert len(audio.envelope_windows(20)) == 5
    with pytest.raises(InvalidSpanError):
        audio.envelope_windows(0)


def test_redact_audio_zeroes_supplied_and_padded_spans(
    settings: RedactionSettings,
) -> None:
    audio = FakeAudio.synthetic("fake://speech", 16000, 500, seed=3)
    result = redact_audio(audio, [_span(100, 200)], settings=settings)
    assert result.audio.redacted
    assert result.merged_spans == (TimestampSpan(80, 220, SpanCategory.PERSON_NAME),)
    assert result.zeroed_samples > 0
    assert result.audio.envelope(80, 220) == 0
    assert result.audio.envelope(0, 500) > 0
    assert_no_unredacted_span(result.audio, [_span(100, 200)])
    assert result.to_dict()["synthetic"] is True


def test_default_audio_api_zeroes_full_cued_name_without_known_names(
    settings: RedactionSettings,
) -> None:
    """Regression: "Mr" alone must not leave "Smith." audible in fake audio."""
    text = "The name is Mr. Smith."
    words = word_timings_from_text(text, 1000)
    spans = P109WordSpanDetector().detect_spans(text, words)
    assert len(spans) == 1
    assert (spans[0].start_ms, spans[0].end_ms) == (words[3].start_ms, words[4].end_ms)
    audio = FakeAudio.synthetic("fake://cued", 1000, 1000, seed=11)
    result = redact_audio(audio, spans, settings=settings)
    assert result.audio.envelope(words[3].start_ms, words[3].end_ms) == 0
    assert result.audio.envelope(words[4].start_ms, words[4].end_ms) == 0
    assert_no_unredacted_span(result.audio, spans)


def test_overlapping_spans_merge_and_leave_no_gap(settings: RedactionSettings) -> None:
    audio = FakeAudio.synthetic("fake://speech", 8000, 400, seed=4)
    result = redact_audio(
        audio,
        [_span(50, 150), _span(150, 250), _span(120, 180)],
        settings=settings,
    )
    assert len(result.merged_spans) == 1
    assert result.merged_spans[0].start_ms == 30
    assert result.merged_spans[0].end_ms == 270
    assert result.audio.envelope(30, 270) == 0
    assert result.verified_spans >= 3


@pytest.mark.parametrize(
    ("start", "end", "code"),
    [
        (-5, 10, "negative_span"),
        (30, 20, "reversed_span"),
        (0, 501, "span_beyond_duration"),
    ],
)
def test_invalid_spans_fail_closed(
    settings: RedactionSettings, start: int, end: int, code: str
) -> None:
    audio = FakeAudio.synthetic("fake://speech", 8000, 500, seed=5)
    with pytest.raises(InvalidSpanError) as exc:
        redact_audio(audio, [_span(start, end)], settings=settings)
    assert exc.value.code == code
    assert audio.envelope(0, 500) > 0


def test_verification_catches_unredacted_span() -> None:
    audio = FakeAudio.synthetic("fake://speech", 8000, 200, seed=6)
    with pytest.raises(UnverifiedRedactionError) as exc:
        assert_no_unredacted_span(audio, [_span(50, 100)])
    assert exc.value.code == "unverified_audio_span"
    assert int(exc.value.details["peak"]) > 0


def test_audio_store_and_redacted_copy_are_fake_only(
    settings: RedactionSettings,
) -> None:
    store = InMemoryAudioStore()
    audio = FakeAudio.synthetic("fake://raw", 8000, 200, seed=7)
    store.put_raw(audio)
    result = redact_audio_uri("fake://raw", store, [_span(20, 80)], settings=settings)
    assert store.load_redacted(result.redacted_uri) == result.audio
    assert store.load_raw("fake://raw") == audio
    with pytest.raises(StorageError):
        store.load_raw("fake://missing")
    with pytest.raises(StorageError):
        store.load_redacted("fake-redacted://missing")


def test_redaction_audit_events_are_pii_free(
    settings: RedactionSettings,
) -> None:
    from services.datasets.redaction.settings import InMemoryAuditSink

    audit = InMemoryAuditSink()
    audio = FakeAudio.synthetic("fake://raw", 16000, 200, seed=8)
    redact_audio(
        audio,
        [_span(20, 60)],
        settings=settings,
        audit=audit,
        clock=SystemClock(),
    )
    assert len(audit.events) == 1
    payload = str(audit.to_dicts())
    assert "PERSON_" not in payload
    assert "fake://raw" in payload
    assert audit.events[0].event_type == "audio_redacted"
