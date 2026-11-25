"""Deterministic fake audio and conservative timestamp-based redaction (P3-10.1).

Audio is modelled as a signed 16-bit fake waveform with an explicit sample rate
and duration; there are no audio libraries, codecs, object stores or speech
recognition calls. Every supplied PII span is validated, padded, merged with
overlapping/adjacent spans and zeroed. Verification checks the fake envelope
of both supplied and merged spans, so a missed span fails closed.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from voiceloop_common import StorageError

from .settings import (
    COMPLIANCE_STATUS,
    POLICY_STATUS,
    SYNTHETIC_LABEL,
    AuditEvent,
    AuditSink,
    Clock,
    InMemoryAuditSink,
    RedactionSettings,
    SystemClock,
)
from .spans import (
    InvalidSpanError,
    TimestampSpan,
    UnverifiedRedactionError,
    merge_spans,
    validate_span,
)


@dataclass(frozen=True, slots=True)
class FakeAudio:
    """Fake signed 16-bit waveform; never real guest audio.

    Attributes:
        uri: Synthetic object reference, never resolved against a real store.
        sample_rate_hz: Explicit sample rate; 8 kHz fixtures are never resampled.
        duration_ms: Explicit duration in milliseconds.
        samples: Exact integer sample count for rate/duration.
        redacted: Whether this instance is a redacted copy.

    """

    uri: str
    sample_rate_hz: int
    duration_ms: int
    samples: tuple[int, ...]
    redacted: bool = False

    def __post_init__(self) -> None:
        """Fail closed on a malformed fake waveform; accept list-like samples."""
        object.__setattr__(self, "samples", tuple(self.samples))
        if self.sample_rate_hz <= 0 or self.duration_ms < 0:
            raise InvalidSpanError(
                "invalid_audio",
                "sample rate must be positive and duration non-negative",
                {
                    "sample_rate_hz": self.sample_rate_hz,
                    "duration_ms": self.duration_ms,
                },
            )
        expected = self.sample_rate_hz * self.duration_ms // 1000
        if len(self.samples) != expected:
            raise InvalidSpanError(
                "invalid_audio",
                "sample count must match sample rate and duration exactly",
                {"expected_samples": expected, "actual_samples": len(self.samples)},
            )
        if any(not -32768 <= sample <= 32767 for sample in self.samples):
            raise InvalidSpanError("invalid_audio", "samples must be signed 16-bit integers")

    @classmethod
    def synthetic(
        cls,
        uri: str,
        sample_rate_hz: int,
        duration_ms: int,
        *,
        seed: int = 0,
        amplitude: int = 1000,
    ) -> FakeAudio:
        """Generate a deterministic fake tone; no speech and no real audio.

        Args:
            uri: Synthetic object reference to carry through reports.
            sample_rate_hz: Explicit native sample rate.
            duration_ms: Explicit duration in milliseconds.
            seed: Deterministic phase/frequency selector.
            amplitude: Peak absolute 16-bit amplitude (1..32767).

        Returns:
            A fake waveform labelled by its synthetic constructor.

        """
        if not 1 <= amplitude <= 32767:
            raise InvalidSpanError("invalid_audio", "amplitude must be between 1 and 32767")
        count = sample_rate_hz * duration_ms // 1000
        phase = (seed % 7) * 0.31
        frequency = (seed % 3 + 1) * 220.0
        samples = tuple(
            int(
                round(
                    amplitude * math.sin(2 * math.pi * frequency * index / sample_rate_hz + phase)
                )
            )
            for index in range(count)
        )
        return cls(uri, sample_rate_hz, duration_ms, samples)

    @property
    def sample_count(self) -> int:
        """Return the exact number of fake samples."""
        return len(self.samples)

    def sample_bounds(self, start_ms: int, end_ms: int) -> tuple[int, int]:
        """Map a half-open millisecond span to conservative sample indices.

        Floor/ceil rounding ensures every sample whose timestamp touches the
        span is included; bounds are clamped to the waveform.

        Args:
            start_ms: Inclusive span start in milliseconds.
            end_ms: Exclusive span end in milliseconds.

        Returns:
            Half-open sample index bounds.

        """
        if start_ms < 0 or end_ms < start_ms:
            raise InvalidSpanError(
                "invalid_span",
                "envelope span must be non-negative and ordered",
                {"start_ms": start_ms, "end_ms": end_ms},
            )
        start = min(self.sample_count, (start_ms * self.sample_rate_hz) // 1000)
        end = min(self.sample_count, (end_ms * self.sample_rate_hz + 999) // 1000)
        return start, max(start, end)

    def envelope(self, start_ms: int, end_ms: int) -> int:
        """Return the peak absolute amplitude inside a timestamped span.

        Args:
            start_ms: Inclusive span start in milliseconds.
            end_ms: Exclusive span end in milliseconds.

        Returns:
            Maximum absolute sample value, or 0 for an empty span.

        """
        start, end = self.sample_bounds(start_ms, end_ms)
        return max((abs(sample) for sample in self.samples[start:end]), default=0)

    def envelope_windows(self, window_ms: int) -> tuple[int, ...]:
        """Return a deterministic per-window envelope of the fake waveform.

        Args:
            window_ms: Positive window width in milliseconds.

        Returns:
            Peak absolute amplitude for each window of the waveform.

        """
        if window_ms <= 0:
            raise InvalidSpanError(
                "invalid_window", "envelope window must be positive", {"window_ms": window_ms}
            )
        return tuple(
            self.envelope(start, min(self.duration_ms, start + window_ms))
            for start in range(0, self.duration_ms, window_ms)
        )


class AudioStore(Protocol):
    """Injectable raw/redacted audio store seam; implementations are in-memory fakes."""

    def load_raw(self, uri: str) -> FakeAudio:
        """Load a raw fake waveform by synthetic URI."""
        ...

    def save_redacted(self, uri: str, audio: FakeAudio) -> str:
        """Persist a redacted fake waveform and return its synthetic URI."""
        ...


@dataclass(slots=True)
class InMemoryAudioStore:
    """Fake object-store stand-in; no bytes leave this process."""

    raw: dict[str, FakeAudio] = field(default_factory=dict)
    redacted: dict[str, FakeAudio] = field(default_factory=dict)

    def put_raw(self, audio: FakeAudio) -> None:
        """Store one raw fake waveform under its URI."""
        self.raw[audio.uri] = audio

    def load_raw(self, uri: str) -> FakeAudio:
        """Load a raw fake waveform, failing closed when it is absent."""
        if uri not in self.raw:
            raise StorageError(f"unknown synthetic raw audio URI: {uri}")
        return self.raw[uri]

    def save_redacted(self, uri: str, audio: FakeAudio) -> str:
        """Store a redacted copy and return a deterministic fake URI."""
        redacted_uri = f"fake-redacted://{uri}"
        self.redacted[redacted_uri] = audio
        return redacted_uri

    def load_redacted(self, uri: str) -> FakeAudio:
        """Load a previously saved redacted fake waveform."""
        if uri not in self.redacted:
            raise StorageError(f"unknown synthetic redacted audio URI: {uri}")
        return self.redacted[uri]

    @property
    def raw_uris(self) -> tuple[str, ...]:
        """Return raw URIs in deterministic order."""
        return tuple(sorted(self.raw))


@dataclass(frozen=True, slots=True)
class AudioRedactionResult:
    """Auditable, PII-free summary of one fake-audio redaction."""

    source_uri: str
    redacted_uri: str
    audio: FakeAudio
    detected_spans: tuple[TimestampSpan, ...]
    merged_spans: tuple[TimestampSpan, ...]
    zeroed_samples: int
    verified_spans: int

    def to_dict(self) -> dict[str, object]:
        """Return offsets, counts and policy labels; never waveform values."""
        return {
            "source_uri": self.source_uri,
            "redacted_uri": self.redacted_uri,
            "sample_rate_hz": self.audio.sample_rate_hz,
            "duration_ms": self.audio.duration_ms,
            "detected_spans": [span.to_dict() for span in self.detected_spans],
            "merged_spans": [span.to_dict() for span in self.merged_spans],
            "zeroed_samples": self.zeroed_samples,
            "verified_spans": self.verified_spans,
            "policy_status": POLICY_STATUS,
            "compliance_status": COMPLIANCE_STATUS,
            "synthetic": True,
            "label": SYNTHETIC_LABEL,
        }


def assert_no_unredacted_span(audio: FakeAudio, spans: Iterable[TimestampSpan]) -> None:
    """Prove that every supplied timestamp span is zero in the fake envelope.

    Args:
        audio: Redacted fake waveform to verify.
        spans: Spans that must contain only zero samples.

    Raises:
        InvalidSpanError: A supplied span is malformed for this waveform.
        UnverifiedRedactionError: Any span still has non-zero fake amplitude.

    """
    checked = 0
    for span in spans:
        validate_span(span, audio.duration_ms)
        peak = audio.envelope(span.start_ms, span.end_ms)
        if peak != 0:
            raise UnverifiedRedactionError(
                "unverified_audio_span",
                "a PII span still has non-zero fake amplitude",
                {"start_ms": span.start_ms, "end_ms": span.end_ms, "peak": peak},
            )
        checked += 1


def redact_audio(
    audio: FakeAudio,
    spans: Sequence[TimestampSpan],
    *,
    settings: RedactionSettings | None = None,
    store: AudioStore | None = None,
    audit: AuditSink | None = None,
    clock: Clock | None = None,
) -> AudioRedactionResult:
    """Zero every validated, padded and merged PII span in fake audio.

    Args:
        audio: Raw synthetic waveform with explicit sample rate/duration.
        spans: Timestamp spans supplied by an injected detector.
        settings: Typed ``VOICELOOP_REDACTION_`` settings; defaults to env.
        store: Injected audio store; defaults to an in-memory fake.
        audit: Injected PII-free audit sink; defaults to an in-memory fake.
        clock: Injected UTC clock for audit timestamps.

    Returns:
        Redacted fake audio plus a PII-free report and synthetic redacted URI.

    Raises:
        InvalidSpanError: Any supplied span or padding is invalid.
        UnverifiedRedactionError: Verification found a non-zero span.

    """
    config = settings or RedactionSettings()
    audio_store = store or InMemoryAudioStore()
    audit_sink = audit or InMemoryAuditSink()
    audit_clock = clock or SystemClock()

    merged = merge_spans(spans, padding_ms=config.padding_ms, duration_ms=audio.duration_ms)
    samples = list(audio.samples)
    changed: set[int] = set()
    for span in merged:
        start, end = audio.sample_bounds(span.start_ms, span.end_ms)
        for index in range(start, end):
            if samples[index] != 0:
                changed.add(index)
                samples[index] = 0

    redacted = FakeAudio(
        uri=audio.uri,
        sample_rate_hz=audio.sample_rate_hz,
        duration_ms=audio.duration_ms,
        samples=tuple(samples),
        redacted=True,
    )
    verify_spans = tuple(dict.fromkeys((*spans, *merged)))
    assert_no_unredacted_span(redacted, verify_spans)
    redacted_uri = audio_store.save_redacted(audio.uri, redacted)

    audit_sink.emit(
        AuditEvent(
            event_type="audio_redacted",
            occurred_at=audit_clock.now(),
            details={
                "source_uri": audio.uri,
                "redacted_uri": redacted_uri,
                "sample_rate_hz": audio.sample_rate_hz,
                "duration_ms": audio.duration_ms,
                "detected_spans": len(spans),
                "merged_spans": len(merged),
                "zeroed_samples": len(changed),
                "policy_status": POLICY_STATUS,
                "label": SYNTHETIC_LABEL,
            },
        )
    )
    return AudioRedactionResult(
        source_uri=audio.uri,
        redacted_uri=redacted_uri,
        audio=redacted,
        detected_spans=tuple(spans),
        merged_spans=merged,
        zeroed_samples=len(changed),
        verified_spans=len(verify_spans),
    )


def redact_audio_uri(
    uri: str,
    store: AudioStore,
    spans: Sequence[TimestampSpan],
    *,
    settings: RedactionSettings | None = None,
    audit: AuditSink | None = None,
    clock: Clock | None = None,
) -> AudioRedactionResult:
    """Load raw fake audio from an injected store and redact it.

    Args:
        uri: Synthetic raw audio URI.
        store: Injected audio store seam.
        spans: Validated timestamp spans to zero.
        settings: Typed redaction settings.
        audit: Injected PII-free audit sink.
        clock: Injected UTC clock for audit timestamps.

    Returns:
        The redaction result for the loaded fake waveform.

    """
    return redact_audio(
        store.load_raw(uri),
        spans,
        settings=settings,
        store=store,
        audit=audit,
        clock=clock,
    )


__all__ = [
    "AudioRedactionResult",
    "AudioStore",
    "FakeAudio",
    "InMemoryAudioStore",
    "assert_no_unredacted_span",
    "redact_audio",
    "redact_audio_uri",
]
