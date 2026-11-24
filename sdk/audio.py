"""Audio capture helpers that preserve the original sample rate."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AudioCapture:
    """Native-rate audio metadata ready for a turn event's ``audio`` block."""

    uri: str
    sample_rate_hz: int
    channels: int
    duration_ms: int
    frames: bytes | None = None

    def event_audio(self) -> dict[str, Any]:
        """Return the schema-shaped audio block, excluding raw frames."""
        return {
            "uri": self.uri,
            "sample_rate_hz": self.sample_rate_hz,
            "channels": self.channels,
            "duration_ms": self.duration_ms,
        }


def capture_wav(
    path: str | Path,
    *,
    uri: str | None = None,
    include_frames: bool = False,
) -> AudioCapture:
    """Read WAV metadata with stdlib ``wave``; never resample.

    Args:
        path: WAV file to inspect.
        uri: Audio URI to record in the event; defaults to the POSIX path.
        include_frames: When true, retain the raw PCM frames unchanged.

    Returns:
        Native sample rate, channel count, duration and optional frames.

    """
    wav_path = Path(path)
    with wave.open(str(wav_path), "rb") as handle:
        sample_rate_hz = handle.getframerate()
        channels = handle.getnchannels()
        frame_count = handle.getnframes()
        frames = handle.readframes(frame_count) if include_frames else None
    duration_ms = round(frame_count * 1000 / sample_rate_hz) if sample_rate_hz > 0 else 0
    return AudioCapture(
        uri=uri or wav_path.as_posix(),
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        duration_ms=duration_ms,
        frames=frames,
    )


def capture_audio(
    source: str | Path,
    *,
    uri: str | None = None,
    include_frames: bool = False,
) -> AudioCapture:
    """Capture native-rate metadata from a WAV source.

    This is a discoverable alias for :func:`capture_wav`; both read the header
    as-is so 8 kHz telephony audio stays at 8 kHz.
    """
    return capture_wav(source, uri=uri, include_frames=include_frames)


def capture_pcm(
    frames: bytes,
    *,
    uri: str,
    sample_rate_hz: int,
    channels: int,
    sample_width_bytes: int = 2,
) -> AudioCapture:
    """Wrap raw PCM frames at their existing rate; no conversion is performed.

    Args:
        frames: Raw PCM bytes exactly as captured (8 kHz input stays 8 kHz).
        uri: Audio URI to record in the turn event.
        sample_rate_hz: Native sample rate reported by the capture source.
        channels: Native channel count.
        sample_width_bytes: Bytes per sample per channel; 16-bit PCM is the default.

    Returns:
        An :class:`AudioCapture` holding the unchanged frames.

    Raises:
        ValueError: If any rate, channel or width value is not positive.

    """
    if sample_rate_hz <= 0 or channels <= 0 or sample_width_bytes <= 0:
        raise ValueError("sample_rate_hz, channels and sample_width_bytes must be positive")
    bytes_per_frame = sample_rate_hz * channels * sample_width_bytes
    duration_ms = round(len(frames) * 1000 / bytes_per_frame)
    return AudioCapture(
        uri=uri,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        duration_ms=duration_ms,
        frames=frames,
    )
