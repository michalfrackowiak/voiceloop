"""Tests for native-rate audio capture helpers."""

import struct
import wave
from pathlib import Path

import pytest

from sdk import capture_audio, capture_pcm, capture_wav


def _frames(count: int) -> bytes:
    """Return deterministic signed 16-bit PCM frames."""
    return b"".join(struct.pack("<h", value % 1000) for value in range(count))


def _write_wav(path: Path, sample_rate_hz: int, frame_count: int) -> bytes:
    """Write a mono 16-bit WAV and return its raw PCM frames."""
    frames = _frames(frame_count)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(frames)
    return frames


@pytest.mark.parametrize("sample_rate_hz", [8000, 16000])
def test_capture_wav_preserves_header_rate_and_raw_frames(
    tmp_path: Path,
    sample_rate_hz: int,
) -> None:
    path = tmp_path / "turn.wav"
    frames = _write_wav(path, sample_rate_hz, 800)

    captured = capture_wav(path, uri="s3://bucket/turn.wav", include_frames=True)

    assert captured.sample_rate_hz == sample_rate_hz
    assert captured.channels == 1
    assert captured.duration_ms == 800 * 1000 // sample_rate_hz
    assert captured.frames == frames
    assert captured.event_audio() == {
        "uri": "s3://bucket/turn.wav",
        "sample_rate_hz": sample_rate_hz,
        "channels": 1,
        "duration_ms": captured.duration_ms,
    }


def test_capture_audio_alias_reads_native_8k_header(tmp_path: Path) -> None:
    path = tmp_path / "telephony.wav"
    _write_wav(path, 8000, 400)

    captured = capture_audio(path)

    assert captured.sample_rate_hz == 8000
    assert captured.uri == path.as_posix()
    assert captured.frames is None


def test_capture_pcm_keeps_8k_frames_unchanged() -> None:
    frames = _frames(800)

    captured = capture_pcm(
        frames,
        uri="memory://turn-0",
        sample_rate_hz=8000,
        channels=1,
    )

    assert captured.frames is frames
    assert captured.sample_rate_hz == 8000
    assert captured.duration_ms == 100
    assert captured.event_audio()["sample_rate_hz"] == 8000


def test_capture_pcm_rejects_invalid_native_rate() -> None:
    with pytest.raises(ValueError):
        capture_pcm(b"", uri="memory://turn", sample_rate_hz=0, channels=1)
