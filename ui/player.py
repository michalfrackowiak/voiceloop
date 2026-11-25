"""Deterministic synthetic waveform player state and token diff.

No audio bytes are decoded, stored or played. The waveform amplitude sequence
is derived only from a SHA-256 digest of the pseudonymous item ID, so it is
reproducible and carries no information about any real recording.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any

from .constants import ALLOWED_SPEEDS, LOW_CONFIDENCE_CEILING, WAVEFORM_BARS
from .errors import UiValidationError


@dataclass(frozen=True)
class LowConfidenceSpan:
    """A highlighted time range carrying only a pseudonymous token ID."""

    start_ms: int
    end_ms: int
    token_id: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable highlight span."""
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "token_id": self.token_id,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class WaveformBar:
    """One synthetic amplitude bar with a low-confidence flag."""

    index: int
    amplitude: int
    low_confidence: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable bar for the browser renderer."""
        return {
            "index": self.index,
            "amplitude": self.amplitude,
            "low_confidence": self.low_confidence,
        }


@dataclass(frozen=True)
class PlayerState:
    """Pure state of the synthetic timeline; safe to serialize and compare."""

    item_id: str
    duration_ms: int
    position_ms: int
    speed: float
    loop: bool
    playing: bool
    loop_count: int
    bars: tuple[WaveformBar, ...]
    spans: tuple[LowConfidenceSpan, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable player state snapshot."""
        return {
            "item_id": self.item_id,
            "duration_ms": self.duration_ms,
            "position_ms": self.position_ms,
            "speed": self.speed,
            "loop": self.loop,
            "playing": self.playing,
            "loop_count": self.loop_count,
            "audio_playback": False,
            "waveform_source": "sha256_of_pseudonymous_item_id",
            "bars": [bar.to_dict() for bar in self.bars],
            "spans": [span.to_dict() for span in self.spans],
        }


@dataclass(frozen=True)
class DiffOp:
    """One token-level diff operation between hypothesis and correction."""

    op: str
    token: str
    hypothesis_index: int | None
    corrected_index: int | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable diff operation."""
        return {
            "op": self.op,
            "token": self.token,
            "hypothesis_index": self.hypothesis_index,
            "corrected_index": self.corrected_index,
        }


def synth_waveform(item_id: str, bar_count: int = WAVEFORM_BARS) -> tuple[int, ...]:
    """Derive amplitudes from a pseudonymous ID digest, never from audio bytes.

    Args:
        item_id: Non-empty pseudonymous item ID.
        bar_count: Number of bars to emit; must be positive.

    Returns:
        A tuple of integer amplitudes in ``[8, 96]``.

    Raises:
        UiValidationError: If the ID is empty or the bar count is invalid.

    """
    if not item_id or not isinstance(item_id, str):
        raise UiValidationError("item_id is required for synthetic waveform")
    if isinstance(bar_count, bool) or not isinstance(bar_count, int) or bar_count < 1:
        raise UiValidationError("bar_count must be a positive integer")
    digest = hashlib.sha256(item_id.encode("utf-8")).digest()
    return tuple(8 + digest[index % len(digest)] % 89 for index in range(bar_count))


def new_player(item_id: str, duration_ms: int, spans: tuple[LowConfidenceSpan, ...]) -> PlayerState:
    """Create a paused player for a synthetic timeline.

    Args:
        item_id: Pseudonymous item ID.
        duration_ms: Positive synthetic duration in milliseconds.
        spans: Explicit low-confidence spans, already screened for pseudonyms.

    Returns:
        A paused player with deterministic waveform bars and highlights.

    Raises:
        UiValidationError: If duration or any span is malformed.

    """
    if isinstance(duration_ms, bool) or not isinstance(duration_ms, int) or duration_ms <= 0:
        raise UiValidationError("duration_ms must be a positive integer")
    for span in spans:
        if not isinstance(span.start_ms, int) or not isinstance(span.end_ms, int):
            raise UiValidationError("span offsets must be integers")
        if span.start_ms < 0 or span.start_ms >= span.end_ms or span.end_ms > duration_ms:
            raise UiValidationError("span must be inside the timeline")
        if not span.token_id:
            raise UiValidationError("span token_id is required")
        if isinstance(span.confidence, bool) or not 0.0 <= span.confidence <= 1.0:
            raise UiValidationError("span confidence must be in [0, 1]")
    amplitudes = synth_waveform(item_id)
    step = duration_ms / len(amplitudes)
    bars: list[WaveformBar] = []
    for index, amplitude in enumerate(amplitudes):
        start = index * step
        end = (index + 1) * step
        highlighted = any(
            span.confidence < LOW_CONFIDENCE_CEILING and span.start_ms < end and span.end_ms > start
            for span in spans
        )
        bars.append(WaveformBar(index=index, amplitude=amplitude, low_confidence=highlighted))
    return PlayerState(
        item_id=item_id,
        duration_ms=duration_ms,
        position_ms=0,
        speed=1.0,
        loop=False,
        playing=False,
        loop_count=0,
        bars=tuple(bars),
        spans=spans,
    )


def with_speed(state: PlayerState, speed: float) -> PlayerState:
    """Set the playback speed from the fixed, deterministic speed list."""
    if speed not in ALLOWED_SPEEDS:
        raise UiValidationError(f"unsupported speed: {speed!r}")
    return replace(state, speed=speed)


def with_loop(state: PlayerState, loop: bool) -> PlayerState:
    """Enable or disable deterministic loop behaviour."""
    if not isinstance(loop, bool):
        raise UiValidationError("loop must be a boolean")
    return replace(state, loop=loop)


def seek(state: PlayerState, position_ms: int) -> PlayerState:
    """Move the synthetic cursor, clamped to the timeline bounds."""
    if isinstance(position_ms, bool) or not isinstance(position_ms, int):
        raise UiValidationError("position_ms must be an integer")
    return replace(state, position_ms=max(0, min(position_ms, state.duration_ms)))


def tick(state: PlayerState, delta_ms: int) -> PlayerState:
    """Advance the synthetic cursor by ``delta_ms * speed`` milliseconds.

    With looping enabled the cursor wraps and ``loop_count`` increments;
    otherwise it stops exactly at the end. No real time or audio is involved.
    """
    if isinstance(delta_ms, bool) or not isinstance(delta_ms, int) or delta_ms < 0:
        raise UiValidationError("delta_ms must be a non-negative integer")
    advanced = state.position_ms + int(delta_ms * state.speed)
    if advanced < state.duration_ms:
        return replace(state, position_ms=advanced, playing=state.playing)
    if state.loop:
        if state.duration_ms == 0:  # pragma: no cover - new_player forbids zero
            return replace(state, position_ms=0, loop_count=state.loop_count + 1)
        return replace(
            state,
            position_ms=advanced % state.duration_ms,
            loop_count=state.loop_count + 1,
            playing=True,
        )
    return replace(state, position_ms=state.duration_ms, playing=False)


def diff_tokens(hypothesis: tuple[str, ...], corrected: tuple[str, ...]) -> tuple[DiffOp, ...]:
    """Compute a deterministic token diff using Levenshtein backtracking.

    Ties prefer ``equal``, then ``replace``, then ``delete``, then ``insert``.
    Tokens are pseudonymous in all committed fixtures.
    """
    rows = len(hypothesis) + 1
    cols = len(corrected) + 1
    distance = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        distance[i][0] = i
    for j in range(cols):
        distance[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            substitution = distance[i - 1][j - 1] + (hypothesis[i - 1] != corrected[j - 1])
            distance[i][j] = min(substitution, distance[i - 1][j] + 1, distance[i][j - 1] + 1)
    ops: list[DiffOp] = []
    i, j = rows - 1, cols - 1
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and hypothesis[i - 1] == corrected[j - 1]
            and distance[i][j] == distance[i - 1][j - 1]
        ):
            ops.append(DiffOp("equal", hypothesis[i - 1], i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and distance[i][j] == distance[i - 1][j - 1] + 1:
            ops.append(DiffOp("replace", corrected[j - 1], i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and distance[i][j] == distance[i - 1][j] + 1:
            ops.append(DiffOp("delete", hypothesis[i - 1], i - 1, None))
            i -= 1
        else:
            ops.append(DiffOp("insert", corrected[j - 1], None, j - 1))
            j -= 1
    return tuple(reversed(ops))
