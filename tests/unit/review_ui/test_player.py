"""Deterministic synthetic waveform, low-confidence and diff tests."""

from __future__ import annotations

import pytest

from ui.constants import LOW_CONFIDENCE_CEILING, WAVEFORM_BARS
from ui.errors import UiValidationError
from ui.player import (
    LowConfidenceSpan,
    diff_tokens,
    new_player,
    seek,
    synth_waveform,
    tick,
    with_loop,
    with_speed,
)


def test_waveform_is_deterministic_from_pseudonymous_id() -> None:
    """Amplitudes repeat for one ID, differ for another and carry no audio."""
    first = synth_waveform("item-token-0001")
    second = synth_waveform("item-token-0001")
    other = synth_waveform("item-token-0002")

    assert first == second
    assert first != other
    assert len(first) == WAVEFORM_BARS
    assert all(isinstance(value, int) and 8 <= value <= 96 for value in first)


def test_waveform_rejects_missing_id_and_invalid_bar_count() -> None:
    """Malformed synthetic inputs fail closed."""
    with pytest.raises(UiValidationError):
        synth_waveform("")
    with pytest.raises(UiValidationError):
        synth_waveform("item-token-0001", bar_count=0)


def test_low_confidence_spans_only_highlight_below_ceiling() -> None:
    """Only fixture spans below the confidence ceiling highlight bars."""
    spans = (
        LowConfidenceSpan(20, 120, "tok-b", 0.41),
        LowConfidenceSpan(160, 300, "tok-g", 0.85),
    )
    player = new_player("item-token-0003", 400, spans)

    assert LOW_CONFIDENCE_CEILING == 0.60
    assert any(bar.low_confidence for bar in player.bars)
    assert not player.bars[20].low_confidence
    assert player.to_dict()["spans"][1]["token_id"] == "tok-g"


def test_new_player_rejects_out_of_bounds_spans() -> None:
    """A span outside the synthetic timeline fails closed."""
    bad_span = (LowConfidenceSpan(0, 501, "tok-a", 0.4),)
    with pytest.raises(UiValidationError):
        new_player("item-token-0001", 500, bad_span)


def test_speed_loop_and_seek_are_deterministic_state() -> None:
    """Speed, loop count and cursor are pure state transitions."""
    state = with_loop(with_speed(new_player("item-token-0001", 100, ()), 2.0), True)

    forward = tick(state, 30)
    wrapped = tick(forward, 30)
    assert forward.position_ms == 60
    assert wrapped.position_ms == 20
    assert wrapped.loop_count == 1
    assert wrapped.playing is True

    assert seek(wrapped, -5).position_ms == 0
    assert seek(wrapped, 1000).position_ms == 100
    assert tick(with_loop(state, False), 500).position_ms == 100


def test_unsupported_speed_fails_closed() -> None:
    """Only the documented deterministic speed list is accepted."""
    with pytest.raises(UiValidationError):
        with_speed(new_player("item-token-0001", 100, ()), 1.7)


def test_player_state_declares_no_audio_playback() -> None:
    """Serialized player state never claims real audio."""
    payload = new_player("item-token-0001", 320, ()).to_dict()

    assert payload["audio_playback"] is False
    assert payload["waveform_source"] == "sha256_of_pseudonymous_item_id"
    assert payload["position_ms"] == 0


def test_diff_tokens_is_deterministic_and_pseudonymous() -> None:
    """Token diffs are stable and expose only pseudonymous tokens."""
    hypothesis = ("tok-a", "tok-b", "tok-c")
    corrected = ("tok-a", "tok-x", "tok-c")
    first = diff_tokens(hypothesis, corrected)
    second = diff_tokens(hypothesis, corrected)

    assert first == second
    assert [(op.op, op.token) for op in first] == [
        ("equal", "tok-a"),
        ("replace", "tok-x"),
        ("equal", "tok-c"),
    ]
    assert all(" " not in op.token for op in first)
