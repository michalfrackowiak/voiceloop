"""Deterministic line-noise and telephony-band stand-in tests for P3-03.2."""

from __future__ import annotations

import pytest

from services.datasets.augment.noise import (
    NoiseProfile,
    TelephonyBand,
    add_line_noise,
    apply_telephony_band,
)
from services.datasets.augment.settings import STAND_IN_LABEL
from voiceloop_common import ValidationError


def test_line_noise_is_deterministic_and_bounded() -> None:
    samples = tuple(range(-200, 200, 25))
    first = add_line_noise(samples, seed=5, level=10, profile="line_hiss")
    assert first == add_line_noise(samples, seed=5, level=10, profile="line_hiss")
    assert first != add_line_noise(samples, seed=6, level=10, profile="line_hiss")
    assert len(first) == len(samples)
    assert all(-32768 <= value <= 32767 for value in first)
    assert add_line_noise(samples, seed=5, level=0) == samples
    crackle = add_line_noise(samples, seed=5, level=10, profile="line_crackle")
    assert crackle == add_line_noise(samples, seed=5, level=10, profile="line_crackle")


def test_telephony_band_preserves_native_rate_and_count() -> None:
    samples = tuple((i % 41) * 17 - 340 for i in range(320))
    band = TelephonyBand(low_hz=300, high_hz=3400, sample_rate_hz=8000)
    first = apply_telephony_band(samples, band=band)
    assert first == apply_telephony_band(samples, band=band)
    assert len(first) == len(samples)
    assert all(-32768 <= value <= 32767 for value in first)
    params = band.params()
    assert params["sample_rate_hz"] == 8000
    assert params["low_hz"] == 300
    assert params["high_hz"] == 3400
    assert params["label"] == STAND_IN_LABEL


def test_transform_params_and_validation_fail_closed() -> None:
    profile = NoiseProfile(name="line_hiss", level=8)
    params = profile.params(seed=12)
    assert params == {"profile": "line_hiss", "level": 8, "seed": 12, "label": STAND_IN_LABEL}
    with pytest.raises(ValidationError):
        NoiseProfile(name="real_audio", level=8)
    with pytest.raises(ValidationError):
        NoiseProfile(level=101)
    with pytest.raises(ValidationError):
        TelephonyBand(low_hz=3400, high_hz=300)
    with pytest.raises(ValidationError):
        TelephonyBand(low_hz=100, high_hz=5000, sample_rate_hz=8000)
    with pytest.raises(ValidationError):
        add_line_noise((1, 2), seed=1, level=4, profile="unknown")
