"""Deterministic line-noise and telephony-band stand-in transforms (P3-03.2).

These are integer-only approximations for rehearsal, **not** audio codecs,
filters, resamplers or real telephony simulation. Native 8 kHz audio is never
resampled (context.md §16). Every output stays in memory as a labelled stand-in.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from voiceloop_common import ValidationError

from .settings import STAND_IN_LABEL

_INT16_MIN = -32768
_INT16_MAX = 32767
_NOISE_PROFILES: frozenset[str] = frozenset({"line_hiss", "line_crackle"})


def _clamp16(value: int) -> int:
    """Clamp an integer to signed 16-bit range without rounding or codecs."""
    return max(_INT16_MIN, min(_INT16_MAX, value))


@dataclass(frozen=True, slots=True)
class NoiseProfile:
    """Parameters for the deterministic fake line-noise transform."""

    name: str = "line_hiss"
    level: int = 8

    def __post_init__(self) -> None:
        """Reject unknown profiles and out-of-range levels."""
        if self.name not in _NOISE_PROFILES:
            raise ValidationError(f"unknown noise profile: {self.name!r}")
        if not 0 <= self.level <= 100:
            raise ValidationError("noise level must be between 0 and 100")

    def params(self, *, seed: int) -> dict[str, int | str]:
        """Return manifest-safe noise params; no samples or PII-bearing text."""
        return {
            "profile": self.name,
            "level": self.level,
            "seed": seed,
            "label": STAND_IN_LABEL,
        }


@dataclass(frozen=True, slots=True)
class TelephonyBand:
    """Parameters for the deterministic fake 300–3400 Hz band transform."""

    low_hz: int = 300
    high_hz: int = 3400
    sample_rate_hz: int = 8000

    def __post_init__(self) -> None:
        """Reject a band that does not fit inside the native sample rate."""
        if self.sample_rate_hz <= 0:
            raise ValidationError("sample rate must be positive")
        if not 0 < self.low_hz < self.high_hz:
            raise ValidationError("telephony band must satisfy 0 < low < high")
        if self.high_hz * 2 > self.sample_rate_hz:
            raise ValidationError("telephony high cutoff exceeds the Nyquist rate")

    def params(self) -> dict[str, int | str]:
        """Return manifest-safe band params; no samples or decoded audio."""
        return {
            "low_hz": self.low_hz,
            "high_hz": self.high_hz,
            "sample_rate_hz": self.sample_rate_hz,
            "label": STAND_IN_LABEL,
        }


def add_line_noise(
    samples: Sequence[int],
    *,
    seed: int,
    level: int,
    profile: str = "line_hiss",
) -> tuple[int, ...]:
    """Add a seeded integer perturbation to an in-memory stand-in signal.

    Args:
        samples: Original signed 16-bit stand-in samples.
        seed: Local RNG seed, derived deterministically by the variant plan.
        level: Maximum absolute perturbation (0–100).
        profile: ``line_hiss`` (uniform) or ``line_crackle`` (sparse spikes).

    Returns:
        The same number of clamped signed 16-bit samples.

    Raises:
        ValidationError: If the profile or level is invalid.

    """
    if profile not in _NOISE_PROFILES:
        raise ValidationError(f"unknown noise profile: {profile!r}")
    if not 0 <= level <= 100 or seed < 0:
        raise ValidationError("noise level and seed must be non-negative and in range")
    rng = random.Random(seed)
    if profile == "line_hiss":
        perturbations = (rng.randint(-level, level) for _ in samples)
    else:
        perturbations = (
            rng.choice((-2 * level, -level, 0, level, 2 * level)) if rng.random() < 0.2 else 0
            for _ in samples
        )
    return tuple(
        _clamp16(sample + noise) for sample, noise in zip(samples, perturbations, strict=True)
    )


def apply_telephony_band(
    samples: Sequence[int],
    *,
    band: TelephonyBand,
) -> tuple[int, ...]:
    """Apply an integer one-pole band stand-in without any audio library.

    This is deliberately a coarse approximation: it uses two fixed-point
    accumulators with coefficients derived from the configured cutoffs to model
    a mid-band emphasis. It is not a biquad, FFT, codec or resampler, and it
    preserves the native sample rate and sample count exactly.

    Args:
        samples: Original signed 16-bit stand-in samples.
        band: Validated band parameters, including the native sample rate.

    Returns:
        The same number of clamped signed 16-bit samples.

    """
    scale = 16
    high_k = max(1, min(scale, band.high_hz * scale // band.sample_rate_hz))
    low_k = max(1, min(scale - 1, band.low_hz * scale // band.sample_rate_hz))
    low_pass = 0
    baseline = 0
    output: list[int] = []
    for sample in samples:
        low_pass = (low_pass * (scale - high_k) + sample * high_k) // scale
        baseline = (baseline * (scale - low_k) + sample * low_k) // scale
        output.append(_clamp16((low_pass - baseline) * 2))
    return tuple(output)
