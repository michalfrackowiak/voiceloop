"""Deterministic fake TTS variants for repeated failure classes (P3-03.1).

The built-in backend emits a short, deterministic integer carrier derived from a
pseudonymous token hash. It is **not** speech, not a codec and not real audio.
A real TTS engine can be injected later through :class:`TtsBackend` without
changing the manifest shape: manifests carry only digests and transform params.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from voiceloop_common import ValidationError

from .settings import STAND_IN_LABEL, AugmentSettings

_CALL_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
NAME_TOKEN_CLASSES: frozenset[str] = frozenset({"misheard_surname", "new_surname_tokens"})
TELEPHONY_CLASSES: frozenset[str] = frozenset({"misheard_guests"})
NOISE_CLASSES: frozenset[str] = frozenset({"line_noise", "misheard_guests"})
_INT16_MIN = -32768
_INT16_MAX = 32767


def _digest(text: str) -> str:
    """Return a stable SHA-256 hex digest of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clamp_int16(value: int) -> int:
    """Clamp an integer to the signed 16-bit range used by the stand-in."""
    return max(_INT16_MIN, min(_INT16_MAX, value))


@dataclass(frozen=True, slots=True)
class AudioStandIn:
    """In-memory deterministic samples explicitly labelled as not real audio."""

    samples: tuple[int, ...]
    sample_rate_hz: int
    duration_ms: int
    label: str = STAND_IN_LABEL

    def __post_init__(self) -> None:
        """Reject empty, non-integer or out-of-range sample sequences."""
        if not self.samples:
            raise ValidationError("audio stand-in must contain at least one sample")
        if self.sample_rate_hz <= 0 or self.duration_ms <= 0:
            raise ValidationError("audio stand-in needs a positive rate and duration")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in self.samples):
            raise ValidationError("audio stand-in samples must be integers")
        if any(value < _INT16_MIN or value > _INT16_MAX for value in self.samples):
            raise ValidationError("audio stand-in samples must fit signed 16-bit")

    @property
    def digest(self) -> str:
        """Return a SHA-256 digest of the native-rate samples, never the bytes."""
        payload = struct.pack("<" + "h" * len(self.samples), *self.samples)
        return hashlib.sha256(f"{self.sample_rate_hz}|".encode() + payload).hexdigest()

    def metadata(self) -> dict[str, int | str]:
        """Return manifest-safe metadata: no samples or audio bytes are included."""
        return {
            "digest": self.digest,
            "sample_count": len(self.samples),
            "sample_rate_hz": self.sample_rate_hz,
            "duration_ms": self.duration_ms,
            "label": self.label,
        }


@dataclass(frozen=True, slots=True)
class VariantSpec:
    """One planned synthetic variant tied to a source call and failure class."""

    variant_id: str
    failure_class: str
    source_call_id: str
    token: str
    sample_rate_hz: int
    duration_ms: int
    seed: int
    telephony: bool
    noisy: bool

    def __post_init__(self) -> None:
        """Reject malformed lineage or pseudonymous token values."""
        if not _CALL_ID_RE.fullmatch(self.source_call_id):
            raise ValidationError("source call IDs must be short pseudonymous identifiers")
        if not _TOKEN_RE.fullmatch(self.token):
            raise ValidationError("variant tokens must be pseudonymous identifiers")
        if self.sample_rate_hz <= 0 or self.duration_ms <= 0 or self.seed < 0:
            raise ValidationError("variant rate, duration and seed must be valid")

    @property
    def token_digest(self) -> str:
        """Return the digest of the pseudonymous token used for this variant."""
        return _digest(self.token)

    def as_manifest_fields(self) -> dict[str, object]:
        """Return the PII-free fields contributed to a manifest record."""
        return {
            "variant_id": self.variant_id,
            "failure_class": self.failure_class,
            "source_call_id": self.source_call_id,
            "token_digest": self.token_digest,
            "telephony": self.telephony,
            "noisy": self.noisy,
        }


@runtime_checkable
class TtsBackend(Protocol):
    """Seam for a TTS backend; the default implementation is a deterministic fake."""

    def synthesize(
        self,
        *,
        token: str,
        sample_rate_hz: int,
        duration_ms: int,
        seed: int,
    ) -> tuple[int, ...]:
        """Return native-rate signed 16-bit samples for one pseudonymous token."""
        ...

    def params(self, *, spec: VariantSpec) -> dict[str, int | str]:
        """Return manifest-safe params; implementors must never emit PII."""
        ...


class DeterministicFakeTts:
    """Deterministic tone-carrier stand-in; produces no intelligible speech."""

    def synthesize(
        self,
        *,
        token: str,
        sample_rate_hz: int,
        duration_ms: int,
        seed: int,
    ) -> tuple[int, ...]:
        """Generate a seeded integer carrier whose frequency depends on the token.

        Args:
            token: Pseudonymous token ID; never a name, email or phone number.
            sample_rate_hz: Native sample rate; 8 kHz is never resampled.
            duration_ms: Requested stand-in duration.
            seed: Deterministic seed supplied by the variant plan.

        Returns:
            A tuple of signed 16-bit stand-in samples.

        Raises:
            ValidationError: If the parameters cannot produce a non-empty signal.

        """
        if sample_rate_hz <= 0 or duration_ms <= 0 or seed < 0:
            raise ValidationError("fake TTS needs a positive rate/duration and a non-negative seed")
        if not _TOKEN_RE.fullmatch(token):
            raise ValidationError("fake TTS accepts only pseudonymous tokens")
        count = sample_rate_hz * duration_ms // 1000
        if count <= 0:
            raise ValidationError("requested duration is shorter than one sample")
        token_value = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:4], "big")
        frequency_hz = 180 + token_value % 220
        amplitude = 600 + (token_value // 256) % 400
        rng_seed = int.from_bytes(hashlib.sha256(f"{token}|{seed}".encode()).digest()[:8], "big")
        # A local RNG keeps results independent of global random state and call order.
        rng = random.Random(rng_seed)
        return tuple(
            _clamp_int16(
                int(amplitude * math.sin(2.0 * math.pi * frequency_hz * i / sample_rate_hz))
                + rng.randint(-12, 12)
            )
            for i in range(count)
        )

    def params(
        self,
        *,
        spec: VariantSpec,
    ) -> dict[str, int | str]:
        """Return manifest-safe TTS transform parameters (no text or byte values)."""
        return {
            "token_digest": spec.token_digest,
            "sample_rate_hz": spec.sample_rate_hz,
            "duration_ms": spec.duration_ms,
            "seed": spec.seed,
            "label": STAND_IN_LABEL,
        }


def build_variant_specs(
    source_call_ids: Sequence[str],
    settings: AugmentSettings,
    *,
    failure_classes: Sequence[str] | None = None,
    name_tokens: Sequence[str] | None = None,
) -> tuple[VariantSpec, ...]:
    """Plan one TTS stand-in per eligible source call and failure class.

    Args:
        source_call_ids: Call IDs from a read-only P2-05 manifest, already
            filtered against golden and regression exclusion lists.
        settings: Typed augmentation settings.
        failure_classes: Override for the repeated failure classes.
        name_tokens: Override for pseudonymous new-name tokens.

    Returns:
        Deterministically ordered variant specs with stable hashed IDs.

    Raises:
        ValidationError: If calls, classes or tokens are malformed or PII-like.

    """
    from services.datasets.filters import detect_pii

    classes = tuple(failure_classes) if failure_classes is not None else settings.failure_classes()
    tokens = tuple(name_tokens) if name_tokens is not None else settings.name_tokens()
    if not classes or not tokens:
        raise ValidationError("at least one failure class and one name token are required")
    if len(set(classes)) != len(classes) or len(set(tokens)) != len(tokens):
        raise ValidationError("failure classes and name tokens must be unique")

    for call_id in source_call_ids:
        if not isinstance(call_id, str) or not _CALL_ID_RE.fullmatch(call_id):
            raise ValidationError("source call IDs must be short pseudonymous identifiers")
    for token in tokens:
        if not _TOKEN_RE.fullmatch(token) or detect_pii(token):
            raise ValidationError("name tokens must be pseudonymous and PII-free")
    for failure_class in classes:
        if not _TOKEN_RE.fullmatch(failure_class) or detect_pii(failure_class):
            raise ValidationError("failure classes must be pseudonymous and PII-free")

    calls = tuple(sorted(set(source_call_ids)))
    specs: list[VariantSpec] = []
    for call_index, call_id in enumerate(calls):
        for class_index, failure_class in enumerate(classes):
            if failure_class in NAME_TOKEN_CLASSES:
                token = tokens[(call_index + class_index) % len(tokens)]
            else:
                token = f"class-{_digest(failure_class)[:10]}"
            seed = settings.seed + (
                int(_digest(f"{call_id}|{failure_class}|{token}")[:8], 16) & 0xFFFF
            )
            variant_id = f"syn-{_digest(f'{call_id}|{failure_class}|{token}|{seed}')[:12]}"
            telephony = failure_class in TELEPHONY_CLASSES
            specs.append(
                VariantSpec(
                    variant_id=variant_id,
                    failure_class=failure_class,
                    source_call_id=call_id,
                    token=token,
                    sample_rate_hz=(
                        settings.telephony_sample_rate_hz if telephony else settings.sample_rate_hz
                    ),
                    duration_ms=settings.duration_ms,
                    seed=seed,
                    telephony=telephony,
                    noisy=failure_class in NOISE_CLASSES,
                )
            )
    return tuple(specs)
