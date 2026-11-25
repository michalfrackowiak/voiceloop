"""Deterministic fake-TTS variant planning tests for P3-03.1."""

from __future__ import annotations

from typing import Any

import pytest

from services.datasets.augment.cli import build_augmentation
from services.datasets.augment.settings import STAND_IN_LABEL, AugmentSettings
from services.datasets.augment.variants import (
    AudioStandIn,
    DeterministicFakeTts,
    VariantSpec,
    build_variant_specs,
)
from voiceloop_common import ValidationError


def test_fake_tts_is_deterministic_and_not_real_audio(settings: AugmentSettings) -> None:
    tts = DeterministicFakeTts()
    first = tts.synthesize(
        token="surname-token-01",
        sample_rate_hz=settings.sample_rate_hz,
        duration_ms=settings.duration_ms,
        seed=7,
    )
    second = tts.synthesize(
        token="surname-token-01",
        sample_rate_hz=settings.sample_rate_hz,
        duration_ms=settings.duration_ms,
        seed=7,
    )
    other = tts.synthesize(
        token="surname-token-02",
        sample_rate_hz=settings.sample_rate_hz,
        duration_ms=settings.duration_ms,
        seed=7,
    )
    assert first == second
    assert first != other
    assert len(first) == settings.sample_rate_hz * settings.duration_ms // 1000
    assert all(isinstance(sample, int) and -32768 <= sample <= 32767 for sample in first)
    stand_in = AudioStandIn(first, settings.sample_rate_hz, settings.duration_ms)
    assert stand_in.label == STAND_IN_LABEL
    assert stand_in.metadata()["digest"] == stand_in.digest
    assert "samples" not in stand_in.metadata()


def test_fake_tts_rejects_bad_inputs() -> None:
    tts = DeterministicFakeTts()
    with pytest.raises(ValidationError):
        tts.synthesize(token="Ada Fixture", sample_rate_hz=8000, duration_ms=80, seed=1)
    with pytest.raises(ValidationError):
        tts.synthesize(token="token-1", sample_rate_hz=0, duration_ms=80, seed=1)
    with pytest.raises(ValidationError):
        AudioStandIn((40000,), 8000, 80)


def test_plan_covers_failure_classes_without_pii(
    settings: AugmentSettings,
    source_manifest: dict[str, Any],
) -> None:
    calls = ("call-0003", "call-0004", "call-0005")
    specs = build_variant_specs(calls, settings)
    assert specs == build_variant_specs(calls, settings)
    assert {spec.source_call_id for spec in specs} == set(calls)
    assert {spec.failure_class for spec in specs} == set(settings.failure_classes())
    assert len({spec.variant_id for spec in specs}) == len(specs) == 12
    for spec in specs:
        fields = spec.as_manifest_fields()
        assert len(spec.token_digest) == 64
        assert spec.token not in str(fields)
        assert spec.sample_rate_hz in (8000, settings.sample_rate_hz)
    telephony = [spec for spec in specs if spec.failure_class == "misheard_guests"]
    assert telephony and all(spec.telephony for spec in telephony)
    assert all(not spec.telephony for spec in specs if spec not in telephony)
    assert all(spec.noisy for spec in specs if spec.failure_class == "line_noise")
    assert source_manifest["source_call_ids"]  # source stays untouched


def test_plan_rejects_pii_like_or_malformed_inputs(settings: AugmentSettings) -> None:
    with pytest.raises(ValidationError):
        build_variant_specs(("not a call id!",), settings)
    with pytest.raises(ValidationError):
        build_variant_specs(("call-0001",), settings, name_tokens=("Ada Fixture",))
    with pytest.raises(ValidationError):
        build_variant_specs(("call-0001",), settings, failure_classes=("Name is Ada",))


class CountingTts:
    """Injected fake backend that records synthesis calls and returns a constant."""

    def __init__(self) -> None:
        """Start with no recorded calls."""
        self.calls: list[dict[str, int | str]] = []

    def synthesize(
        self,
        *,
        token: str,
        sample_rate_hz: int,
        duration_ms: int,
        seed: int,
    ) -> tuple[int, ...]:
        """Return deterministic constant samples and record the parameters."""
        self.calls.append(
            {"token": token, "rate": sample_rate_hz, "duration": duration_ms, "seed": seed}
        )
        return (7,) * (sample_rate_hz * duration_ms // 1000)

    def params(self, *, spec: VariantSpec) -> dict[str, int | str]:
        """Return a pseudonymous digest rather than the raw token."""
        return {"token_digest": spec.token_digest, "seed": spec.seed}


def test_injected_backend_is_used_without_leaking_tokens(
    settings: AugmentSettings,
    source_manifest: dict[str, Any],
) -> None:
    backend = CountingTts()
    run = build_augmentation(
        settings,
        tts=backend,
        dataset_manifest=source_manifest,
        golden_call_ids=("call-0001",),
        regression_call_ids=(),
    )
    assert len(backend.calls) == len(run.variants)
    assert any("surname-token" in str(call) for call in backend.calls)
    assert all(
        "surname-token" not in str(record.to_dict())
        for variant in run.variants
        for record in variant.transforms
    )
