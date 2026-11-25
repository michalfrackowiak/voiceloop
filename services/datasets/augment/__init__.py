"""Deterministic synthetic augmentation for repeated STT failure classes (P3-03).

Everything here is a **SYNTHETIC/FIXTURE STAND-IN - NOT REAL AUDIO**. No TTS
engine, codec, network or GPU is used by the built-in backends. The sealed golden
stand-in, regression calls and fixture manifests are read-only inputs.
"""

from __future__ import annotations

from .cli import AugmentationRun, build_augmentation, main
from .manifest import (
    AugmentedVariant,
    SourceHashes,
    TransformRecord,
    assert_source_hashes_unchanged,
    build_manifest,
    load_dataset_manifest,
    snapshot_source_hashes,
    validate_manifest,
    write_manifest,
)
from .noise import (
    NoiseProfile,
    TelephonyBand,
    add_line_noise,
    apply_telephony_band,
)
from .settings import (
    ALLOWED_TRANSFORMS,
    ARTIFACT_TYPE,
    AUGMENTATION_SCHEMA_VERSION,
    EXPECTED_FIXTURE_SHA256,
    EXPECTED_GOLDEN_CONTENT_HASH,
    STAND_IN_LABEL,
    AugmentSettings,
)
from .variants import (
    AudioStandIn,
    DeterministicFakeTts,
    TtsBackend,
    VariantSpec,
    build_variant_specs,
)

__all__ = [
    "ALLOWED_TRANSFORMS",
    "ARTIFACT_TYPE",
    "AUGMENTATION_SCHEMA_VERSION",
    "EXPECTED_FIXTURE_SHA256",
    "EXPECTED_GOLDEN_CONTENT_HASH",
    "STAND_IN_LABEL",
    "AugmentationRun",
    "AugmentSettings",
    "AugmentedVariant",
    "AudioStandIn",
    "DeterministicFakeTts",
    "NoiseProfile",
    "SourceHashes",
    "TelephonyBand",
    "TransformRecord",
    "TtsBackend",
    "VariantSpec",
    "add_line_noise",
    "apply_telephony_band",
    "assert_source_hashes_unchanged",
    "build_augmentation",
    "build_manifest",
    "build_variant_specs",
    "load_dataset_manifest",
    "main",
    "snapshot_source_hashes",
    "validate_manifest",
    "write_manifest",
]
