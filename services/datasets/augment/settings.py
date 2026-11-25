"""Typed environment-backed settings for synthetic augmentation (P3-03).

Every field reads ``VOICELOOP_AUGMENT_*``. All generated data is a deterministic
stand-in: no TTS engine, audio codec, network or GPU is used. Inputs are read
only; ``output_path`` is the only path written by the CLI.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from voiceloop_common import BaseSettings, ConfigError, settings_config

# Provenance anchors from the committed synthetic rehearsal. Both are checked
# before and after every augmentation run; a change fails closed.
EXPECTED_GOLDEN_CONTENT_HASH = "662cdaf43ee2d5fde03faf5e69fde196114d4f312406d94c00bf53714d03da8e"
EXPECTED_FIXTURE_SHA256 = "aca23a303954d035245a97f78319df3694d63611b717a2ff2131bc2e0a4453fd"

STAND_IN_LABEL = "SYNTHETIC/FIXTURE STAND-IN - NOT REAL AUDIO"
ARTIFACT_TYPE = "voiceloop.augmentation.manifest"
AUGMENTATION_SCHEMA_VERSION = "internal-1.0.0"

# Failure classes observed repeatedly in the synthetic fixtures (context.md §7).
# New name/surname lists are represented by pseudonymous tokens, never by real
# names, and only their digests reach a manifest.
DEFAULT_FAILURE_CLASSES: tuple[str, ...] = (
    "misheard_surname",
    "new_surname_tokens",
    "misheard_guests",
    "line_noise",
)
DEFAULT_NAME_TOKENS: tuple[str, ...] = (
    "surname-token-01",
    "surname-token-02",
    "surname-token-03",
)
ALLOWED_TRANSFORMS: frozenset[str] = frozenset({"fake_tts", "line_noise", "telephony_band"})


def parse_csv(value: str, *, field: str) -> tuple[str, ...]:
    """Parse a non-empty comma-separated setting value into distinct items.

    Args:
        value: Raw environment/CLI value.
        field: Setting name used in error messages.

    Returns:
        Stripped, non-empty, distinct values in input order.

    Raises:
        ConfigError: If the list is empty or contains duplicates.

    """
    items = tuple(part.strip() for part in value.split(",") if part.strip())
    if not items:
        raise ConfigError(f"{field} must contain at least one value")
    if len(set(items)) != len(items):
        raise ConfigError(f"{field} must not contain duplicates")
    return items


class AugmentSettings(BaseSettings):
    """Configuration for deterministic synthetic augmentation."""

    model_config = settings_config("VOICELOOP_AUGMENT_")

    dataset_manifest_path: str = Field(
        default="",
        description="Read-only P2-05 dataset_manifest.schema.json 1.0.0 input path.",
    )
    output_path: str = "reports/datasets/augment/manifest.json"
    fixtures_path: str = "tests/fixtures/turns.jsonl"
    golden_manifest_path: str = "services/datasets/golden/artifacts/golden.v1.json"
    expected_golden_content_hash: str = EXPECTED_GOLDEN_CONTENT_HASH
    expected_fixture_sha256: str = EXPECTED_FIXTURE_SHA256

    failure_classes_csv: str = ",".join(DEFAULT_FAILURE_CLASSES)
    name_tokens_csv: str = ",".join(DEFAULT_NAME_TOKENS)
    seed: int = 17
    duration_ms: int = Field(default=240, ge=80, le=2000)
    sample_rate_hz: int = Field(default=16000, ge=8000, le=48000)
    telephony_sample_rate_hz: int = Field(default=8000, ge=8000, le=48000)
    noise_level: int = Field(default=8, ge=0, le=100)
    telephony_low_hz: int = Field(default=300, ge=1)
    telephony_high_hz: int = Field(default=3400, ge=2)

    @model_validator(mode="after")
    def _telephony_band_is_sane(self) -> AugmentSettings:
        """Reject a band that cannot fit inside the native telephony rate."""
        if self.telephony_low_hz >= self.telephony_high_hz:
            raise ValueError("telephony_low_hz must be below telephony_high_hz")
        if self.telephony_high_hz * 2 > self.telephony_sample_rate_hz:
            raise ValueError("telephony_high_hz exceeds the Nyquist rate")
        return self

    def failure_classes(self) -> tuple[str, ...]:
        """Return the configured repeated-failure classes."""
        return parse_csv(self.failure_classes_csv, field="failure_classes_csv")

    def name_tokens(self) -> tuple[str, ...]:
        """Return pseudonymous name-list tokens (never real names)."""
        return parse_csv(self.name_tokens_csv, field="name_tokens_csv")
