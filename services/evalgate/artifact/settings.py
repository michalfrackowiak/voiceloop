"""Typed environment-backed settings for P3-04 artifact safety tests.

Every field reads an environment variable prefixed with ``VOICELOOP_ARTIFACT_``.
Inputs (fixtures and the sealed synthetic golden stand-in) are read-only; the
only paths this package writes are the caller-supplied report output and the
content-addressed fake-artifact store.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field

from voiceloop_common import BaseSettings, ConfigError, settings_config

ArtifactBehaviour = Literal["healthy", "empty", "truncated", "hallucinating", "forgetting"]

EXPECTED_FIXTURE_SHA256 = "aca23a303954d035245a97f78319df3694d63611b717a2ff2131bc2e0a4453fd"
EXPECTED_GOLDEN_CONTENT_HASH = "662cdaf43ee2d5fde03faf5e69fde196114d4f312406d94c00bf53714d03da8e"
DEFAULT_HOTWORDS: tuple[str, ...] = ("deluxe", "suite")
DEFAULT_INITIAL_PROMPT = "Hotel booking."
DEFAULT_HALLUCINATION_TEXT = "Thank you for calling."
_LANGUAGE_CODE = re.compile(r"[a-z]{2}(?:-[a-z]{2,4})?")


class ArtifactSettings(BaseSettings):
    """Configuration for exact-artifact decoding and failure-injection probes.

    Defaults are deliberately synthetic: the committed tone/noise/silence
    fixtures and a fake content-addressed artifact stand in for the real
    converted/quantised artifact and production decoding settings
    (``context.md`` section 2 invariants 2 and 3).
    """

    model_config = settings_config("VOICELOOP_ARTIFACT_")

    fixtures_path: str = "tests/fixtures/turns.jsonl"
    golden_artifact_dir: str = "services/datasets/golden/artifacts"
    golden_version: str = "golden.v1"
    expected_fixture_sha256: str = EXPECTED_FIXTURE_SHA256
    expected_golden_content_hash: str = EXPECTED_GOLDEN_CONTENT_HASH
    output_path: str = "reports/evalgate/artifact/stt_report.json"
    artifact_store_dir: str = "reports/evalgate/artifact/artifacts"

    # Stand-in for the trainer/registry checkpoint URI. It is never evaluated:
    # the converter always substitutes the content-addressed deployable artifact.
    checkpoint_uri: str = "fake://trainer/checkpoint"
    runtime_label: str = "fake-ctranslate2"
    artifact_behaviour: ArtifactBehaviour = "healthy"

    # Exact decoding settings recorded in every report and varied by the probes.
    hotwords: list[str] = Field(default_factory=lambda: list(DEFAULT_HOTWORDS))
    initial_prompt: str = DEFAULT_INITIAL_PROMPT
    condition_on_previous_text: bool = False
    language: str = "en"

    # Deterministic synthetic failure injection. No human labels or real audio.
    hallucination_text: str = DEFAULT_HALLUCINATION_TEXT
    truncation_ratio: float = Field(default=0.5, gt=0, lt=1)
    served_languages_csv: str = "en"
    other_language_pattern: str = r"[^\x00-\x7f]"
    max_error_details: int = Field(default=25, ge=1)

    bootstrap_level: float = Field(default=0.95, gt=0, lt=1)
    bootstrap_iterations: int = Field(default=1000, ge=1)
    bootstrap_seed: int = 42

    def served_languages(self) -> tuple[str, ...]:
        """Parse and validate the configured served language codes.

        Returns:
            Lowercase language codes in configuration order; ``en`` is the
            roadmap default until the human owner states otherwise.

        Raises:
            ConfigError: If no code is present or a code is not a language tag.

        """
        codes = tuple(
            code.strip().lower() for code in self.served_languages_csv.split(",") if code.strip()
        )
        if not codes:
            raise ConfigError("at least one served language code is required")
        invalid = [code for code in codes if not _LANGUAGE_CODE.fullmatch(code)]
        if invalid:
            raise ConfigError(f"invalid served language codes: {', '.join(invalid)}")
        return codes

    def non_english_languages(self) -> tuple[str, ...]:
        """Return served languages other than the report's base language."""
        base = self.language.strip().lower()
        return tuple(code for code in self.served_languages() if code != base)
