"""Typed, environment-backed settings for the offline vocabulary tier."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from voiceloop_common import BaseSettings, settings_config

SttRuntimeName = Literal["faster-whisper-ctranslate2", "injected"]

# Content hash of the sealed synthetic v1 stand-in recorded in P1-06/P1-08.
# It is checked before and after every regression run; a change is an error.
EXPECTED_GOLDEN_HASH = "662cdaf43ee2d5fde03faf5e69fde196114d4f312406d94c00bf53714d03da8e"


class VocabSettings(BaseSettings):
    """Configuration for P2-08; every field reads ``VOICELOOP_VOCAB_*``.

    Paths are repo-relative and all inputs are read-only. The runtime name is a
    configuration label for the emitted JSON config, never a model or runtime
    that this package imports. All improvements are SYNTHETIC/FIXTURE STAND-IN
    results from in-memory fakes.
    """

    model_config = settings_config("VOICELOOP_VOCAB_")

    fixtures_path: str = "tests/fixtures/turns.jsonl"
    golden_artifact_dir: str = "services/datasets/golden/artifacts"
    golden_version: str = "golden.v1"
    expected_golden_hash: str = EXPECTED_GOLDEN_HASH

    # Roadmap default for the unanswered STT-runtime question. Only emitted as
    # metadata for the injected runtime seam; no runtime is constructed here.
    runtime: SttRuntimeName = "faster-whisper-ctranslate2"
    language: str = "en"

    enable_aliases: bool = True
    enable_normalisation: bool = True
    enable_boosting: bool = True
    rewrite_transcript: bool = False

    max_hotwords: int = Field(default=64, ge=1, le=1024)
    hotword_boost: float = Field(default=5.0, ge=-10.0, le=20.0)
    initial_prompt: str = ""
    condition_on_previous_text: bool = False
    low_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # Optional manual extensions to fixture-derived tables. Keys are surfaces
    # and values are canonical forms; both are case-folded on comparison.
    name_alias_overrides: dict[str, str] = Field(default_factory=dict)
    room_alias_overrides: dict[str, str] = Field(default_factory=dict)

    output_path: str = "reports/vocab/boosting-config.json"
