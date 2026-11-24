"""Typed, environment-backed settings for the evaluation harness."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from voiceloop_common import BaseSettings, settings_config

ArtifactKind = Literal["identity", "oracle", "no_slots"]
EvalSetName = Literal["golden", "regression", "general_capability"]


def _default_eval_sets() -> list[EvalSetName]:
    """Return the default synthetic-fixture evaluation set."""
    return ["general_capability"]


class EvalgateSettings(BaseSettings):
    """Configuration for P1-07; every field reads ``VOICELOOP_EVALGATE_*``.

    Fixtures are read-only. ``output_path`` and all other paths are resolved by
    the caller against the process working directory (normally the repo root).
    """

    model_config = settings_config("VOICELOOP_EVALGATE_")

    fixtures_path: str = "tests/fixtures/turns.jsonl"
    output_path: str = "reports/evalgate/report.json"
    artifact_uri: str = "fake://evalgate/identity"

    # In-memory fakes: identity replays the recorded STT/parser output, oracle
    # uses the synthetic labels as an upper bound, no_slots emits nothing.
    champion_artifact: ArtifactKind = "identity"
    challenger_artifact: ArtifactKind = "identity"

    bootstrap_level: float = Field(default=0.95, gt=0, lt=1)
    bootstrap_iterations: int = Field(default=1000, ge=1)
    bootstrap_seed: int = 0

    eval_sets: list[EvalSetName] = Field(default_factory=_default_eval_sets)

    # Exact decoding settings recorded in the report (context.md section 10).
    hotwords: list[str] = Field(default_factory=list)
    initial_prompt: str = ""
    condition_on_previous_text: bool = False
    language: str = "en"

    # Regex replacements keyed by names/dates/numbers; parsed from a JSON env value.
    entity_pattern_overrides: dict[str, str] = Field(default_factory=dict)
    other_language_pattern: str = r"[^\x00-\x7f]"
