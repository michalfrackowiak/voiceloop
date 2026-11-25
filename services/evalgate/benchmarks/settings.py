"""Typed environment-backed settings for the P3-05 benchmark harness."""

from __future__ import annotations

from pydantic import Field

from voiceloop_common import BaseSettings, ConfigError, settings_config

EXPECTED_GOLDEN_CONTENT_HASH = "662cdaf43ee2d5fde03faf5e69fde196114d4f312406d94c00bf53714d03da8e"
DEFAULT_CANDIDATES_CSV = "identity,degraded"


class BenchmarkSettings(BaseSettings):
    """Configuration for synthetic candidate comparisons.

    Every field reads an environment variable prefixed with ``VOICELOOP_BENCH_``.
    Inputs are read-only; ``output_path`` is the only path this harness writes.
    """

    model_config = settings_config("VOICELOOP_BENCH_")

    fixtures_path: str = "tests/fixtures/turns.jsonl"
    golden_artifact_dir: str = "services/datasets/golden/artifacts"
    golden_version: str = "golden.v1"
    expected_golden_content_hash: str = EXPECTED_GOLDEN_CONTENT_HASH
    output_path: str = "reports/evalgate/benchmarks/comparison.json"

    candidates_csv: str = DEFAULT_CANDIDATES_CSV
    bootstrap_level: float = Field(default=0.95, gt=0, lt=1)
    bootstrap_iterations: int = Field(default=1000, ge=1)
    bootstrap_seed: int = 42

    # Placeholder decoding settings recorded in every per-candidate report.
    # Real adapters must be evaluated with their exact production decoding
    # settings (context.md section 10); nothing here claims to be production.
    hotwords: list[str] = Field(default_factory=list)
    initial_prompt: str = ""
    condition_on_previous_text: bool = False
    language: str = "en"

    entity_pattern_overrides: dict[str, str] = Field(default_factory=dict)
    other_language_pattern: str = r"[^\x00-\x7f]"

    def candidate_names(self) -> tuple[str, ...]:
        """Parse ``candidates_csv`` into distinct, non-empty candidate names.

        Returns:
            Candidate names in CLI order; the first is candidate A and the
            second is candidate B for the ``A - B`` delta report.

        Raises:
            ConfigError: If the value is empty or contains duplicates.

        """
        names = tuple(name.strip() for name in self.candidates_csv.split(",") if name.strip())
        if not names:
            raise ConfigError("at least one candidate name is required")
        if len(set(names)) != len(names):
            raise ConfigError("candidate names must be unique")
        return names

    def decoding_settings(self) -> dict[str, object]:
        """Return the placeholder decoding settings stored in reports."""
        return {
            "hotwords": list(self.hotwords),
            "initial_prompt": self.initial_prompt,
            "condition_on_previous_text": self.condition_on_previous_text,
            "language": self.language,
        }
