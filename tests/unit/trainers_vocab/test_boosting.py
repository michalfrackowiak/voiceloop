"""Phrase-boosting config generator and synthetic runtime seam tests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from services.trainers.vocab import (
    BoostConfig,
    FakeSttRuntime,
    SttRuntime,
    VocabSettings,
    apply_boosting,
    generate_boosting_config,
    write_boosting_config,
)


def test_generated_config_is_synthetic_and_contains_fixture_provenance(
    settings: VocabSettings,
) -> None:
    config = generate_boosting_config(settings)
    assert config.schema_version == "voiceloop.vocab-boost.v1"
    assert config.synthetic is True
    assert "SYNTHETIC/FIXTURE STAND-IN" in config.warning
    assert config.runtime == "faster-whisper-ctranslate2"
    phrases = {hotword.phrase for hotword in config.hotwords}
    assert {"Mr. Smith", "four guests", "deluxe room", "standard room", "suite"} <= phrases
    cases = {item.source_case_id for item in config.substitutions}
    assert cases == {"misheard_surname", "misheard_guests"}
    assert all(item.source_call_id.startswith("call-") for item in config.substitutions)


def test_config_json_round_trip_and_max_hotwords(tmp_path: Path, settings: VocabSettings) -> None:
    low_limit = settings.model_copy(update={"max_hotwords": 3})
    config = generate_boosting_config(low_limit)
    assert len(config.hotwords) == 3
    destination = write_boosting_config(config, tmp_path / "nested" / "boost.json")
    decoded = json.loads(destination.read_text(encoding="utf-8"))
    assert decoded == config.to_dict()
    assert decoded["synthetic"] is True
    assert decoded["max_hotwords"] == 3


class _RecordingRuntime:
    """Injected runtime stub that records what config it received."""

    def __init__(self) -> None:
        self.calls: list[BoostConfig] = []

    def transcribe(
        self,
        transcript: str,
        words: Sequence[Mapping[str, Any]],
        config: BoostConfig,
    ) -> str:
        """Record the config and return the transcript unchanged."""
        self.calls.append(config)
        return transcript


def test_injected_runtime_seam_is_used(settings: VocabSettings) -> None:
    config = generate_boosting_config(settings)
    runtime = _RecordingRuntime()
    assert isinstance(runtime, SttRuntime)
    assert apply_boosting("unchanged", [], config, runtime) == "unchanged"
    assert runtime.calls == [config]


def test_fake_runtime_only_corrects_low_confidence_fixture_phrases(
    settings: VocabSettings,
) -> None:
    config = generate_boosting_config(settings)
    low = [{"w": "Mr.", "conf": 0.95}, {"w": "Smythe.", "conf": 0.41}]
    high = [{"w": "Mr.", "conf": 0.95}, {"w": "Smythe.", "conf": 0.95}]
    runtime = FakeSttRuntime()
    assert runtime.transcribe("Mr. Smythe.", low, config) == "Mr. Smith."
    assert runtime.transcribe("Mr. Smythe.", high, config) == "Mr. Smythe."
    assert runtime.transcribe("Ada Fixture.", low, config) == "Ada Fixture."


def test_fake_runtime_does_not_rewrite_high_confidence_guest_count(
    settings: VocabSettings,
) -> None:
    config = generate_boosting_config(settings)
    low = [{"w": "two", "conf": 0.41}, {"w": "guests.", "conf": 0.94}]
    high = [{"w": "two", "conf": 0.96}, {"w": "guests.", "conf": 0.94}]
    runtime = FakeSttRuntime()
    assert runtime.transcribe("Make it two guests.", low, config) == "Make it four guests."
    assert runtime.transcribe("Check in January for two guests.", high, config) == (
        "Check in January for two guests."
    )
