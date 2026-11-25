"""CandidateRunner protocol and deterministic fake tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest

from services.evalgate.benchmarks import runners as runners_module
from services.evalgate.benchmarks.runners import (
    CandidateRunner,
    DegradedRunner,
    IdentityRunner,
    available_candidates,
    get_candidate,
    register_candidate,
)
from services.evalgate.harness import load_event, load_fixture_rows
from services.evalgate.metrics import Prediction
from voiceloop_common import ConfigError


@dataclass
class _CustomRunner:
    """Minimal protocol implementation used to exercise registration."""

    name: str = "custom-test"
    artifact_uri: str = "fake://benchmark/custom-test"

    def predict(self, row: Mapping[str, Any], event: Mapping[str, Any]) -> Prediction:
        """Return a fixed prediction."""
        return Prediction(transcript="custom", slots={})


def test_fakes_are_protocol_compatible_and_deterministic() -> None:
    """Both built-ins implement the protocol and replay deterministic outputs."""
    row = load_fixture_rows("tests/fixtures/turns.jsonl")[1]
    event = load_event(str(row["event_path"]))
    identity = IdentityRunner()
    degraded = DegradedRunner()
    assert isinstance(identity, CandidateRunner)
    assert isinstance(degraded, CandidateRunner)

    first = identity.predict(row, event)
    assert first == identity.predict(row, event)
    assert (
        first.transcript
        == "Check in on the twelfth of December for Ada Fixture, two guests, please."
    )
    assert first.slots == event["parser"]["output"]

    degraded_first = degraded.predict(row, event)
    assert degraded_first == degraded.predict(row, event)
    assert degraded_first.transcript == f"{first.transcript} zero"
    assert "guests" not in degraded_first.slots
    assert {
        key: value for key, value in first.slots.items() if key != "guests"
    } == degraded_first.slots


def test_registry_resolves_builtins_and_custom_runners() -> None:
    """Custom runners are pluggable without importing real model libraries."""
    assert {"identity", "degraded"} <= set(available_candidates())
    custom = _CustomRunner()
    register_candidate(custom)
    try:
        assert get_candidate(custom.name) is custom
        assert custom.name in available_candidates()
        with pytest.raises(ConfigError, match="reserved"):
            register_candidate(_CustomRunner(name="identity"))
    finally:
        runners_module._EXTRAS.pop(custom.name, None)


def test_unknown_candidate_is_reported() -> None:
    """An unregistered name raises a structured config error."""
    with pytest.raises(ConfigError, match="unknown candidate runner"):
        get_candidate("whisper-large-v3")
