"""Candidate runner protocol and deterministic fake candidates.

:class:`CandidateRunner` is the extension seam for base-model swaps. A real
Whisper variant, Parakeet/NeMo model, Qwen3-ASR model or any other STT runtime
can implement ``predict(row, event) -> Prediction`` against its deployed-form
artifact and be registered with :func:`register_candidate`. This module
deliberately imports none of those runtimes: no GPU, network or heavy ML
dependency is needed, and the built-ins are deterministic fixture stand-ins.

``row`` is a read-only labelled fixture record (including ``audio_path``) and
``event`` is the committed synthetic turn event. The returned
:class:`services.evalgate.metrics.Prediction` is scored by the P1-07 harness
primitives exactly as any real candidate output would be.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from services.evalgate.harness import predict
from services.evalgate.metrics import Prediction
from voiceloop_common import ConfigError

IDENTITY_NAME = "identity"
DEGRADED_NAME = "degraded"


@runtime_checkable
class CandidateRunner(Protocol):
    """Minimal interface every benchmark candidate must implement."""

    @property
    def name(self) -> str:
        """Stable candidate name used in the comparison artifact."""
        ...

    @property
    def artifact_uri(self) -> str:
        """Stand-in URI for the evaluated artifact (fake for built-ins)."""
        ...

    def predict(self, row: Mapping[str, Any], event: Mapping[str, Any]) -> Prediction:
        """Return one transcript and parser-slot prediction for a labelled turn.

        Real adapters should read ``row["audio_path"]`` at the artifact's
        native sample rate and use their exact production decoding settings.
        Synthetic fakes replay or deterministically perturb committed fixtures.
        """
        ...


@dataclass(frozen=True)
class IdentityRunner:
    """Fake candidate that replays recorded STT text and parser slots."""

    name: str = IDENTITY_NAME
    artifact_uri: str = "fake://benchmark/identity"

    def predict(self, row: Mapping[str, Any], event: Mapping[str, Any]) -> Prediction:
        """Replay the fixture event using the read-only P1-07 identity fake."""
        return predict("identity", row, event)


@dataclass(frozen=True)
class DegradedRunner:
    """Deterministically worse fake: one hallucinated number and a lost slot.

    The transcript gains a trailing ``zero`` token (an entity insertion on
    every turn) and the ``guests`` slot is removed when present. This exercises
    all five metrics without pretending to model any real ASR error profile.
    """

    name: str = DEGRADED_NAME
    artifact_uri: str = "fake://benchmark/degraded"

    def predict(self, row: Mapping[str, Any], event: Mapping[str, Any]) -> Prediction:
        """Return a deterministic degraded version of the identity prediction."""
        baseline = predict("identity", row, event)
        transcript = f"{baseline.transcript} zero".strip()
        slots = {key: value for key, value in baseline.slots.items() if key != "guests"}
        return Prediction(transcript=transcript, slots=slots)


_BUILTINS: dict[str, CandidateRunner] = {
    IdentityRunner().name: IdentityRunner(),
    DegradedRunner().name: DegradedRunner(),
}
_EXTRAS: dict[str, CandidateRunner] = {}


def register_candidate(runner: CandidateRunner) -> None:
    """Register a custom candidate runner under its ``name``.

    Args:
        runner: Any object implementing :class:`CandidateRunner`.

    Raises:
        ConfigError: If the name is blank or collides with a built-in fake.

    """
    name = runner.name.strip()
    if not name:
        raise ConfigError("candidate runner name must not be blank")
    if name in _BUILTINS:
        raise ConfigError(f"candidate name {name!r} is reserved for a built-in fake")
    _EXTRAS[name] = runner


def get_candidate(name: str) -> CandidateRunner:
    """Resolve a candidate by name from registered and built-in runners."""
    if name in _EXTRAS:
        return _EXTRAS[name]
    if name in _BUILTINS:
        return _BUILTINS[name]
    available = ", ".join(available_candidates())
    raise ConfigError(f"unknown candidate runner: {name!r}; available: {available}")


def available_candidates() -> tuple[str, ...]:
    """Return all registered candidate names in stable sorted order."""
    return tuple(sorted(set(_BUILTINS) | set(_EXTRAS)))
