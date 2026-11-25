"""Post-swap health probe seam and deterministic fakes.

> [!IMPORTANT]
> **FAKE ONLY.** :class:`FakeHealthProbe` returns scripted results and
> :class:`ServingStateProbe` inspects in-memory serving state. Neither performs
> HTTP requests, GPU inference or model decoding.

A real adapter would call the serving health endpoints (STT readiness and a
parser schema-valid smoke check), compare their version tags with the expected
champion version, and return stable check IDs such as ``stt.version`` or
``parser.adapter``. Raw HTTP bodies, transcripts and guest data must never be
put into a :class:`HealthResult`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from .serving import ServingStack


class HealthStatus(StrEnum):
    """Coarse probe outcome; any non-healthy result triggers auto-revert."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True)
class HealthResult:
    """Health outcome with stable, non-sensitive failed-check identifiers."""

    status: HealthStatus
    failed_checks: tuple[str, ...] = ()

    @property
    def healthy(self) -> bool:
        """Return True only for a HEALTHY result."""
        return self.status is HealthStatus.HEALTHY


class HealthProbe(Protocol):
    """Post-commit verification seam."""

    def check(self, expected_version: str | None = None) -> HealthResult:
        """Check serving health, optionally asserting the active version."""
        ...


class FakeHealthProbe:
    """FAKE probe that returns queued results, then HEALTHY indefinitely."""

    def __init__(self, results: Sequence[HealthResult] | None = None) -> None:
        """Initialise with an optional finite script of results."""
        self._results = deque(results or [])
        self.calls = 0
        self.expected_versions: list[str | None] = []

    def enqueue(self, result: HealthResult) -> None:
        """Append one scripted result to the queue."""
        if not isinstance(result, HealthResult):
            raise TypeError("result must be a HealthResult")
        self._results.append(result)

    def check(self, expected_version: str | None = None) -> HealthResult:
        """Return the next scripted result or a deterministic healthy result."""
        self.calls += 1
        self.expected_versions.append(expected_version)
        return self._results.popleft() if self._results else HealthResult(HealthStatus.HEALTHY)


class ServingStateProbe:
    """FAKE probe that verifies every serving target after a swap."""

    def __init__(self, stack: ServingStack, fail_checks: Sequence[str] = ()) -> None:
        """Watch a fake stack with optional forced failed check IDs."""
        self._stack = stack
        self._fail_checks = set(fail_checks)
        self.calls = 0
        self.expected_versions: list[str | None] = []

    def check(self, expected_version: str | None = None) -> HealthResult:
        """Fail when a target is inactive, still staged, or forced to fail."""
        self.calls += 1
        self.expected_versions.append(expected_version)
        failed: list[str] = []
        for state in self._stack.states():
            check_id = f"{state.target.value}.version"
            if state.current_version is None or (
                expected_version is not None and state.current_version != expected_version
            ):
                failed.append(check_id)
            if state.staged_version is not None:
                failed.append(f"{state.target.value}.staged")
        failed.extend(sorted(self._fail_checks))
        if failed:
            return HealthResult(HealthStatus.UNHEALTHY, tuple(sorted(set(failed))))
        return HealthResult(HealthStatus.HEALTHY)
