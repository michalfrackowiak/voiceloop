"""Transactional serving adapter protocol and deterministic fakes.

> [!IMPORTANT]
> **FAKE ONLY.** :class:`FakeServingAdapter` and :class:`FakeServingStack`
> mutate in-memory version strings. They never load a model, call vLLM, touch a
> GPU or contact a serving HTTP endpoint.

Both STT and parser use the same ``prepare`` → ``commit`` / ``rollback``
protocol, so :class:`ServingStack` can roll every target back in reverse order
and never leave a partial swap active. Real adapter seams:

- **STT (CTranslate2 / ONNX):** ``prepare`` copies the verified artifact into a
  versioned staging directory (never references the agent's temp dir);
  ``commit`` flips one atomic symlink/rename to activate it; ``rollback`` points
  the symlink back at the exact recorded previous version before deleting the
  staging copy.
- **Parser (vLLM LoRA):** ``prepare`` calls the load-adapter endpoint and keeps
  the new adapter inactive; ``commit`` switches the serving adapter alias in
  one control-plane write; ``rollback`` switches back to the previous adapter
  first, then unloads the new one.

Neither CTranslate2, ONNX nor vLLM is imported; production adapters should
implement this protocol outside this package and be injected into the agent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol

from voiceloop_common import ValidationError

from .artifacts import DownloadedArtifact


class ServingTarget(StrEnum):
    """The two serving components mirrored by the same transaction."""

    STT = "stt"
    PARSER = "parser"


TARGET_ORDER = (ServingTarget.STT, ServingTarget.PARSER)


@dataclass(frozen=True)
class ServingState:
    """Observable state of one serving target."""

    target: ServingTarget
    current_version: str | None
    staged_version: str | None = None
    artifact_hash: str | None = None


class ServingAdapter(Protocol):
    """Atomic-swap seam shared by STT and parser serving adapters."""

    @property
    def target(self) -> ServingTarget:
        """Return the serving component this adapter controls."""
        ...

    def state(self) -> ServingState:
        """Return the current (and any staged) serving state."""
        ...

    def prepare(self, version: str, artifact: DownloadedArtifact, artifact_hash: str) -> None:
        """Stage a verified artifact without activating it.

        A real adapter must copy the artifact out of the agent's temp directory
        and clean up its own partial staging if this method raises.
        """
        ...

    def commit(self) -> None:
        """Activate the staged artifact in one atomic serving-side switch."""
        ...

    def rollback(self) -> None:
        """Restore the exact state recorded before ``prepare`` (idempotent)."""
        ...


@dataclass(frozen=True)
class SwapPlan:
    """Transaction record for one alias-driven swap across all targets."""

    version: str
    artifact_hash: str
    before: tuple[ServingState, ...]
    prepared: tuple[ServingTarget, ...] = ()
    committed: tuple[ServingTarget, ...] = ()


class ServingStack:
    """Orchestrates prepare/commit/rollback across the STT and parser adapters."""

    def __init__(self, adapters: Mapping[ServingTarget, ServingAdapter]) -> None:
        """Require exactly one adapter for each serving target."""
        if set(adapters) != set(TARGET_ORDER):
            raise ValidationError("serving stack needs exactly one STT and one parser adapter")
        for target in TARGET_ORDER:
            if adapters[target].target != target:
                raise ValidationError(f"adapter for {target.value} reports a different target")
        self._adapters: dict[ServingTarget, ServingAdapter] = dict(adapters)

    def states(self) -> tuple[ServingState, ...]:
        """Return states in the deterministic STT-then-parser order."""
        return tuple(self._adapters[target].state() for target in TARGET_ORDER)

    def prepare(self, plan: SwapPlan, artifact: DownloadedArtifact) -> SwapPlan:
        """Stage the artifact on every target, rolling back on any failure."""
        prepared = plan.prepared
        for target in TARGET_ORDER:
            adapter = self._adapters[target]
            try:
                adapter.prepare(plan.version, artifact, plan.artifact_hash)
            except Exception:
                self.rollback(replace(plan, prepared=prepared))
                raise
            prepared += (target,)
        return replace(plan, prepared=prepared)

    def commit(self, plan: SwapPlan) -> SwapPlan:
        """Activate all staged targets, rolling back every target on failure."""
        committed = plan.committed
        for target in plan.prepared:
            try:
                self._adapters[target].commit()
            except Exception:
                self.rollback(plan)
                raise
            committed += (target,)
        return replace(plan, committed=committed)

    def rollback(self, plan: SwapPlan) -> tuple[str, ...]:
        """Roll back prepared targets in reverse order, returning errors.

        This method never raises: a failing rollback is reported to the caller
        as one stable ``target:ExceptionType`` string per failed target.
        """
        errors: list[str] = []
        for target in reversed(plan.prepared):
            try:
                self._adapters[target].rollback()
            except Exception as exc:
                errors.append(f"{target.value}:{type(exc).__name__}")
        return tuple(errors)


class FakeServingAdapter:
    """Deterministic in-memory FAKE for one serving target."""

    def __init__(self, target: ServingTarget, current_version: str | None = None) -> None:
        """Initialise a fake target at an optional current version."""
        if not isinstance(target, ServingTarget):
            raise ValidationError("target must be a ServingTarget")
        if current_version is not None and (
            not isinstance(current_version, str) or not current_version.strip()
        ):
            raise ValidationError("current_version must be a non-blank string or None")
        self._target = target
        self._current_version = current_version
        self._staged_version: str | None = None
        self._artifact_hash: str | None = None
        self._before: ServingState | None = None
        self.prepare_calls: list[tuple[str, str]] = []
        self.commit_calls = 0
        self.rollback_calls = 0
        self.fail_prepare = False
        self.fail_commit = False
        self.fail_rollback = False

    @property
    def target(self) -> ServingTarget:
        """Return the fake target this adapter controls."""
        return self._target

    def state(self) -> ServingState:
        """Return current, staged and artifact-hash fields."""
        return ServingState(
            target=self._target,
            current_version=self._current_version,
            staged_version=self._staged_version,
            artifact_hash=self._artifact_hash,
        )

    def prepare(self, version: str, artifact: DownloadedArtifact, artifact_hash: str) -> None:
        """Record a staged version without activating it (FAKE)."""
        if not isinstance(version, str) or not version.strip():
            raise ValidationError("version must be a non-blank string")
        if self.fail_prepare:
            raise RuntimeError("fake serving prepare failure")
        self._before = self.state()
        self._staged_version = version
        self._artifact_hash = artifact_hash
        self.prepare_calls.append((version, artifact.root.as_posix()))

    def commit(self) -> None:
        """Activate the staged version (FAKE)."""
        if self.fail_commit:
            raise RuntimeError("fake serving commit failure")
        if self._staged_version is None:
            raise ValidationError("nothing staged for commit")
        self._current_version = self._staged_version
        self._staged_version = None
        self.commit_calls += 1

    def rollback(self) -> None:
        """Restore the exact pre-prepare state (FAKE)."""
        self.rollback_calls += 1
        if self.fail_rollback:
            raise RuntimeError("fake serving rollback failure")
        if self._before is not None:
            self._current_version = self._before.current_version
            self._staged_version = self._before.staged_version
            self._artifact_hash = self._before.artifact_hash
            self._before = None
        else:
            self._staged_version = None
            self._artifact_hash = None


class FakeServingStack(ServingStack):
    """FAKE stack exposing its fake adapters for assertions."""

    def __init__(self, stt_version: str | None = None, parser_version: str | None = None) -> None:
        """Build one fake STT adapter and one fake parser adapter."""
        adapters: dict[ServingTarget, FakeServingAdapter] = {
            ServingTarget.STT: FakeServingAdapter(ServingTarget.STT, stt_version),
            ServingTarget.PARSER: FakeServingAdapter(ServingTarget.PARSER, parser_version),
        }
        super().__init__(adapters)
        self.fake_adapters = adapters
