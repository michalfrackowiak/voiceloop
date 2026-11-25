"""MLflow logging seam and in-memory fake.

No MLflow, network or container runtime is imported here. A real adapter
implements :class:`MlflowLogger` by translating :class:`TrainingRun` into MLflow
runs/params/tags/metrics; see ``services/trainers/core/README.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from .base import TRAINER_CONTRACT_VERSION


@dataclass(frozen=True, slots=True)
class TrainingRun:
    """One completed training job with full lineage for MLflow."""

    run_id: str
    trainer: str
    dataset_hash: str
    git_sha: str
    image_digest: str
    artifact_uri: str
    metrics: dict[str, float]
    started_at: datetime
    finished_at: datetime
    schema_version: str = TRAINER_CONTRACT_VERSION

    @property
    def adapter_uri(self) -> str:
        """Return the deployable adapter/converted-artifact URI.

        This aliases ``artifact_uri`` from the trainer contract: for LoRA
        trainers it is the merged adapter URI, not a raw training checkpoint.
        """
        return self.artifact_uri


class MlflowLogger(Protocol):
    """Minimal interface implemented by fakes and a future real MLflow adapter."""

    def log_run(self, run: TrainingRun) -> None:
        """Persist one completed training run and its lineage."""
        ...


class FakeMlflowLogger:
    """In-memory MLflow logger fake for tests and local runs."""

    def __init__(self) -> None:
        """Create a logger with no recorded runs."""
        self._runs: list[TrainingRun] = []

    def log_run(self, run: TrainingRun) -> None:
        """Append a run to the in-memory log in call order."""
        self._runs.append(run)

    @property
    def runs(self) -> tuple[TrainingRun, ...]:
        """Return an immutable snapshot of all logged runs."""
        return tuple(self._runs)
