"""Core trainer plugin interface, registry and pinned-image job runner."""

from __future__ import annotations

from .base import (
    TRAINER_CONTRACT_SCHEMA,
    TRAINER_CONTRACT_VERSION,
    TrainerLineage,
    TrainerPlugin,
    TrainerResult,
    build_trainer_result,
)
from .lock import FileJobLock, JobBusyError, JobLock, SingleJobLock
from .mlflow_logging import FakeMlflowLogger, MlflowLogger, TrainingRun
from .registry import (
    DuplicatePluginError,
    PluginNotFoundError,
    TrainerRegistry,
    TrainerRegistryError,
)
from .runner import TrainerRunner
from .settings import TrainerSettings

__all__ = [
    "TRAINER_CONTRACT_SCHEMA",
    "TRAINER_CONTRACT_VERSION",
    "DuplicatePluginError",
    "FakeMlflowLogger",
    "FileJobLock",
    "JobBusyError",
    "JobLock",
    "MlflowLogger",
    "PluginNotFoundError",
    "SingleJobLock",
    "TrainerLineage",
    "TrainerPlugin",
    "TrainerRegistry",
    "TrainerRegistryError",
    "TrainerResult",
    "TrainerRunner",
    "TrainerSettings",
    "TrainingRun",
    "build_trainer_result",
]
