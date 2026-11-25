"""Pinned-image job runner implementing the trainer contract end to end.

The runner validates the 1.0.0 TrainRequest with ``voiceloop_common.validate``,
executes exactly one registered plugin under a non-blocking job lock, validates
its TrainResponse, and logs lineage plus timestamps through an injectable
MLflow logger fake. Docker, MLflow and network access are never used here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from voiceloop_common import ValidationError, utcnow, validate

from .base import (
    TRAINER_CONTRACT_SCHEMA,
    TRAINER_CONTRACT_VERSION,
    TrainerLineage,
    TrainerPlugin,
    TrainerResult,
)
from .lock import JobLock, SingleJobLock
from .mlflow_logging import FakeMlflowLogger, MlflowLogger, TrainingRun
from .registry import TrainerRegistry
from .settings import TrainerSettings


class TrainerRunner:
    """Run one registered trainer at a time and log a lineage-complete run."""

    def __init__(
        self,
        registry: TrainerRegistry,
        settings: TrainerSettings | None = None,
        *,
        logger: MlflowLogger | None = None,
        lock: JobLock | None = None,
    ) -> None:
        """Wire the runner to a registry and optional settings/logger/lock fakes.

        Args:
            registry: Registry holding the trainer plugins.
            settings: Typed job settings; instantiated from env when omitted.
            logger: MLflow logger implementation; in-memory fake by default.
            lock: Job lock; in-process single-job lock by default.

        """
        self._registry = registry
        self._settings = settings or TrainerSettings()
        self._logger: MlflowLogger = logger or FakeMlflowLogger()
        self._lock = lock or SingleJobLock()

    def run(self, trainer_name: str, dataset_manifest: Mapping[str, Any]) -> TrainerResult:
        """Validate, execute and log one training job.

        Args:
            trainer_name: Name of a plugin in the registry.
            dataset_manifest: Manifest with a non-empty ``content_hash``.

        Returns:
            The validated 1.0.0 contract output payload, including lineage.

        Raises:
            PluginNotFoundError: If ``trainer_name`` is not registered.
            JobBusyError: If another job holds the lock.
            ValidationError: If the request or plugin result violates the
                contract, or the result lineage does not match job settings.

        """
        plugin: TrainerPlugin = self._registry.get(trainer_name)
        with self._lock.held():
            request: dict[str, Any] = {
                "schema_version": TRAINER_CONTRACT_VERSION,
                "contract_part": "input",
                "dataset_manifest": dict(dataset_manifest),
                "config": self._settings.as_contract_config(trainer_name),
            }
            validate(request, TRAINER_CONTRACT_SCHEMA)

            started_at = utcnow()
            result = plugin.train(dict(dataset_manifest), self._settings)
            validate(result, TRAINER_CONTRACT_SCHEMA)

            expected_lineage: TrainerLineage = {
                "dataset_hash": str(dataset_manifest["content_hash"]),
                "git_sha": self._settings.git_sha,
                "image_digest": self._settings.image_digest,
            }
            if result["lineage"] != expected_lineage:
                raise ValidationError("trainer result lineage does not match job settings")
            finished_at = utcnow()

            self._logger.log_run(
                TrainingRun(
                    run_id=result["run_id"],
                    trainer=trainer_name,
                    dataset_hash=expected_lineage["dataset_hash"],
                    git_sha=expected_lineage["git_sha"],
                    image_digest=expected_lineage["image_digest"],
                    artifact_uri=result["artifact_uri"],
                    metrics=dict(result["metrics"]),
                    started_at=started_at,
                    finished_at=finished_at,
                )
            )
            return result
