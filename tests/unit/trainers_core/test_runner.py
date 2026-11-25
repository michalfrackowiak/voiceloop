"""End-to-end and concurrency tests for the pinned-image trainer runner."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from threading import Event, Thread
from typing import Any

import pytest

from services.trainers.core import (
    TRAINER_CONTRACT_SCHEMA,
    TRAINER_CONTRACT_VERSION,
    FakeMlflowLogger,
    JobBusyError,
    TrainerPlugin,
    TrainerRegistry,
    TrainerResult,
    TrainerRunner,
    TrainerSettings,
    build_trainer_result,
)
from voiceloop_common import ValidationError, frozen_clock, use_clock, validate

from .conftest import DATASET_HASH, DummyTrainer


def _make_runner(
    settings: TrainerSettings, *plugins: TrainerPlugin
) -> tuple[TrainerRunner, FakeMlflowLogger]:
    """Build a runner, registry and injectable logger fake for a test."""
    registry = TrainerRegistry()
    for plugin in plugins:
        registry.register(plugin)
    logger = FakeMlflowLogger()
    return TrainerRunner(registry, settings, logger=logger), logger


def test_dummy_trainer_runs_end_to_end(manifest: dict[str, Any], settings: TrainerSettings) -> None:
    """A dummy plugin produces a contract-valid run with all lineage logged."""
    runner, logger = _make_runner(settings, DummyTrainer())
    frozen = datetime(2026, 11, 25, 9, 30, tzinfo=UTC)

    with use_clock(frozen_clock(frozen)):
        result = runner.run("dummy", manifest)

    validate(result, TRAINER_CONTRACT_SCHEMA)
    assert result["schema_version"] == TRAINER_CONTRACT_VERSION
    assert result["contract_part"] == "output"
    assert result["metrics"] == {"train_loss": 0.25}
    assert result["artifact_uri"] == "fake://trainer-artifacts/dummy-adapter"
    assert result["lineage"] == {
        "dataset_hash": DATASET_HASH,
        "git_sha": settings.git_sha,
        "image_digest": settings.image_digest,
    }

    assert len(logger.runs) == 1
    run = logger.runs[0]
    assert run.run_id == result["run_id"]
    assert run.trainer == "dummy"
    assert run.dataset_hash == DATASET_HASH
    assert run.git_sha == settings.git_sha
    assert run.image_digest == settings.image_digest
    assert run.artifact_uri == result["artifact_uri"]
    assert run.adapter_uri == result["artifact_uri"]
    assert run.metrics == result["metrics"]
    assert run.started_at == frozen
    assert run.finished_at == frozen


def test_invalid_manifest_is_rejected_before_training(
    manifest: dict[str, Any], settings: TrainerSettings
) -> None:
    """Contract validation stops a run with no dataset hash."""
    runner, logger = _make_runner(settings, DummyTrainer())
    with pytest.raises(ValidationError, match="dataset_manifest"):
        runner.run("dummy", {"dataset_id": "unhashed"})
    assert logger.runs == ()


def test_runner_rejects_lineage_that_does_not_match_settings(
    manifest: dict[str, Any], settings: TrainerSettings
) -> None:
    """A plugin cannot claim a different git SHA or image than the job settings."""

    class LyingTrainer(TrainerPlugin):
        """Plugin that reports a git SHA other than the pinned job's SHA."""

        @property
        def name(self) -> str:
            """Return the registered name."""
            return "lying"

        def train(
            self, dataset_manifest: Mapping[str, Any], job_settings: TrainerSettings
        ) -> TrainerResult:
            """Return a result with tampered lineage."""
            result = build_trainer_result(
                dataset_manifest, job_settings, artifact_uri="fake://lying"
            )
            result["lineage"]["git_sha"] = "f" * 40
            return result

    runner, logger = _make_runner(settings, LyingTrainer())
    with pytest.raises(ValidationError, match="lineage"):
        runner.run("lying", manifest)
    assert logger.runs == ()


def test_second_concurrent_job_is_rejected(
    manifest: dict[str, Any], settings: TrainerSettings
) -> None:
    """The single-job lock rejects a concurrent job instead of queueing it."""
    started = Event()
    release = Event()

    class BlockingTrainer(TrainerPlugin):
        """Plugin that holds a job open until the test releases it."""

        @property
        def name(self) -> str:
            """Return the registered name."""
            return "blocking"

        def train(
            self, dataset_manifest: Mapping[str, Any], job_settings: TrainerSettings
        ) -> TrainerResult:
            """Signal start, wait for release, then return a valid result."""
            started.set()
            assert release.wait(timeout=10), "test did not release the trainer"
            return build_trainer_result(
                dataset_manifest, job_settings, artifact_uri="fake://blocking"
            )

    runner, logger = _make_runner(settings, BlockingTrainer())
    outcomes: dict[str, Any] = {}

    def first_job() -> None:
        """Run the blocking trainer on a worker thread."""
        try:
            outcomes["result"] = runner.run("blocking", manifest)
        except Exception as exc:  # captured for the main thread to assert
            outcomes["error"] = exc

    thread = Thread(target=first_job, name="trainer-first-job")
    thread.start()
    assert started.wait(timeout=10), "first job did not start"
    with pytest.raises(JobBusyError):
        runner.run("blocking", manifest)
    release.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert "error" not in outcomes
    validate(outcomes["result"], TRAINER_CONTRACT_SCHEMA)
    assert len(logger.runs) == 1


def test_lock_is_released_after_training_exception(
    manifest: dict[str, Any], settings: TrainerSettings
) -> None:
    """A failed job cannot wedge the runner for all later jobs."""

    class FailingTrainer(TrainerPlugin):
        """Plugin that always raises mid-training."""

        @property
        def name(self) -> str:
            """Return the registered name."""
            return "failing"

        def train(
            self, dataset_manifest: Mapping[str, Any], job_settings: TrainerSettings
        ) -> TrainerResult:
            """Raise a training failure."""
            raise RuntimeError("training exploded")

    runner, logger = _make_runner(settings, FailingTrainer(), DummyTrainer())
    with pytest.raises(RuntimeError, match="training exploded"):
        runner.run("failing", manifest)
    assert logger.runs == ()

    result = runner.run("dummy", manifest)
    assert result["run_id"]
    assert len(logger.runs) == 1
