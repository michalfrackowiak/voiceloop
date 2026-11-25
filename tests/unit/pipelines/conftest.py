"""Shared fakes for deterministic P2-10 pipeline tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest

from infra.pipelines.chain import PipelineSteps
from infra.pipelines.settings import PipelineSettings
from voiceloop_common import frozen_clock

FIXED_NOW = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)

DATASET_HASH = "a" * 64

MANIFEST: dict[str, Any] = {
    "schema_version": "1.0.0",
    "dataset_id": "ds-test",
    "content_hash": DATASET_HASH,
    "source_call_ids": ["call-1", "call-2"],
    "splits": {"train": ["call-1"], "val": ["call-2"], "test": []},
    "filter_counts": {
        "consent_dropped": 0,
        "pii_dropped": 0,
        "replay_added": 0,
        "golden_excluded": 0,
        "regression_excluded": 0,
        "synthetic_tagged": 2,
    },
    "created_at": FIXED_NOW.isoformat(),
    "examples": 2,
    "synthetic": True,
}

TRAIN_RESULT: dict[str, Any] = {
    "schema_version": "1.0.0",
    "contract_part": "output",
    "artifact_uri": "fake://artifact/deployable",
    "metrics": {"train_loss": 0.1},
    "run_id": "run-test-1",
    "lineage": {
        "dataset_hash": DATASET_HASH,
        "git_sha": "1" * 40,
        "image_digest": "fake@sha256:" + "2" * 64,
    },
}

EVAL_RESULT: dict[str, Any] = {
    "report_path": "fake://evalgate/report.json",
    "artifact_uri": "fake://artifact/deployable",
    "report": {"schema_version": "1.0.0"},
}


@dataclass
class Recorder:
    """Captured calls from a recording fake pipeline."""

    steps: PipelineSteps
    dataset_calls: list[int] = field(default_factory=list)
    train_manifests: list[dict[str, Any]] = field(default_factory=list)
    eval_calls: list[tuple[str, Sequence[str]]] = field(default_factory=list)


def make_recording_steps(
    *,
    fail_at: str | None = None,
    train_result: dict[str, Any] | None = None,
    eval_result: dict[str, Any] | None = None,
) -> Recorder:
    """Create injectable steps that record calls and can fail at one stage."""
    recorder: Recorder

    def dataset() -> dict[str, Any]:
        """Return the fake manifest, or fail if requested."""
        recorder.dataset_calls.append(1)
        if fail_at == "dataset":
            raise RuntimeError("dataset boom")
        return dict(MANIFEST)

    def trainer(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return the fake trainer result, or fail if requested."""
        recorder.train_manifests.append(dict(manifest))
        if fail_at == "train":
            raise RuntimeError("train boom")
        return dict(train_result if train_result is not None else TRAIN_RESULT)

    def evaluator(artifact_uri: str, eval_sets: Sequence[str]) -> Mapping[str, Any]:
        """Return the fake eval result, or fail if requested."""
        recorder.eval_calls.append((artifact_uri, eval_sets))
        if fail_at == "evaluate":
            raise RuntimeError("evaluate boom")
        return dict(eval_result if eval_result is not None else EVAL_RESULT)

    recorder = Recorder(steps=PipelineSteps(dataset=dataset, trainer=trainer, evaluator=evaluator))
    return recorder


@pytest.fixture
def pipeline_settings() -> PipelineSettings:
    """Return local-safe settings with a small trigger threshold."""
    return PipelineSettings(min_new_verified=2, fixtures_path="tests/fixtures/turns.jsonl")


@pytest.fixture
def fixed_clock() -> Any:
    """Return a deterministic UTC clock."""
    return frozen_clock(FIXED_NOW)
