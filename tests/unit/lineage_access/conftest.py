"""P2-12 fixtures built from read-only P2-05/P2-06 public APIs.

Manifests come from the committed synthetic fixture pool; trainer results are
contract-valid P2-06 payloads. No fixture edits and no real guest data.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from services.datasets import build_dataset_from_fixtures, build_manifest
from services.datasets.auth import Authorizer
from services.datasets.lineage import (
    DeletionWorkflow,
    FakeManifestSource,
    FakeTrainerLineageSource,
    FixedClock,
    InMemoryAuditSink,
    InMemoryModelFlagSink,
    InMemoryTombstoneSink,
    LineageGraph,
    P205ExampleIdResolver,
    Principal,
    Role,
)
from services.trainers.core import TrainerSettings, build_trainer_result

CALL_A = "call-0001"
CALL_B = "call-0002"
UNKNOWN_CALL = "call-does-not-exist"
FIXED_NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
MANIFEST_TIME = "2026-01-01T00:00:00Z"
RUN_PARSER = "run-parser-v1"
RUN_STT = "run-stt-v1"
MODEL_PARSER = "model-parser-v1"


@pytest.fixture
def dataset_build() -> Any:
    """Return a real P2-05 build over the synthetic fixture pool."""
    return build_dataset_from_fixtures()


@pytest.fixture
def manifest(dataset_build: Any) -> dict[str, Any]:
    """Return a schema-valid P2-05 manifest with a fixed creation time."""
    return build_manifest(dataset_build, created_at=MANIFEST_TIME)


@pytest.fixture
def example_id(dataset_build: Any) -> str:
    """Return one committed example ID for call A."""
    candidates = [
        example.example_id
        for example in dataset_build.examples
        if example.call_id == CALL_A and example.kind == "parser_clean"
    ]
    assert candidates
    return candidates[0]


@pytest.fixture
def trainer_results(manifest: dict[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return two contract-valid trainer results for the same dataset hash."""
    settings = TrainerSettings()
    return (
        build_trainer_result(
            manifest,
            settings,
            artifact_uri="fake://artifacts/parser-v1",
            run_id=RUN_PARSER,
        ),
        build_trainer_result(
            manifest,
            settings,
            artifact_uri="fake://artifacts/stt-v1",
            run_id=RUN_STT,
        ),
    )


@pytest.fixture
def graph(
    manifest: dict[str, Any],
    trainer_results: tuple[Mapping[str, Any], ...],
    dataset_build: Any,
) -> LineageGraph:
    """Build a graph with duplicated manifests/results to exercise dedup."""
    return LineageGraph(
        FakeManifestSource([manifest, dict(manifest)]),
        FakeTrainerLineageSource(
            [*trainer_results, *trainer_results],
            model_ids={RUN_PARSER: MODEL_PARSER},
        ),
        example_resolver=P205ExampleIdResolver(
            example.example_id for example in dataset_build.examples
        ),
    )


@pytest.fixture
def clock() -> FixedClock:
    """Return a deterministic UTC clock."""
    return FixedClock(FIXED_NOW)


@pytest.fixture
def audit_sink() -> InMemoryAuditSink:
    """Return an empty in-memory audit chain."""
    return InMemoryAuditSink()


@pytest.fixture
def authorizer(audit_sink: InMemoryAuditSink, clock: FixedClock) -> Authorizer:
    """Return a default-deny authorizer wired to fake audit and clock."""
    return Authorizer(audit_sink, clock)


@pytest.fixture
def tombstones() -> InMemoryTombstoneSink:
    """Return an empty fake tombstone sink and exclusion source."""
    return InMemoryTombstoneSink()


@pytest.fixture
def model_flags() -> InMemoryModelFlagSink:
    """Return an empty fake model flag sink."""
    return InMemoryModelFlagSink()


@pytest.fixture
def workflow(
    graph: LineageGraph,
    authorizer: Authorizer,
    tombstones: InMemoryTombstoneSink,
    model_flags: InMemoryModelFlagSink,
    audit_sink: InMemoryAuditSink,
    clock: FixedClock,
) -> DeletionWorkflow:
    """Return the full deletion workflow with only fakes injected."""
    return DeletionWorkflow(
        graph=graph,
        authorizer=authorizer,
        tombstones=tombstones,
        model_flags=model_flags,
        audit_sink=audit_sink,
        clock=clock,
    )


@pytest.fixture
def admin() -> Principal:
    """Return an authenticated admin principal."""
    return Principal("admin-1", Role.ADMIN)


@pytest.fixture
def reviewer() -> Principal:
    """Return an authenticated reviewer principal."""
    return Principal("reviewer-1", Role.REVIEWER)


@pytest.fixture
def ml_engineer() -> Principal:
    """Return an authenticated ML engineer principal."""
    return Principal("ml-1", Role.ML_ENGINEER)
