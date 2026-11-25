"""P2-12.1: exact, deduplicated lineage in both directions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from services.datasets.lineage import (
    FakeManifestSource,
    FakeTrainerLineageSource,
    LineageGraph,
    MappingExampleResolver,
    UnknownCallError,
    UnknownExampleError,
)

CALL_A = "call-0001"
CALL_B = "call-0002"
UNKNOWN_CALL = "call-does-not-exist"
RUN_PARSER = "run-parser-v1"
RUN_STT = "run-stt-v1"
MODEL_PARSER = "model-parser-v1"


def test_example_to_call_returns_only_committed_datasets_and_models(
    graph: LineageGraph, manifest: dict[str, Any], example_id: str
) -> None:
    lineage = graph.example_to_call(example_id)

    assert lineage.example_id == example_id
    assert lineage.call_id == CALL_A
    assert tuple(dataset.dataset_id for dataset in lineage.datasets) == (manifest["dataset_id"],)
    assert tuple(dataset.content_hash for dataset in lineage.datasets) == (
        manifest["content_hash"],
    )
    assert {model.run_id for model in lineage.models} == {RUN_PARSER, RUN_STT}
    assert {model.model_id for model in lineage.models} == {MODEL_PARSER, RUN_STT}
    assert all(model.dataset_id == manifest["dataset_id"] for model in lineage.models)
    assert all(model.content_hash == manifest["content_hash"] for model in lineage.models)
    assert all(model.git_sha and model.image_digest for model in lineage.models)


def test_call_to_models_covers_every_trained_model_once(graph: LineageGraph) -> None:
    for call_id in (CALL_A, CALL_B):
        models = graph.models_for_call(call_id)
        assert tuple(model.run_id for model in models) == (RUN_PARSER, RUN_STT)
        assert len(models) == 2  # duplicates in both sources are deduplicated
    assert len(graph.datasets()) == 1
    assert len(graph.models()) == 2


def test_unknown_call_and_example_fail_closed(graph: LineageGraph) -> None:
    with pytest.raises(UnknownCallError):
        graph.datasets_for_call(UNKNOWN_CALL)
    with pytest.raises(UnknownCallError):
        graph.models_for_call(UNKNOWN_CALL)
    with pytest.raises(UnknownExampleError):
        graph.example_to_call(f"{CALL_A}:999:stt")  # well-formed, never committed
    with pytest.raises(UnknownExampleError):
        graph.example_to_call("not-an-example-id")


def test_example_query_without_index_fails_closed(
    manifest: dict[str, Any], trainer_results: tuple[Mapping[str, Any], ...]
) -> None:
    graph = LineageGraph(
        FakeManifestSource([manifest]),
        FakeTrainerLineageSource(trainer_results),
    )
    with pytest.raises(UnknownExampleError, match="no exact example index"):
        graph.example_to_call(f"{CALL_A}:1:stt")
    assert len(graph.models_for_call(CALL_A)) == 2


def test_explicit_example_index_is_still_checked_against_manifests(
    manifest: dict[str, Any], trainer_results: tuple[Mapping[str, Any], ...]
) -> None:
    graph = LineageGraph(
        FakeManifestSource([manifest]),
        FakeTrainerLineageSource(trainer_results),
        example_resolver=MappingExampleResolver({f"{CALL_A}:1:stt": UNKNOWN_CALL}),
    )
    with pytest.raises(UnknownExampleError, match="no committed dataset lineage"):
        graph.example_to_call(f"{CALL_A}:1:stt")


def test_missing_manifest_blocks_call_queries(
    trainer_results: tuple[Mapping[str, Any], ...],
) -> None:
    graph = LineageGraph(
        FakeManifestSource(),
        FakeTrainerLineageSource(trainer_results),
    )
    with pytest.raises(UnknownCallError):
        graph.models_for_call(CALL_A)
    # Raw trainer results remain visible, but no dataset ID can be joined to
    # them without a committed manifest, so no call link is fabricated.
    assert len(graph.models()) == 2
    assert all(model.dataset_id == "" for model in graph.models())
