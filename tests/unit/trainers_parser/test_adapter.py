"""Deterministic adapter export, store and fake vLLM load/unload tests."""

from __future__ import annotations

import json

import pytest

from services.trainers.parser import (
    FakeAdapterStore,
    FakeLoraAdapter,
    FakeLoraTrainingBackend,
    FakeVllmRuntime,
    ParserExample,
    canonical_json,
    sha256_text,
)
from tests.unit.trainers_parser.conftest import DATASET_HASH, make_example, make_slots
from voiceloop_common import ValidationError


def test_from_mapping_rejects_malformed_examples() -> None:
    valid = {
        "example_id": "ex-1",
        "kind": "gold",
        "source_text": "hello",
        "slots": make_slots(),
    }
    with pytest.raises(ValidationError, match="kind"):
        ParserExample.from_mapping({**valid, "kind": "bad"})
    with pytest.raises(ValidationError, match="source_text"):
        ParserExample.from_mapping({**valid, "source_text": "  "})
    with pytest.raises(ValidationError, match="slots"):
        ParserExample.from_mapping({**valid, "slots": {}})
    with pytest.raises(ValidationError, match="example_id"):
        ParserExample.from_mapping({**valid, "example_id": ""})


def test_fake_backend_is_deterministic_and_order_independent(
    examples: tuple[ParserExample, ...],
) -> None:
    backend = FakeLoraTrainingBackend()
    first = backend.train(examples, model="fixture-base", seed=7, params={"epochs": 1})
    reordered = backend.train(
        tuple(reversed(examples)), model="fixture-base", seed=7, params={"epochs": 1}
    )
    assert first == reordered
    changed = backend.train(examples, model="fixture-base", seed=8, params={"epochs": 1})
    assert changed != first
    assert all(delta.kind in ("gold", "noisy") for delta in first)


def test_gold_and_noisy_examples_produce_distinct_deltas() -> None:
    backend = FakeLoraTrainingBackend()
    gold = (make_example("ex-1", "gold", "Check in on the twelfth."),)
    both = gold + (make_example("ex-2", "noisy", "Check in on the twelf."),)
    gold_deltas = backend.train(gold, model="", seed=0, params={})
    both_deltas = backend.train(both, model="", seed=0, params={})
    assert gold_deltas != both_deltas
    assert {delta.example_id for delta in both_deltas} == {"ex-1", "ex-2"}


def test_adapter_content_hash_changes_with_payload() -> None:
    backend = FakeLoraTrainingBackend()
    deltas = backend.train(
        (make_example("ex-1", "gold", "Check in on the twelfth."),),
        model="",
        seed=0,
        params={},
    )
    base = FakeLoraAdapter(dataset_hash=DATASET_HASH, deltas=deltas)
    other_data = FakeLoraAdapter(dataset_hash="e" * 64, deltas=deltas)
    other_params = FakeLoraAdapter(dataset_hash=DATASET_HASH, deltas=deltas, params={"epochs": 2})
    assert len(base.content_hash()) == 64
    assert base.content_hash() != other_data.content_hash()
    assert base.content_hash() != other_params.content_hash()


def test_export_does_not_copy_source_transcript(examples: tuple[ParserExample, ...]) -> None:
    backend = FakeLoraTrainingBackend()
    adapter = FakeLoraAdapter(
        dataset_hash=DATASET_HASH,
        deltas=backend.train(examples, model="", seed=0, params={}),
    )
    exported = canonical_json(adapter.to_dict())
    assert examples[0].source_text not in exported
    assert '"synthetic_training_standin":true' in exported
    assert json.loads(exported)["kind"] == "fake-lora"


def test_store_save_load_roundtrip_is_idempotent(
    store: FakeAdapterStore, examples: tuple[ParserExample, ...]
) -> None:
    deltas = FakeLoraTrainingBackend().train(examples, model="m", seed=1, params={})
    adapter = FakeLoraAdapter(dataset_hash=DATASET_HASH, deltas=deltas)
    uri = store.save(adapter, prefix="fake://parser-adapters")
    assert uri.endswith(adapter.content_hash())
    assert store.save(adapter, prefix="fake://parser-adapters") == uri
    assert len(store.uris) == 1
    assert store.load(uri) == adapter


def test_store_fails_closed_on_tampered_or_missing_artifact(
    store: FakeAdapterStore, examples: tuple[ParserExample, ...]
) -> None:
    adapter = FakeLoraAdapter(
        dataset_hash=DATASET_HASH,
        deltas=FakeLoraTrainingBackend().train(examples, model="m", seed=1, params={}),
    )
    uri = store.save(adapter, prefix="fake://parser-adapters")
    raw = json.loads(canonical_json(adapter.to_dict()))
    raw["deltas"][0]["weight"] = 0.5
    store.put_raw(uri, canonical_json(raw))
    with pytest.raises(ValidationError, match="content hash"):
        store.load(uri)

    missing = "fake://parser-adapters/fake-lora/" + "f" * 64
    with pytest.raises(ValidationError, match="not found"):
        store.load(missing)
    store.put_raw(missing, "{not json")
    with pytest.raises(ValidationError, match="not valid JSON"):
        store.load(missing)

    nan_raw = canonical_json(raw).replace(str(raw["deltas"][0]["weight"]), "NaN", 1)
    nan_uri = f"fake://parser-adapters/fake-lora/{sha256_text(nan_raw)}"
    store.put_raw(nan_uri, nan_raw)
    with pytest.raises(ValidationError, match="non-standard JSON constant"):
        store.load(nan_uri)


def test_fake_runtime_load_unload_is_idempotent(
    store: FakeAdapterStore, examples: tuple[ParserExample, ...]
) -> None:
    adapter = FakeLoraAdapter(
        dataset_hash=DATASET_HASH,
        deltas=FakeLoraTrainingBackend().train(examples, model="m", seed=1, params={}),
    )
    uri = store.save(adapter, prefix="fake://parser-adapters")
    runtime = FakeVllmRuntime(store)
    handle = runtime.load_adapter(uri)
    assert runtime.load_adapter(uri) == handle
    assert runtime.stats.loads == 1
    assert runtime.loaded_handles == (handle,)

    prompt = f"Context: {{}}\nUtterance: {examples[0].source_text}"
    assert json.loads(runtime.generate(handle, prompt)) == examples[0].slots
    runtime.unload_adapter(handle)
    runtime.unload_adapter(handle)
    assert runtime.stats.unloads == 1
    assert runtime.loaded_handles == ()
    with pytest.raises(ValidationError, match="not loaded"):
        runtime.generate(handle, prompt)


def test_fake_runtime_rejects_invalid_artifact(
    store: FakeAdapterStore, examples: tuple[ParserExample, ...]
) -> None:
    adapter = FakeLoraAdapter(
        dataset_hash=DATASET_HASH,
        deltas=FakeLoraTrainingBackend().train(examples, model="m", seed=1, params={}),
    )
    uri = store.save(adapter, prefix="fake://parser-adapters")
    raw = json.loads(canonical_json(adapter.to_dict()))
    raw["deltas"][0]["slots"] = {}
    store.put_raw(uri, canonical_json(raw))
    runtime = FakeVllmRuntime(store)
    with pytest.raises(ValidationError):
        runtime.load_adapter(uri)
    assert runtime.loaded_handles == ()
