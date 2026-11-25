"""Parser trainer contract, determinism, dataset seam and optional GLiNER2 tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from services.trainers.core import (
    FakeMlflowLogger,
    TrainerRegistry,
    TrainerRunner,
)
from services.trainers.parser import (
    FakeAdapterStore,
    FakeLoraTrainingBackend,
    Gliner2SlotTrainer,
    ParserLoraTrainer,
    ParserSettings,
    SftJsonlExampleLoader,
)
from tests.unit.trainers_parser.conftest import DATASET_HASH, make_example, make_slots
from voiceloop_common import ConfigError, ValidationError, validate


def test_runner_end_to_end_returns_contract_valid_result(
    manifest: dict[str, Any], core_settings: Any, examples: tuple[Any, ...]
) -> None:
    registry = TrainerRegistry()
    registry.register(ParserLoraTrainer(examples=examples))
    logger = FakeMlflowLogger()
    result = TrainerRunner(registry, settings=core_settings, logger=logger).run(
        "parser-lora", manifest
    )
    validate(result, "trainer_contract.schema.json")
    assert result["artifact_uri"].startswith("fake://parser-adapters/fake-lora/")
    assert result["metrics"]["examples_gold"] == 2
    assert result["metrics"]["examples_noisy"] == 2
    assert result["lineage"]["dataset_hash"] == DATASET_HASH
    assert result["lineage"]["git_sha"] == core_settings.git_sha
    assert len(logger.runs) == 1
    assert logger.runs[0].adapter_uri == result["artifact_uri"]


def test_export_is_deterministic_and_labels_standin(
    manifest: dict[str, Any], core_settings: Any, examples: tuple[Any, ...]
) -> None:
    store = FakeAdapterStore()
    first = ParserLoraTrainer(examples=examples, store=store).train(manifest, core_settings)
    second = ParserLoraTrainer(examples=tuple(reversed(examples)), store=store).train(
        manifest, core_settings
    )
    assert first["artifact_uri"] == second["artifact_uri"]
    assert first["run_id"] != second["run_id"]
    adapter = store.load(first["artifact_uri"])
    assert adapter.synthetic_training_standin is True
    assert adapter.dataset_hash == DATASET_HASH
    assert {delta.kind for delta in adapter.deltas} == {"gold", "noisy"}


def test_noisy_example_changes_payload_and_gold_only_is_rejected(
    manifest: dict[str, Any], core_settings: Any
) -> None:
    gold = make_example("ex-gold", "gold", "Check in on the twelfth.")
    noisy_a = make_example("ex-noisy-a", "noisy", "Check in on the twelf.")
    noisy_b = make_example("ex-noisy-b", "noisy", "Check in on the twelfth, maybe.")
    uri_a = ParserLoraTrainer(examples=(gold, noisy_a)).train(manifest, core_settings)[
        "artifact_uri"
    ]
    uri_b = ParserLoraTrainer(examples=(gold, noisy_b)).train(manifest, core_settings)[
        "artifact_uri"
    ]
    assert uri_a != uri_b
    with pytest.raises(ValidationError, match="both gold and noisy"):
        ParserLoraTrainer(examples=(gold,)).train(manifest, core_settings)


def test_excluded_call_and_invalid_slots_are_rejected(
    manifest: dict[str, Any], core_settings: Any
) -> None:
    example = make_example("ex-1", "gold", "Check in on the twelfth.", call_id="golden-call")
    noisy = make_example("ex-noisy", "noisy", "Check in on the twelf.", call_id="golden-call")
    settings = ParserSettings(exclude_call_ids=("golden-call",))
    with pytest.raises(ValidationError, match="excluded call"):
        ParserLoraTrainer(examples=(example, noisy), settings=settings).train(
            manifest, core_settings
        )

    bad = make_example("ex-2", "noisy", "Check out before check in.")
    bad.slots["check_out"] = "2026-12-01"
    with pytest.raises(ValidationError, match="invalid parser training target"):
        ParserLoraTrainer(examples=(example, bad)).train(manifest, core_settings)


def test_real_backend_label_requires_injection(
    manifest: dict[str, Any], core_settings: Any, examples: tuple[Any, ...]
) -> None:
    with pytest.raises(ConfigError, match="requires an injected SlotTrainingBackend"):
        ParserLoraTrainer(settings=ParserSettings(backend="unsloth"), examples=examples)
    trainer = ParserLoraTrainer(
        settings=ParserSettings(backend="trl"),
        examples=examples,
        backend=FakeLoraTrainingBackend(),
    )
    assert trainer.name == "parser-lora"
    assert trainer.train(manifest, core_settings)["artifact_uri"].startswith("fake://")


def test_missing_content_hash_is_rejected(core_settings: Any, examples: tuple[Any, ...]) -> None:
    with pytest.raises(ValidationError, match="content_hash"):
        ParserLoraTrainer(examples=examples).train({}, core_settings)


def test_sft_jsonl_loader_maps_kinds_and_excludes_golden(
    tmp_path: Path, manifest: dict[str, Any], core_settings: Any
) -> None:
    slots = json.dumps(make_slots(), sort_keys=True)
    rows = [
        {
            "example_id": "ex-gold",
            "call_id": "call-train",
            "kind": "parser_clean",
            "messages": [
                {"role": "system", "content": "extract slots"},
                {"role": "user", "content": "Context: {}\nUtterance: Check in on the twelfth."},
                {"role": "assistant", "content": slots},
            ],
            "synthetic": True,
        },
        {
            "example_id": "ex-noisy",
            "call_id": "call-train",
            "kind": "parser_noisy",
            "messages": [
                {"role": "user", "content": "Context: {}\nUtterance: Check in on the twelf."},
                {"role": "assistant", "content": slots},
            ],
            "synthetic": True,
        },
        {
            "example_id": "ex-golden",
            "call_id": "call-golden",
            "kind": "parser_clean",
            "messages": [
                {"role": "user", "content": "Context: {}\nUtterance: Never train on this."},
                {"role": "assistant", "content": slots},
            ],
            "synthetic": True,
        },
    ]
    path = tmp_path / "dataset.sft.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    loader = SftJsonlExampleLoader(path, exclude_call_ids=("call-golden",))
    examples = loader.load_examples(manifest)
    assert [example.kind for example in examples] == ["gold", "noisy"]
    assert all(example.call_id == "call-train" for example in examples)
    assert examples[0].slots == make_slots()

    result = ParserLoraTrainer(example_loader=loader).train(
        {**manifest, "sft_path": str(path)}, core_settings
    )
    assert result["metrics"]["examples_total"] == 2

    with pytest.raises(ValidationError, match="sft_path"):
        SftJsonlExampleLoader().load_examples(manifest)


def test_gliner2_plugin_is_optional_and_fails_closed(
    manifest: dict[str, Any], core_settings: Any, examples: tuple[Any, ...]
) -> None:
    with pytest.raises(ConfigError, match="GLiNER2 is optional"):
        Gliner2SlotTrainer(examples=examples).train(manifest, core_settings)
    store = FakeAdapterStore()
    result = Gliner2SlotTrainer(
        examples=examples, backend=FakeLoraTrainingBackend(), store=store
    ).train(manifest, core_settings)
    validate(result, "trainer_contract.schema.json")
    assert "/fake-gliner2/" in result["artifact_uri"]
    assert store.load(result["artifact_uri"]).synthetic_training_standin is True
