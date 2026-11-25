"""Tests for the deterministic Parakeet/NeMo stand-in fine-tune."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from services.trainers.core import TrainerRegistry, TrainerRunner, TrainerSettings
from services.trainers.nemo import (
    PROVENANCE_LABEL,
    FakeArtifactStore,
    FakeNeMoArtifact,
    FakeNeMoFinetuneBackend,
    NemoManifestRow,
    NemoSettings,
    ParakeetNeMoTrainer,
    canonical_json,
)
from voiceloop_common import ConfigError, ValidationError

from .conftest import CALL_IDS, DATASET_HASH, EVENT_IDS

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_MANIFEST = REPO_ROOT / "tests/fixtures/turns.jsonl"
GOLDEN_SEAL = REPO_ROOT / "services/datasets/golden/artifacts/golden.v1.seal.json"
EXPECTED_FIXTURE_HASH = "aca23a303954d035245a97f78319df3694d63611b717a2ff2131bc2e0a4453fd"
EXPECTED_GOLDEN_HASH = "662cdaf43ee2d5fde03faf5e69fde196114d4f312406d94c00bf53714d03da8e"


def _protected_hashes() -> tuple[str, str]:
    """Return hashes of the read-only fixture manifest and sealed golden stand-in."""
    fixture_hash = hashlib.sha256(FIXTURE_MANIFEST.read_bytes()).hexdigest()
    seal = json.loads(GOLDEN_SEAL.read_text(encoding="utf-8"))
    return fixture_hash, str(seal["content_hash"])


def _trainer(settings: NemoSettings, store: FakeArtifactStore) -> ParakeetNeMoTrainer:
    return ParakeetNeMoTrainer(settings=settings, store=store)


def test_fixture_finetune_runs_end_to_end(
    manifest: dict[str, Any],
    nemo_settings: NemoSettings,
    core_settings: TrainerSettings,
    rows: tuple[NemoManifestRow, ...],
) -> None:
    store = FakeArtifactStore()
    registry = TrainerRegistry()
    registry.register(_trainer(nemo_settings, store))
    result = TrainerRunner(registry, settings=core_settings).run("parakeet-nemo", manifest)

    assert result["artifact_uri"].startswith("fake://nemo-artifacts/fake-nemo-finetune/")
    assert result["metrics"]["examples_total"] == float(len(rows))
    assert result["metrics"]["fake_train_loss"] == 0.0
    assert result["lineage"] == {
        "dataset_hash": DATASET_HASH,
        "git_sha": core_settings.git_sha,
        "image_digest": core_settings.image_digest,
    }
    artifact = store.load(result["artifact_uri"])
    assert artifact.provenance_label == PROVENANCE_LABEL
    assert artifact.source_rows == len(rows)
    assert artifact.lineage == result["lineage"]
    assert artifact.base_model == core_settings.model
    assert artifact.base_checkpoint == store.load(result["artifact_uri"]).base_checkpoint
    assert artifact.content_hash() == result["artifact_uri"].rsplit("/", 1)[-1]

    raw = canonical_json(artifact.to_dict())
    for row in rows:
        assert row.text not in raw
        assert row.audio_filepath not in raw


def test_artifact_is_deterministic_and_content_sensitive(
    manifest: dict[str, Any],
    nemo_settings: NemoSettings,
    core_settings: TrainerSettings,
    rows: tuple[NemoManifestRow, ...],
) -> None:
    first_store = FakeArtifactStore()
    second_store = FakeArtifactStore()
    first = _trainer(nemo_settings, first_store).train(manifest, core_settings)
    reversed_source = ParakeetNeMoTrainer(
        settings=nemo_settings,
        source=_InMemorySource(tuple(reversed(rows))),
        store=second_store,
    )
    second = reversed_source.train(manifest, core_settings)
    assert first["artifact_uri"] == second["artifact_uri"]

    changed_seed = TrainerSettings(
        model=core_settings.model,
        seed=core_settings.seed + 1,
        artifact_uri_prefix=core_settings.artifact_uri_prefix,
        git_sha=core_settings.git_sha,
        image_digest=core_settings.image_digest,
    )
    third_store = FakeArtifactStore()
    third = _trainer(nemo_settings, third_store).train(manifest, changed_seed)
    assert third["artifact_uri"] != first["artifact_uri"]


def test_real_backend_requires_injection(
    manifest: dict[str, Any],
    core_settings: TrainerSettings,
    rows: tuple[NemoManifestRow, ...],
) -> None:
    settings = NemoSettings(backend="nemo", checkpoint_uri="fake://checkpoint/parakeet")
    with pytest.raises(ConfigError, match="injected"):
        ParakeetNeMoTrainer(settings=settings)
    trainer = ParakeetNeMoTrainer(
        settings=settings,
        source=_InMemorySource(rows),
        backend=FakeNeMoFinetuneBackend(),
    )
    result = trainer.train(manifest, core_settings)
    assert result["artifact_uri"].startswith("fake://nemo-artifacts/fake-nemo-finetune/")


def test_empty_manifest_rows_fail_closed(
    manifest: dict[str, Any],
    core_settings: TrainerSettings,
) -> None:
    trainer = ParakeetNeMoTrainer(settings=NemoSettings(), source=_InMemorySource(()))
    with pytest.raises(ValidationError, match="at least one"):
        trainer.train(manifest, core_settings)


def test_tampered_artifact_is_rejected(
    manifest: dict[str, Any],
    nemo_settings: NemoSettings,
    core_settings: TrainerSettings,
) -> None:
    store = FakeArtifactStore()
    result = _trainer(nemo_settings, store).train(manifest, core_settings)
    uri = result["artifact_uri"]
    artifact = store.load(uri)
    tampered = artifact.to_dict()
    tampered["source_rows"] += 1
    store.put_raw(uri, canonical_json(tampered))
    with pytest.raises(ValidationError, match="content hash"):
        store.load(uri)


def test_artifact_loader_rejects_pii_leaking_fields() -> None:
    artifact = FakeNeMoArtifact(
        dataset_hash=DATASET_HASH,
        lineage={
            "dataset_hash": DATASET_HASH,
            "git_sha": "b" * 40,
            "image_digest": "registry.example/nemo@sha256:" + "a" * 64,
        },
        base_model="fixture-parakeet-base",
        base_checkpoint="c" * 64,
        seed=7,
        params={"epochs": 1},
        deltas=(),
        source_rows=1,
    )
    payload = artifact.to_dict()
    payload["deltas"] = [{"text": "Ada Fixture"}]
    with pytest.raises(ValidationError):
        FakeNeMoArtifact.from_dict(payload)


def test_confidence_emissions_from_trained_rows(
    manifest: dict[str, Any],
    nemo_settings: NemoSettings,
    core_settings: TrainerSettings,
    rows: tuple[NemoManifestRow, ...],
) -> None:
    trainer = ParakeetNeMoTrainer(settings=nemo_settings)
    trainer.train(manifest, core_settings)
    emissions = trainer.emit_confidences(
        rows,
        core_settings,
        identifiers=list(zip(EVENT_IDS, CALL_IDS, range(len(rows)), strict=True)),
    )
    assert len(emissions) == len(rows)
    assert all(emission.event["stt"]["words"] for emission in emissions)
    assert all(emission.score.signal_names for emission in emissions)


def test_emission_count_must_match_rows(
    nemo_settings: NemoSettings,
    core_settings: TrainerSettings,
    rows: tuple[NemoManifestRow, ...],
) -> None:
    with pytest.raises(ValidationError, match="identifier"):
        ParakeetNeMoTrainer(settings=nemo_settings).emit_confidences(
            rows,
            core_settings,
            identifiers=[(EVENT_IDS[0], CALL_IDS[0], 0)],
        )


def test_protected_hashes_are_unchanged_before_and_after(
    manifest: dict[str, Any],
    nemo_settings: NemoSettings,
    core_settings: TrainerSettings,
) -> None:
    before = _protected_hashes()
    assert before == (EXPECTED_FIXTURE_HASH, EXPECTED_GOLDEN_HASH)
    trainer = ParakeetNeMoTrainer(settings=nemo_settings)
    trainer.train(manifest, core_settings)
    assert _protected_hashes() == before


class _InMemorySource:
    """Tiny injected manifest source for failure-injection tests."""

    def __init__(self, rows: tuple[NemoManifestRow, ...]) -> None:
        self._rows = rows

    def load_rows(self, manifest: Mapping[str, Any]) -> tuple[NemoManifestRow, ...]:
        return self._rows
