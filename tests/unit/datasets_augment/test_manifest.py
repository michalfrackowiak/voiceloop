"""Internal manifest, tagging, exclusion and safe-write tests for P3-03.3."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from services.datasets.augment.cli import AugmentationRun
from services.datasets.augment.manifest import (
    AugmentedVariant,
    TransformRecord,
    assert_source_hashes_unchanged,
    build_manifest,
    load_dataset_manifest,
    snapshot_source_hashes,
    validate_manifest,
    write_manifest,
)
from services.datasets.augment.settings import (
    ARTIFACT_TYPE,
    AUGMENTATION_SCHEMA_VERSION,
    STAND_IN_LABEL,
    AugmentSettings,
)
from services.datasets.augment.variants import AudioStandIn, VariantSpec
from services.datasets.manifest import canonical_json
from voiceloop_common import ConfigError, ValidationError


def test_manifest_tags_every_variant_and_proves_exclusion(
    run: AugmentationRun,
    golden_ids: tuple[str, ...],
    regression_ids: tuple[str, ...],
) -> None:
    manifest = run.manifest
    assert manifest["artifact_type"] == ARTIFACT_TYPE
    assert manifest["augmentation_schema_version"] == AUGMENTATION_SCHEMA_VERSION
    assert manifest["synthetic"] is True
    assert manifest["stand_in_label"] == STAND_IN_LABEL
    assert manifest["promotion_decision"] == "not_evaluated"
    assert manifest["source_call_ids"] == sorted(set(run.eligible_calls))
    assert not set(manifest["source_call_ids"]) & (set(golden_ids) | set(regression_ids))
    assert all(record["synthetic"] is True for record in manifest["variants"])
    assert all(record["audio"]["label"] == STAND_IN_LABEL for record in manifest["variants"])
    assert set(manifest["transform_names"]) == {"fake_tts", "line_noise", "telephony_band"}
    proof = manifest["evaluation_set_exclusion"]
    assert proof["golden_call_ids"] == list(golden_ids)
    assert proof["regression_call_ids"] == list(regression_ids)
    assert proof["synthetic_variants_in_golden"] == 0
    assert proof["synthetic_variants_in_regression"] == 0
    payload = {key: value for key, value in manifest.items() if key != "content_hash"}
    assert manifest["content_hash"] == hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def test_manifest_hash_is_stable_for_batch_order_but_not_content(
    run: AugmentationRun,
    source_manifest: dict[str, Any],
    golden_ids: tuple[str, ...],
    regression_ids: tuple[str, ...],
) -> None:
    reversed_batch = build_manifest(
        tuple(reversed(run.variants)),
        source_manifest=source_manifest,
        golden_call_ids=golden_ids,
        regression_call_ids=regression_ids,
        created_at="2026-11-25T00:00:00Z",
    )
    assert reversed_batch["content_hash"] == run.manifest["content_hash"]
    tampered = copy.deepcopy(run.manifest)
    tampered["variants"][0]["audio"]["digest"] = "0" * 64
    with pytest.raises(ValidationError):
        validate_manifest(tampered)
    changed_time = build_manifest(
        run.variants,
        source_manifest=source_manifest,
        golden_call_ids=golden_ids,
        regression_call_ids=regression_ids,
        created_at="2026-11-26T00:00:00Z",
    )
    assert changed_time["content_hash"] != run.manifest["content_hash"]


def test_all_excluded_calls_yield_empty_synthetic_batch(
    settings: AugmentSettings,
    source_manifest: dict[str, Any],
) -> None:
    excluded = tuple(source_manifest["source_call_ids"])
    run = build_augmentation_for_test(settings, source_manifest, excluded)
    assert run.eligible_calls == ()
    assert run.manifest["variants"] == []
    assert run.manifest["source_call_ids"] == []
    validate_manifest(run.manifest)


def build_augmentation_for_test(
    settings: AugmentSettings,
    source_manifest: dict[str, Any],
    golden: tuple[str, ...],
) -> AugmentationRun:
    """Build a batch with all calls excluded, using the public CLI seam."""
    from services.datasets.augment.cli import build_augmentation

    return build_augmentation(
        settings,
        dataset_manifest=source_manifest,
        golden_call_ids=golden,
        regression_call_ids=(),
        created_at="2026-11-25T00:00:00Z",
    )


def test_leaking_an_excluded_call_fails_closed(
    run: AugmentationRun,
    source_manifest: dict[str, Any],
    golden_ids: tuple[str, ...],
    regression_ids: tuple[str, ...],
) -> None:
    original = run.variants[0]
    leak = replace(original, spec=replace(original.spec, source_call_id=golden_ids[0]))
    with pytest.raises(ValidationError, match="leaked"):
        build_manifest(
            (leak,),
            source_manifest=source_manifest,
            golden_call_ids=golden_ids,
            regression_call_ids=regression_ids,
        )


def test_manifest_rejects_pii_like_text(
    settings: AugmentSettings,
    source_manifest: dict[str, Any],
) -> None:
    spec = VariantSpec(
        variant_id="syn-pii",
        failure_class="123456789012345678",
        source_call_id="call-0004",
        token="token-1",
        sample_rate_hz=settings.sample_rate_hz,
        duration_ms=settings.duration_ms,
        seed=1,
        telephony=False,
        noisy=False,
    )
    variant = AugmentedVariant(
        spec=spec,
        audio=AudioStandIn((1,) * 16, settings.sample_rate_hz, settings.duration_ms),
        transforms=(TransformRecord("fake_tts", {"token_digest": spec.token_digest}),),
    )
    with pytest.raises(ValidationError, match="names, emails, phones or cards"):
        build_manifest(
            (variant,),
            source_manifest=source_manifest,
            golden_call_ids=(),
            regression_call_ids=(),
        )


def test_hash_anchors_are_checked_before_and_after(run: AugmentationRun) -> None:
    assert run.source_hashes_before == run.source_hashes_after
    assert run.source_hashes_before.golden_content_hash == (
        "662cdaf43ee2d5fde03faf5e69fde196114d4f312406d94c00bf53714d03da8e"
    )
    assert run.source_hashes_before.fixtures_sha256 == (
        "aca23a303954d035245a97f78319df3694d63611b717a2ff2131bc2e0a4453fd"
    )
    assert assert_source_hashes_unchanged(AugmentSettings(), run.source_hashes_before) is not None


def test_tampered_anchor_copies_are_rejected(tmp_path: Path, settings: AugmentSettings) -> None:
    golden_copy = tmp_path / "golden.v1.json"
    golden_copy.write_text(json.dumps({"content_hash": "0" * 64}), encoding="utf-8")
    bad_golden = settings.model_copy(update={"golden_manifest_path": str(golden_copy)})
    with pytest.raises(ValidationError, match="golden content hash"):
        snapshot_source_hashes(bad_golden)
    fixture_copy = tmp_path / "turns.jsonl"
    fixture_copy.write_text("changed\n", encoding="utf-8")
    bad_fixtures = settings.model_copy(update={"fixtures_path": str(fixture_copy)})
    with pytest.raises(ValidationError, match="fixture manifest hash"):
        snapshot_source_hashes(bad_fixtures)


def test_write_manifest_is_safe_and_rejects_readonly_targets(
    run: AugmentationRun,
    settings: AugmentSettings,
    tmp_path: Path,
) -> None:
    destination = write_manifest(
        run.manifest, tmp_path / "nested" / "manifest.json", settings=settings
    )
    decoded = json.loads(destination.read_text(encoding="utf-8"))
    assert decoded == run.manifest
    assert "samples" not in json.dumps(decoded)
    with pytest.raises(ValidationError, match="read-only inputs"):
        write_manifest(run.manifest, "tests/fixtures/synthetic.json", settings=settings)
    with pytest.raises(ValidationError, match="read-only inputs"):
        write_manifest(
            run.manifest, "services/datasets/golden/artifacts/synthetic.json", settings=settings
        )
    guarded_settings = settings.model_copy(update={"dataset_manifest_path": "artifacts/p2-05.json"})
    with pytest.raises(ValidationError, match="read-only inputs"):
        write_manifest(run.manifest, "artifacts/p2-05.json", settings=guarded_settings)


def test_manifest_input_validation_fails_closed(
    settings: AugmentSettings,
    tmp_path: Path,
) -> None:
    with pytest.raises(ConfigError, match="manifest path is required"):
        load_dataset_manifest(settings.model_copy(update={"dataset_manifest_path": ""}))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": "1.0.0"}), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_dataset_manifest(settings, path=bad)
