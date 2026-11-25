"""Tests for manifest contract validation, hashing and audit counts."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from services.datasets import (
    build_dataset_from_fixtures,
    build_manifest,
    content_hash,
)
from voiceloop_common import ValidationError, frozen_clock, use_clock, validate

FROZEN = datetime(2026, 11, 25, 0, 0, 0, tzinfo=UTC)


def test_manifest_validates_against_schema_1_0_0() -> None:
    """voiceloop_common.validate accepts the emitted manifest."""
    build = build_dataset_from_fixtures()
    manifest = build_manifest(build)
    validate(manifest, "dataset_manifest")
    validate(manifest, "dataset_manifest.schema.json")
    assert manifest["schema_version"] == "1.0.0"
    assert manifest["examples"] == len(build.examples)
    assert set(manifest) == {
        "schema_version",
        "dataset_id",
        "content_hash",
        "source_call_ids",
        "splits",
        "filter_counts",
        "created_at",
        "examples",
        "synthetic",
    }


def test_manifest_splits_and_filter_counts_are_consistent() -> None:
    """Source calls equal the split union and all counts are audited."""
    build = build_dataset_from_fixtures()
    manifest = build_manifest(build)
    splits = manifest["splits"]
    assigned = splits["train"] + splits["val"] + splits["test"]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == set(manifest["source_call_ids"])
    assert manifest["filter_counts"] == build.filter_counts.to_dict()
    assert all(count >= 0 for count in manifest["filter_counts"].values())
    assert manifest["filter_counts"]["consent_dropped"] == 1
    assert manifest["synthetic"] is True


def test_content_hash_is_order_independent_and_content_sensitive() -> None:
    """Reordering examples leaves the hash unchanged; editing content changes it."""
    build = build_dataset_from_fixtures()
    forward = content_hash(build.examples)
    reversed_hash = content_hash(tuple(reversed(build.examples)))
    assert forward == reversed_hash == build_manifest(build)["content_hash"]
    assert build_manifest(build)["dataset_id"].endswith(forward[:12])
    changed = tuple(
        type(example)(**{**example.to_dict(), "source_text": "altered"}) if index == 0 else example
        for index, example in enumerate(build.examples)
    )
    assert content_hash(changed) != forward
    different_audio = tuple(
        type(example)(**{**example.to_dict(), "audio_uri": "different.wav"})
        if index == 0
        else example
        for index, example in enumerate(build.examples)
    )
    assert content_hash(different_audio) != forward


def test_created_at_uses_injectable_utc_clock() -> None:
    """created_at is generated with voiceloop_common.utcnow()."""
    build = build_dataset_from_fixtures()
    with use_clock(frozen_clock(FROZEN)):
        manifest = build_manifest(build)
    assert manifest["created_at"] == "2026-11-25T00:00:00Z"


def test_manifest_rejects_extra_properties() -> None:
    """The manifest contract still rejects unknown fields."""
    manifest = dict(build_manifest(build_dataset_from_fixtures()))
    manifest["unexpected"] = True
    with pytest.raises(ValidationError, match="Validation failed"):
        validate(manifest, "dataset_manifest")
