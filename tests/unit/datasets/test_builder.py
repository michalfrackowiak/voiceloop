"""End-to-end dataset builder tests: exclusions, replay and lineage."""

from __future__ import annotations

from services.datasets import (
    DatasetSettings,
    RegressionStore,
    assert_no_call_in_multiple_splits,
    build_dataset,
    build_dataset_from_fixtures,
    build_manifest,
    expand_reviewed_item,
    load_golden_call_ids,
    load_reviewed_items_from_fixtures,
    sample_replay,
)
from voiceloop_common import validate


def test_exclusion_applies_to_golden_and_regression_calls() -> None:
    """Golden and regression calls never contribute training examples."""
    build = build_dataset_from_fixtures(
        golden_call_ids=["call-0003"],
        regression_call_ids=["call-0002"],
    )
    assert_no_call_in_multiple_splits(build.split_plan)
    assert build.split_plan.golden == ("call-0003",)
    assert build.split_plan.regression == ("call-0002",)
    assigned = build.split_plan.assigned_calls()
    assert "call-0002" not in assigned
    assert "call-0003" not in assigned
    assert all(example.call_id in assigned for example in build.examples)
    assert build.filter_counts.golden_excluded > 0
    assert build.filter_counts.regression_excluded > 0
    manifest = build_manifest(build)
    validate(manifest, "dataset_manifest")
    assert set(manifest["source_call_ids"]) == assigned


def test_replay_pool_mixes_earlier_verified_turns() -> None:
    """Seeded replay adds examples from an earlier verified pool without duplicates."""
    all_items = load_reviewed_items_from_fixtures()
    reviewed = [item for item in all_items if item.call_id == "call-0001"]
    pool = [
        item
        for item in all_items
        if item.call_id != "call-0001" and item.failure_type in ("clean", "both_fine")
    ]
    settings = DatasetSettings(replay_size=2, replay_seed=7)
    selection = sample_replay(pool, size=settings.replay_size, seed=settings.replay_seed)
    expected = sum(len(expand_reviewed_item(item)) for item in selection.items)

    build = build_dataset(reviewed, settings=settings, replay_pool=pool)
    replayed = [example for example in build.examples if example.replay]
    assert build.filter_counts.replay_added == len(replayed) == expected
    assert replayed
    assert all(example.call_id != "call-0001" for example in replayed)
    assert len({example.example_id for example in build.examples}) == len(build.examples)


def test_regression_store_call_ids_are_excluded_from_build() -> None:
    """Accumulated fixture failures drive the exclusion list end to end."""
    items = load_reviewed_items_from_fixtures()
    store = RegressionStore()
    store.accumulate(items)
    build = build_dataset(items, regression_call_ids=store.call_ids())
    assert build.split_plan.regression == store.call_ids()
    assert set(build.split_plan.assigned_calls()).isdisjoint(store.call_ids())


def test_load_golden_call_ids_uses_sealed_manifest() -> None:
    """Golden call IDs come from the P1-06 public API, with seal verification."""
    ids = load_golden_call_ids()
    assert ids == tuple(f"call-{number:04d}" for number in range(1, 7))
    assert load_golden_call_ids(verify=False) == ids
    build = build_dataset_from_fixtures(golden_call_ids=ids)
    assert build.split_plan.assigned_calls() == set()
    assert build.examples == ()
