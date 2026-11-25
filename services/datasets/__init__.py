"""Dataset builder, splitting, regression, replay and exporters (P2-05)."""

from __future__ import annotations

from .builder import (
    build_dataset,
    build_dataset_from_fixtures,
    load_golden_call_ids,
    load_reviewed_items_from_fixtures,
)
from .examples import (
    REVIEW_ROUTINGS,
    DatasetExample,
    ExampleKind,
    ReviewedItem,
    build_dialogue_context,
    expand_reviewed_item,
)
from .exporters import (
    ExportResult,
    export_all,
    export_hf,
    export_nemo,
    export_sft,
)
from .manifest import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    DatasetBuild,
    FilterCounts,
    build_manifest,
    content_hash,
    dropped_as_dicts,
)
from .regression import (
    REVIEW_FAILURE_LABELS,
    RegressionRecord,
    RegressionStore,
    is_reviewed_failure,
)
from .replay import ReplaySelection, is_verified_replay_turn, sample_replay
from .settings import DatasetSettings
from .splits import (
    SplitPlan,
    assert_no_call_in_multiple_splits,
    assign_calls,
)

__all__ = [
    "REVIEW_FAILURE_LABELS",
    "REVIEW_ROUTINGS",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "DatasetBuild",
    "DatasetExample",
    "DatasetSettings",
    "ExampleKind",
    "ExportResult",
    "FilterCounts",
    "RegressionRecord",
    "RegressionStore",
    "ReplaySelection",
    "ReviewedItem",
    "SplitPlan",
    "assert_no_call_in_multiple_splits",
    "assign_calls",
    "build_dialogue_context",
    "build_dataset",
    "build_dataset_from_fixtures",
    "build_manifest",
    "content_hash",
    "dropped_as_dicts",
    "expand_reviewed_item",
    "export_all",
    "export_hf",
    "export_nemo",
    "export_sft",
    "is_reviewed_failure",
    "is_verified_replay_turn",
    "load_golden_call_ids",
    "load_reviewed_items_from_fixtures",
    "sample_replay",
]
