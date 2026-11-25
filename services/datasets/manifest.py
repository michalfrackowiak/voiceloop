"""Dataset manifest construction and content hashing (P2-05.5).

The emitted object is validated with ``voiceloop_common.validate`` against
``schemas/dataset_manifest.schema.json`` 1.0.0 before it is returned. It carries
a deterministic content hash, source call IDs, call-level splits and audited
filter counts. No schema file is changed by this task.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from voiceloop_common import utcnow, validate

from .examples import DatasetExample
from .filters import DroppedEvent
from .splits import SplitPlan

SCHEMA_VERSION = "1.0.0"
SCHEMA_NAME = "dataset_manifest"


@dataclass(frozen=True, slots=True)
class FilterCounts:
    """Audited row counts for each dataset rule in the manifest contract."""

    consent_dropped: int = 0
    pii_dropped: int = 0
    replay_added: int = 0
    golden_excluded: int = 0
    regression_excluded: int = 0
    synthetic_tagged: int = 0

    def to_dict(self) -> dict[str, int]:
        """Return the exact ``filter_counts`` object required by the schema."""
        return {
            "consent_dropped": self.consent_dropped,
            "pii_dropped": self.pii_dropped,
            "replay_added": self.replay_added,
            "golden_excluded": self.golden_excluded,
            "regression_excluded": self.regression_excluded,
            "synthetic_tagged": self.synthetic_tagged,
        }


@dataclass(frozen=True, slots=True)
class DatasetBuild:
    """Everything the manifest and exporters need from one dataset build."""

    examples: tuple[DatasetExample, ...]
    split_plan: SplitPlan
    filter_counts: FilterCounts
    dropped: tuple[DroppedEvent, ...] = ()


def canonical_json(value: object) -> str:
    """Return stable compact JSON text for hashing and deterministic writes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(examples: Sequence[DatasetExample]) -> str:
    """Hash the sorted example IDs and their full training contents.

    Reordering examples, mutating a transcript, slot, context or split source
    changes the hash; inputs are sorted by ``example_id`` first so batch order
    cannot affect it.

    Args:
        examples: All training examples in the build.

    Returns:
        Lowercase SHA-256 hex digest.

    """
    payload = {
        "examples": [
            example.to_dict() for example in sorted(examples, key=lambda item: item.example_id)
        ]
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def build_manifest(
    build: DatasetBuild,
    *,
    created_at: str | None = None,
    dataset_id_prefix: str = "voiceloop-dataset",
) -> dict[str, Any]:
    """Build and schema-validate a dataset manifest for one dataset build.

    Args:
        build: Examples, split plan and filter counts from :func:`build_dataset`.
        created_at: Optional UTC timestamp for deterministic tests; defaults to
            ``voiceloop_common.utcnow()``.
        dataset_id_prefix: Stable prefix combined with the content hash.

    Returns:
        A JSON-serialisable manifest object conforming to schema version 1.0.0.

    """
    digest = content_hash(build.examples)
    timestamp = created_at or utcnow().isoformat(timespec="seconds").replace("+00:00", "Z")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": f"{dataset_id_prefix}-{digest[:12]}",
        "content_hash": digest,
        "source_call_ids": sorted({example.call_id for example in build.examples}),
        "splits": build.split_plan.to_manifest_splits(),
        "filter_counts": build.filter_counts.to_dict(),
        "created_at": timestamp,
        "examples": len(build.examples),
        "synthetic": any(example.synthetic for example in build.examples),
    }
    validate(manifest, SCHEMA_NAME)
    return manifest


def dropped_as_dicts(dropped: Sequence[DroppedEvent]) -> list[dict[str, object]]:
    """Return PII-free drop records for audit logging alongside the manifest."""
    return [
        {
            "event_id": record.event_id,
            "call_id": record.call_id,
            "turn_index": record.turn_index,
            "audio_uri": record.audio_uri,
            "reasons": [str(reason) for reason in record.reasons],
            "pii_findings": [finding.as_dict() for finding in record.pii_findings],
        }
        for record in dropped
    ]


def filter_counts_from_dict(data: Mapping[str, int]) -> FilterCounts:
    """Rebuild filter counts from a decoded manifest object."""
    return FilterCounts(**{field: int(data[field]) for field in asdict(FilterCounts())})
