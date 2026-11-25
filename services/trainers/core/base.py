"""Trainer plugin ABC and helpers for contract-shaped results.

The plugin method takes typed ``TrainerSettings`` as its config object (the
runner derives the schema's ``config`` object from those settings), which
reconciles ``context.md`` §13's ``train(dataset_manifest, config)`` with the
typed-settings convention in ``docs/conventions.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Literal, TypedDict

from voiceloop_common import ValidationError, new_id, validate

from .settings import TrainerSettings

TRAINER_CONTRACT_SCHEMA = "trainer_contract.schema.json"
TRAINER_CONTRACT_VERSION = "1.0.0"


class TrainerLineage(TypedDict):
    """Provenance fields required by context.md §2 invariant 9."""

    dataset_hash: str
    git_sha: str
    image_digest: str


class TrainerResult(TypedDict):
    """Contract 1.0.0 TrainResponse payload returned by every plugin."""

    schema_version: str
    contract_part: Literal["output"]
    artifact_uri: str
    metrics: dict[str, float]
    run_id: str
    lineage: TrainerLineage


class TrainerPlugin(ABC):
    """Base class for all VoiceLoop trainer plugins (P2-06 interface)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stable registered plugin name, e.g. ``parser-lora``."""

    @abstractmethod
    def train(self, manifest: Mapping[str, Any], settings: TrainerSettings) -> TrainerResult:
        """Train a model and return a contract-valid output payload.

        Args:
            manifest: Dataset manifest with a non-empty ``content_hash``.
            settings: Typed job settings; the plugin's config source. Model
                selection must come from here, never from a hard-coded name.

        Returns:
            A :class:`TrainerResult` whose ``artifact_uri`` addresses the
            deployable artifact (merged adapter or converted model), not a raw
            checkpoint. Use :func:`build_trainer_result` to fill lineage.

        """


def build_trainer_result(
    manifest: Mapping[str, Any],
    settings: TrainerSettings,
    *,
    artifact_uri: str,
    metrics: Mapping[str, float] | None = None,
    run_id: str | None = None,
) -> TrainerResult:
    """Compose a contract-valid result with complete lineage.

    Args:
        manifest: Dataset manifest; its ``content_hash`` becomes lineage.
        settings: Typed settings supplying git SHA and pinned image digest.
        artifact_uri: URI of the deployable artifact, not a training checkpoint.
        metrics: Numeric training-time metrics keyed by name.
        run_id: Optional caller-supplied run ID; generated when omitted.

    Returns:
        A validated 1.0.0 TrainResponse payload including all lineage fields.

    Raises:
        ValidationError: If ``content_hash`` is missing or the composed result
            does not satisfy the trainer contract.

    """
    content_hash = manifest.get("content_hash")
    if not isinstance(content_hash, str) or not content_hash:
        raise ValidationError("dataset_manifest.content_hash must be a non-empty string")
    result: TrainerResult = {
        "schema_version": TRAINER_CONTRACT_VERSION,
        "contract_part": "output",
        "artifact_uri": artifact_uri,
        "metrics": dict(metrics or {}),
        "run_id": run_id or new_id(),
        "lineage": {
            "dataset_hash": content_hash,
            "git_sha": settings.git_sha,
            "image_digest": settings.image_digest,
        },
    }
    validate(result, TRAINER_CONTRACT_SCHEMA)
    return result
