"""Parakeet/NeMo trainer plugin built entirely from deterministic fakes.

SYNTHETIC / FIXTURE STAND-IN - NOT A REAL NEMO FINE-TUNE. The plugin starts
from a fake checkpoint, consumes the read-only P2-05 NeMo manifest export and
exports a hash-addressed stand-in artifact containing only hashes of transcript
text and audio paths. Real NeMo/torch code, GPU jobs, network and MLflow writes
are injected through the Protocols below; nothing here imports NeMo or torch.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from services.trainers.core import (
    TrainerLineage,
    TrainerPlugin,
    TrainerResult,
    TrainerSettings,
    build_trainer_result,
)
from voiceloop_common import ConfigError, ValidationError

from .confidences import ConfidenceEmission, ConfidenceEmitter, TriageSignalSink
from .manifest import (
    JsonlNemoManifestSource,
    NemoManifestRow,
    NemoManifestSource,
    canonical_json,
    sha256_text,
)
from .settings import PROVENANCE_LABEL, NemoSettings

_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_PLUGIN_NAME = "parakeet-nemo"
_ARTIFACT_KIND = "fake-nemo-finetune"
_ARTIFACT_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class FakeCheckpoint:
    """Fake starting checkpoint identified only by URI and a content hash."""

    uri: str
    model_id: str
    checkpoint_hash: str
    provenance_label: str = PROVENANCE_LABEL

    @classmethod
    def from_uri(cls, uri: str, model_id: str) -> FakeCheckpoint:
        """Derive a deterministic fake checkpoint without opening any file.

        Args:
            uri: ``fake://`` checkpoint reference injected via settings.
            model_id: Base model id injected via core settings.

        Returns:
            A labelled fake checkpoint.

        Raises:
            ValidationError: If the URI is not a fake reference or model is empty.

        """
        if not isinstance(uri, str) or not uri.startswith("fake://"):
            raise ValidationError("only fake:// checkpoint URIs are available without a backend")
        if not isinstance(model_id, str) or not model_id:
            raise ValidationError("a non-empty base model is required for fine-tuning")
        digest = sha256_text(f"{uri}|{model_id}|{PROVENANCE_LABEL}")
        return cls(uri=uri, model_id=model_id, checkpoint_hash=digest)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable checkpoint descriptor."""
        return {
            "uri": self.uri,
            "model_id": self.model_id,
            "checkpoint_hash": self.checkpoint_hash,
            "provenance_label": self.provenance_label,
        }


class CheckpointSource(Protocol):
    """Injectable seam for loading a real fine-tune starting checkpoint."""

    def load(self, uri: str, *, model_id: str) -> FakeCheckpoint:
        """Return a labelled checkpoint descriptor for one URI and model."""
        ...


class FakeCheckpointSource:
    """Default source: deterministic fake checkpoints, no file or network IO."""

    def load(self, uri: str, *, model_id: str) -> FakeCheckpoint:
        """Resolve a fake checkpoint URI into a labelled descriptor."""
        return FakeCheckpoint.from_uri(uri, model_id)


@dataclass(frozen=True, slots=True)
class FakeDelta:
    """One pseudo-weight update; stores hashes, never transcript or audio path."""

    example_hash: str
    text_hash: str
    audio_path_hash: str
    duration_ms: int
    weight: float

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-serialisable delta."""
        return {
            "example_hash": self.example_hash,
            "text_hash": self.text_hash,
            "audio_path_hash": self.audio_path_hash,
            "duration_ms": self.duration_ms,
            "weight": self.weight,
        }


class NeMoFinetuneBackend(Protocol):
    """Injectable real-NeMo seam; production implementations own heavy imports."""

    def finetune(
        self,
        rows: Sequence[NemoManifestRow],
        *,
        checkpoint: FakeCheckpoint,
        model: str,
        seed: int,
        params: Mapping[str, Any],
    ) -> tuple[FakeDelta, ...]:
        """Fine-tune from ``checkpoint`` on P2-05 rows and return stand-in deltas."""
        ...


class FakeNeMoFinetuneBackend:
    """Deterministic pseudo-fine-tune: no GPU, weights, audio bytes or network."""

    def finetune(
        self,
        rows: Sequence[NemoManifestRow],
        *,
        checkpoint: FakeCheckpoint,
        model: str,
        seed: int,
        params: Mapping[str, Any],
    ) -> tuple[FakeDelta, ...]:
        """Derive reproducible pseudo-deltas from rows and injected settings.

        Args:
            rows: Validated rows from the read-only P2-05 export.
            checkpoint: Starting fake checkpoint descriptor.
            model: Base model id from core settings.
            seed: Core seed.
            params: NeMo hyperparameters from :meth:`NemoSettings.as_training_params`.

        Returns:
            Deltas sorted by row fingerprint, independent of input ordering.

        """
        prefix = "|".join(
            (checkpoint.checkpoint_hash, model, str(seed), canonical_json(dict(params)))
        )
        deltas = [
            FakeDelta(
                example_hash=row.fingerprint(),
                text_hash=sha256_text(row.text),
                audio_path_hash=sha256_text(row.audio_filepath),
                duration_ms=max(1, int(round(row.duration * 1000))),
                weight=round(
                    int(sha256_text(f"{prefix}|{row.fingerprint()}")[:8], 16) / 0xFFFFFFFF, 6
                ),
            )
            for row in rows
        ]
        return tuple(sorted(deltas, key=lambda delta: delta.example_hash))


@dataclass(frozen=True, slots=True)
class FakeNeMoArtifact:
    """Hash-addressed, labelled deployable stand-in artifact.

    ``schema_version`` is an internal fake artifact version (not a versioned
    repository contract). Lineage is populated only from injected values.
    """

    dataset_hash: str
    lineage: TrainerLineage
    base_model: str
    base_checkpoint: str
    seed: int
    params: dict[str, Any]
    deltas: tuple[FakeDelta, ...]
    source_rows: int
    schema_version: str = _ARTIFACT_SCHEMA_VERSION
    kind: str = _ARTIFACT_KIND
    provenance_label: str = PROVENANCE_LABEL

    def to_dict(self) -> dict[str, Any]:
        """Return canonical JSON data for export and hashing."""
        return {
            "base_checkpoint": self.base_checkpoint,
            "base_model": self.base_model,
            "dataset_hash": self.dataset_hash,
            "deltas": [delta.to_dict() for delta in self.deltas],
            "kind": self.kind,
            "lineage": dict(self.lineage),
            "params": self.params,
            "provenance_label": self.provenance_label,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "source_rows": self.source_rows,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FakeNeMoArtifact:
        """Validate a decoded artifact before it enters a loaded artifact store.

        Raises:
            ValidationError: If fields are missing, unexpected or tampered with.

        """
        expected = {
            "base_checkpoint",
            "base_model",
            "dataset_hash",
            "deltas",
            "kind",
            "lineage",
            "params",
            "provenance_label",
            "schema_version",
            "seed",
            "source_rows",
        }
        if set(raw) != expected:
            raise ValidationError("NeMo stand-in artifact has unexpected fields")
        if raw["schema_version"] != _ARTIFACT_SCHEMA_VERSION or raw["kind"] != _ARTIFACT_KIND:
            raise ValidationError("unsupported NeMo stand-in artifact type")
        if raw["provenance_label"] != PROVENANCE_LABEL:
            raise ValidationError("NeMo artifact is missing its stand-in provenance label")
        dataset_hash = raw["dataset_hash"]
        if not isinstance(dataset_hash, str) or not _DIGEST_RE.fullmatch(dataset_hash):
            raise ValidationError("NeMo artifact dataset_hash must be a sha256 hex digest")
        lineage = raw["lineage"]
        if not isinstance(lineage, Mapping) or set(lineage) != {
            "dataset_hash",
            "git_sha",
            "image_digest",
        }:
            raise ValidationError("NeMo artifact lineage must carry all three fields")
        if any(not isinstance(value, str) or not value for value in lineage.values()):
            raise ValidationError("NeMo artifact lineage values must be non-empty strings")
        if not isinstance(raw["base_model"], str) or isinstance(raw["seed"], bool):
            raise ValidationError("NeMo artifact model/seed fields are invalid")
        if not isinstance(raw["seed"], int) or not isinstance(raw["params"], dict):
            raise ValidationError("NeMo artifact seed/params fields are invalid")
        if isinstance(raw["source_rows"], bool) or not isinstance(raw["source_rows"], int):
            raise ValidationError("NeMo artifact source_rows must be an integer")
        raw_deltas = raw["deltas"]
        if not isinstance(raw_deltas, list) or not raw_deltas:
            raise ValidationError("NeMo artifact deltas must be a non-empty list")
        deltas: list[FakeDelta] = []
        for item in raw_deltas:
            if not isinstance(item, Mapping) or set(item) != {
                "example_hash",
                "text_hash",
                "audio_path_hash",
                "duration_ms",
                "weight",
            }:
                raise ValidationError("NeMo artifact delta has unexpected fields")
            for key in ("example_hash", "text_hash", "audio_path_hash"):
                value = item[key]
                if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
                    raise ValidationError(f"NeMo artifact delta {key} must be a sha256 digest")
            duration_ms = item["duration_ms"]
            weight = item["weight"]
            if (
                isinstance(duration_ms, bool)
                or not isinstance(duration_ms, int)
                or duration_ms <= 0
            ):
                raise ValidationError("NeMo artifact delta duration_ms must be positive")
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise ValidationError("NeMo artifact delta weight must be numeric")
            if not math.isfinite(float(weight)):
                raise ValidationError("NeMo artifact delta weight must be finite")
            deltas.append(
                FakeDelta(
                    example_hash=str(item["example_hash"]),
                    text_hash=str(item["text_hash"]),
                    audio_path_hash=str(item["audio_path_hash"]),
                    duration_ms=int(duration_ms),
                    weight=float(weight),
                )
            )
        if len({delta.example_hash for delta in deltas}) != len(deltas):
            raise ValidationError("NeMo artifact deltas contain duplicate row fingerprints")
        typed_lineage: TrainerLineage = {
            "dataset_hash": str(lineage["dataset_hash"]),
            "git_sha": str(lineage["git_sha"]),
            "image_digest": str(lineage["image_digest"]),
        }
        return cls(
            dataset_hash=dataset_hash,
            lineage=typed_lineage,
            base_model=str(raw["base_model"]),
            base_checkpoint=str(raw["base_checkpoint"]),
            seed=int(raw["seed"]),
            params=dict(raw["params"]),
            deltas=tuple(deltas),
            source_rows=int(raw["source_rows"]),
        )

    def content_hash(self) -> str:
        """Return the SHA-256 content hash of the canonical artifact bytes."""
        return sha256_text(canonical_json(self.to_dict()))


class ArtifactStore(Protocol):
    """Artifact persistence seam for fake and real NeMo exporters."""

    def save(self, artifact: FakeNeMoArtifact, *, prefix: str) -> str:
        """Persist an artifact and return a deployable artifact URI."""
        ...

    def load(self, uri: str) -> FakeNeMoArtifact:
        """Load and validate a previously exported artifact."""
        ...


def _uri_digest(uri: str) -> str:
    """Return and validate the trailing SHA-256 digest of a fake artifact URI."""
    digest = uri.rsplit("/", 1)[-1]
    if not _DIGEST_RE.fullmatch(digest):
        raise ValidationError(f"invalid fake NeMo artifact URI: {uri!r}")
    return digest


class FakeArtifactStore:
    """In-memory hash-addressed artifact store used by tests and local runs."""

    def __init__(self) -> None:
        """Create an empty artifact store."""
        self._objects: dict[str, str] = {}

    @property
    def uris(self) -> tuple[str, ...]:
        """Return all stored artifact URIs in insertion order."""
        return tuple(self._objects)

    def save(self, artifact: FakeNeMoArtifact, *, prefix: str) -> str:
        """Export canonical artifact bytes and return a content-addressed URI.

        Raises:
            ValidationError: If the prefix is empty or the digest collides with
                different artifact bytes.

        """
        if not prefix:
            raise ValidationError("artifact URI prefix must be non-empty")
        digest = artifact.content_hash()
        uri = f"{prefix.rstrip('/')}/{_ARTIFACT_KIND}/{digest}"
        raw = canonical_json(artifact.to_dict())
        existing = self._objects.get(uri)
        if existing is not None and existing != raw:
            raise ValidationError(f"NeMo artifact content hash collision at {uri!r}")
        self._objects[uri] = raw
        return uri

    def load(self, uri: str) -> FakeNeMoArtifact:
        """Load an artifact, rejecting missing, malformed or tampered bytes."""
        digest = _uri_digest(uri)
        try:
            raw = self._objects[uri]
        except KeyError as exc:
            raise ValidationError(f"NeMo artifact not found: {uri!r}") from exc
        try:
            decoded = json.loads(raw, parse_constant=_reject_json_constant)
        except ValueError as exc:
            raise ValidationError(f"NeMo artifact is not valid JSON: {exc}") from exc
        if not isinstance(decoded, Mapping):
            raise ValidationError("NeMo artifact must be a JSON object")
        artifact = FakeNeMoArtifact.from_dict(decoded)
        if artifact.content_hash() != digest:
            raise ValidationError("NeMo artifact content hash does not match its URI")
        return artifact

    def put_raw(self, uri: str, raw: str) -> None:
        """Inject or tamper with artifact bytes for failure-injection tests."""
        _uri_digest(uri)
        self._objects[uri] = raw


def _reject_json_constant(value: str) -> Any:
    """Reject non-standard JSON constants (NaN, Infinity) during artifact load."""
    raise ValueError(f"non-standard JSON constant {value!r}")


class ParakeetNeMoTrainer(TrainerPlugin):
    """Fake Parakeet/NeMo fine-tune plugin registered as ``parakeet-nemo``.

    P3-02.1 is implemented by :meth:`train`; P3-02.2 by :meth:`emit_confidences`,
    which produces schema-shaped events and scores them through an injectable
    triage sink (P2-01 by default).
    """

    def __init__(
        self,
        *,
        settings: NemoSettings | None = None,
        source: NemoManifestSource | None = None,
        checkpoint_source: CheckpointSource | None = None,
        backend: NeMoFinetuneBackend | None = None,
        store: ArtifactStore | None = None,
    ) -> None:
        """Wire settings and injectable fakes/real backends.

        Args:
            settings: NeMo settings; defaults to environment-backed fakes.
            source: NeMo manifest source; defaults to a JSONL reader.
            checkpoint_source: Checkpoint loader; defaults to the fake source.
            backend: Fine-tune backend; required when settings request ``nemo``.
            store: Artifact store; defaults to the in-memory fake.

        Raises:
            ConfigError: If settings request a real backend without injection.

        """
        config = settings or NemoSettings()
        if config.backend != "fake" and backend is None:
            raise ConfigError(
                f"NeMo backend {config.backend!r} requires an injected NeMoFinetuneBackend"
            )
        self._settings = config
        self._source: NemoManifestSource = source or JsonlNemoManifestSource(config.manifest_path)
        self._checkpoint_source: CheckpointSource = checkpoint_source or FakeCheckpointSource()
        self._backend: NeMoFinetuneBackend = backend or FakeNeMoFinetuneBackend()
        self._store: ArtifactStore = store or FakeArtifactStore()

    @property
    def name(self) -> str:
        """Return the stable registered plugin name."""
        return _PLUGIN_NAME

    def train(self, manifest: Mapping[str, Any], settings: TrainerSettings) -> TrainerResult:
        """Run a deterministic stand-in fine-tune from a fake checkpoint.

        Args:
            manifest: Dataset manifest with a non-empty ``content_hash``.
            settings: Core job settings supplying model, seed, lineage and URI.

        Returns:
            A validated 1.0.0 contract result pointing to the exported artifact.

        Raises:
            ValidationError: If the manifest, export rows or artifact are invalid.

        """
        content_hash = manifest.get("content_hash")
        if not isinstance(content_hash, str) or not content_hash:
            raise ValidationError("dataset_manifest.content_hash must be a non-empty string")
        rows = self._source.load_rows(manifest)
        if not rows:
            raise ValidationError("NeMo fine-tune requires at least one manifest row")
        checkpoint = self._checkpoint_source.load(
            self._settings.checkpoint_uri, model_id=settings.model
        )
        params = self._settings.as_training_params()
        deltas = self._backend.finetune(
            rows,
            checkpoint=checkpoint,
            model=settings.model,
            seed=settings.seed,
            params=params,
        )
        lineage: TrainerLineage = {
            "dataset_hash": content_hash,
            "git_sha": settings.git_sha,
            "image_digest": settings.image_digest,
        }
        artifact = FakeNeMoArtifact(
            dataset_hash=content_hash,
            lineage=lineage,
            base_model=settings.model,
            base_checkpoint=checkpoint.checkpoint_hash,
            seed=settings.seed,
            params=params,
            deltas=deltas,
            source_rows=len(rows),
        )
        artifact_uri = self._store.save(artifact, prefix=settings.artifact_uri_prefix)
        metrics = {
            "fake_train_loss": 0.0,
            "examples_total": float(len(rows)),
            "avg_duration_ms": round(sum(row.duration for row in rows) * 1000 / len(rows), 3),
        }
        return build_trainer_result(manifest, settings, artifact_uri=artifact_uri, metrics=metrics)

    def emit_confidences(
        self,
        rows: Sequence[NemoManifestRow],
        core_settings: TrainerSettings,
        *,
        identifiers: Sequence[tuple[str, str, int]],
        sink: TriageSignalSink | None = None,
    ) -> tuple[ConfidenceEmission, ...]:
        """Build schema-shaped events with fake words and score them for triage.

        Args:
            rows: Read-only P2-05 manifest rows (usually the ones just trained on).
            core_settings: Core settings supplying seed and base model.
            identifiers: One ``(event_id, call_id, turn_index)`` triple per row;
                supplied by the caller so no call identity is fabricated.
            sink: Triage sink; P2-01's ``ScoringEngine`` is the default.

        Returns:
            One labelled emission per row, each containing a schema-valid event.

        Raises:
            ValidationError: If the row/identifier counts differ.

        """
        if len(rows) != len(identifiers):
            raise ValidationError("emit_confidences needs one identifier triple per row")
        emitter = ConfidenceEmitter(settings=self._settings, core_settings=core_settings, sink=sink)
        return tuple(
            emitter.emit(row, event_id=event_id, call_id=call_id, turn_index=turn_index)
            for row, (event_id, call_id, turn_index) in zip(rows, identifiers, strict=True)
        )


__all__: Sequence[str] = (
    "ArtifactStore",
    "CheckpointSource",
    "FakeArtifactStore",
    "FakeCheckpoint",
    "FakeCheckpointSource",
    "FakeDelta",
    "FakeNeMoArtifact",
    "FakeNeMoFinetuneBackend",
    "NeMoFinetuneBackend",
    "ParakeetNeMoTrainer",
)
