"""Read-only lineage graph over P2-05 manifests and P2-06 trainer results.

Two directions are supported: training example → source call, and source call →
every model trained on a dataset containing that call. Links come only from
exact IDs in committed manifests (``source_call_ids``, ``content_hash``) and
contract-valid trainer output (``lineage.dataset_hash``, ``run_id``). Nothing is
inferred from names, text, timestamps or partial matches; duplicate inputs are
deduplicated deterministically and conflicting links fail closed.

Dataset manifest schema 1.0.0 records source call IDs but **not** per-example
IDs, so example → call additionally requires an injected exact example index.
Without one the query fails closed rather than reconstructing a link from the
example-ID string alone.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from voiceloop_common import ValidationError, VoiceLoopError, validate

from .store import ManifestSource, TrainerLineageSource


class LineageError(VoiceLoopError):
    """Base class for expected lineage query failures."""


class UnknownCallError(LineageError):
    """The call ID is not present in any committed dataset manifest."""


class UnknownExampleError(LineageError):
    """The example ID has no exact, committed call lineage."""


class ConflictingLineageError(ValidationError):
    """Two inputs claim conflicting lineage for the same exact ID."""


@dataclass(frozen=True, slots=True)
class DatasetRef:
    """Minimal immutable view of a committed P2-05 manifest."""

    dataset_id: str
    content_hash: str
    created_at: str
    source_call_ids: tuple[str, ...]

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> DatasetRef:
        """Validate a manifest and build an exact reference from it."""
        snapshot = deepcopy(dict(manifest))
        validate(snapshot, "dataset_manifest")
        return cls(
            dataset_id=str(snapshot["dataset_id"]),
            content_hash=str(snapshot["content_hash"]),
            created_at=str(snapshot["created_at"]),
            source_call_ids=tuple(str(call_id) for call_id in snapshot["source_call_ids"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModelRef:
    """One training run that consumed a dataset with a known content hash."""

    model_id: str
    run_id: str
    artifact_uri: str
    dataset_id: str
    content_hash: str
    git_sha: str
    image_digest: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExampleLineage:
    """Answer to an example → call → datasets/models query."""

    example_id: str
    call_id: str
    datasets: tuple[DatasetRef, ...]
    models: tuple[ModelRef, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy."""
        return {
            "example_id": self.example_id,
            "call_id": self.call_id,
            "datasets": [dataset.to_dict() for dataset in self.datasets],
            "models": [model.to_dict() for model in self.models],
        }


class ExampleResolver(Protocol):
    """Map a committed example ID to an exact call ID, or fail closed."""

    def resolve(self, example_id: str) -> str:
        """Return the exact source call ID for ``example_id``."""
        ...


class P205ExampleIdResolver:
    """Resolve IDs from an explicit committed example index.

    P2-05 builds IDs as ``<call_id>:<turn_index>:<kind>`` where ``kind`` is one
    of ``stt``, ``parser_clean`` or ``parser_noisy``. Membership in the supplied
    index proves the example was committed by a build; the returned call ID is
    then checked against the dataset index by :class:`LineageGraph`. A
    well-formed but uncommitted ID is rejected, so no link is fabricated.
    """

    _PATTERN = re.compile(
        r"^(?P<call_id>[^:]+):(?P<turn>[0-9]+):(?P<kind>stt|parser_clean|parser_noisy)$"
    )

    def __init__(self, example_ids: Iterable[str]) -> None:
        """Initialise with the exact example IDs emitted by a committed build."""
        self._example_ids = frozenset(example_ids)

    def resolve(self, example_id: str) -> str:
        """Return the call ID component of an indexed P2-05 example ID."""
        if not isinstance(example_id, str) or example_id not in self._example_ids:
            raise UnknownExampleError("example ID is absent from the committed example index")
        match = self._PATTERN.fullmatch(example_id)
        if match is None:
            raise UnknownExampleError("example ID does not match the committed P2-05 convention")
        return match.group("call_id")


class MappingExampleResolver:
    """Fake resolver for sources with an explicit example-ID index.

    Production adapters can inject the real example index here. The graph
    still validates the returned call ID against committed manifests.
    """

    def __init__(self, mapping: Mapping[str, str]) -> None:
        """Initialise with exact example-ID → call-ID mappings."""
        self._mapping = dict(mapping)

    def resolve(self, example_id: str) -> str:
        """Return the mapped call ID or fail closed."""
        call_id = self._mapping.get(example_id)
        if not call_id:
            raise UnknownExampleError("example ID is absent from the injected example index")
        return call_id


class LineageGraph:
    """Immutable lineage index built from read-only, schema-valid sources."""

    def __init__(
        self,
        manifest_source: ManifestSource,
        trainer_source: TrainerLineageSource,
        *,
        example_resolver: ExampleResolver | None = None,
    ) -> None:
        """Load and validate all manifests and trainer results.

        Args:
            manifest_source: Source of committed P2-05 dataset manifests.
            trainer_source: Source of P2-06 trainer contract results.
            example_resolver: Optional explicit example index. Example → call
                queries fail closed when it is omitted; call → model queries
                do not need it.

        Raises:
            ValidationError: If any source payload is malformed.
            ConflictingLineageError: If duplicate IDs carry different data.

        """
        self._datasets = self._load_datasets(manifest_source.manifests())
        self._models = self._load_models(trainer_source.results(), trainer_source)
        self._resolver = example_resolver
        self._datasets_by_call = self._index_datasets_by_call(self._datasets)
        self._models_by_hash = self._index_models_by_hash(self._models)

    @staticmethod
    def _load_datasets(manifests: Iterable[Mapping[str, Any]]) -> tuple[DatasetRef, ...]:
        """Validate, deduplicate and deterministically sort manifests."""
        refs = [DatasetRef.from_manifest(manifest) for manifest in manifests]
        refs.sort(key=lambda ref: (ref.dataset_id, ref.content_hash, ref.created_at))
        by_hash: dict[str, DatasetRef] = {}
        by_id: dict[str, DatasetRef] = {}
        unique: list[DatasetRef] = []
        for ref in refs:
            if ref.content_hash in by_hash and by_hash[ref.content_hash] != ref:
                raise ConflictingLineageError("conflicting manifests for one content hash")
            if ref.dataset_id in by_id and by_id[ref.dataset_id] != ref:
                raise ConflictingLineageError("conflicting manifests for one dataset ID")
            if ref not in unique:
                unique.append(ref)
                by_hash[ref.content_hash] = ref
                by_id[ref.dataset_id] = ref
        return tuple(unique)

    @staticmethod
    def _load_models(
        results: Iterable[Mapping[str, Any]],
        trainer_source: TrainerLineageSource,
    ) -> tuple[ModelRef, ...]:
        """Validate, deduplicate and deterministically sort trainer results."""
        refs: list[ModelRef] = []
        for result in results:
            snapshot = deepcopy(dict(result))
            validate(snapshot, "trainer_contract")
            if snapshot.get("contract_part") != "output":
                raise ValidationError("trainer lineage source requires contract_part=output")
            lineage = snapshot.get("lineage")
            if not isinstance(lineage, Mapping):
                raise ValidationError("trainer result has no lineage object")
            dataset_hash = str(lineage.get("dataset_hash", ""))
            if not dataset_hash:
                raise ValidationError("trainer result lineage has no dataset_hash")
            run_id = str(snapshot.get("run_id", ""))
            if not run_id:
                raise ValidationError("trainer result has no run_id")
            mapped_model_id = trainer_source.model_id_for_run(run_id)
            refs.append(
                ModelRef(
                    model_id=mapped_model_id or run_id,
                    run_id=run_id,
                    artifact_uri=str(snapshot["artifact_uri"]),
                    dataset_id="",
                    content_hash=dataset_hash,
                    git_sha=str(lineage["git_sha"]),
                    image_digest=str(lineage["image_digest"]),
                )
            )
        refs.sort(
            key=lambda ref: (
                ref.model_id,
                ref.run_id,
                ref.content_hash,
                ref.artifact_uri,
                ref.git_sha,
                ref.image_digest,
            )
        )
        by_run: dict[str, ModelRef] = {}
        unique: list[ModelRef] = []
        for ref in refs:
            if ref.run_id in by_run and by_run[ref.run_id] != ref:
                raise ConflictingLineageError("conflicting trainer results for one run ID")
            if ref not in unique:
                unique.append(ref)
                by_run[ref.run_id] = ref
        return tuple(unique)

    @staticmethod
    def _index_datasets_by_call(
        datasets: Iterable[DatasetRef],
    ) -> dict[str, tuple[DatasetRef, ...]]:
        """Index datasets by each exact call ID in ``source_call_ids``."""
        index: dict[str, list[DatasetRef]] = {}
        for dataset in datasets:
            for call_id in set(dataset.source_call_ids):
                index.setdefault(call_id, []).append(dataset)
        return {call_id: tuple(refs) for call_id, refs in index.items()}

    @staticmethod
    def _index_models_by_hash(models: Iterable[ModelRef]) -> dict[str, tuple[ModelRef, ...]]:
        """Index models by the exact dataset content hash they consumed."""
        index: dict[str, list[ModelRef]] = {}
        for model in models:
            index.setdefault(model.content_hash, []).append(model)
        return {digest: tuple(refs) for digest, refs in index.items()}

    @staticmethod
    def _join_datasets_models(
        datasets: Iterable[DatasetRef],
        models_by_hash: Mapping[str, tuple[ModelRef, ...]],
    ) -> tuple[ModelRef, ...]:
        """Return every model linked to any supplied dataset, deduplicated."""
        joined: tuple[ModelRef, ...] = ()
        for dataset in datasets:
            for model in models_by_hash.get(dataset.content_hash, ()):
                linked = ModelRef(
                    model_id=model.model_id,
                    run_id=model.run_id,
                    artifact_uri=model.artifact_uri,
                    dataset_id=dataset.dataset_id,
                    content_hash=model.content_hash,
                    git_sha=model.git_sha,
                    image_digest=model.image_digest,
                )
                if linked not in joined:
                    joined += (linked,)
        return tuple(
            sorted(
                joined,
                key=lambda ref: (
                    ref.dataset_id,
                    ref.model_id,
                    ref.run_id,
                    ref.content_hash,
                ),
            )
        )

    def datasets(self) -> tuple[DatasetRef, ...]:
        """Return every validated dataset, deterministically ordered."""
        return self._datasets

    def models(self) -> tuple[ModelRef, ...]:
        """Return every validated model with its raw dataset hash.

        ``dataset_id`` is empty here because a trainer result carries only the
        exact ``content_hash``; joining it to a dataset happens per query.
        """
        return self._models

    def datasets_for_call(self, call_id: str) -> tuple[DatasetRef, ...]:
        """Return every committed dataset that contains the exact call ID.

        Args:
            call_id: Exact source call ID from a P2-05 manifest.

        Raises:
            UnknownCallError: If no committed manifest contains the call.

        """
        if not isinstance(call_id, str) or not call_id:
            raise UnknownCallError("call ID must be a non-empty exact identifier")
        found = self._datasets_by_call.get(call_id)
        if not found:
            raise UnknownCallError("call ID is absent from all committed dataset manifests")
        return found

    def models_for_call(self, call_id: str) -> tuple[ModelRef, ...]:
        """Return every model trained on any dataset containing the call.

        Args:
            call_id: Exact source call ID.

        Raises:
            UnknownCallError: If no committed manifest contains the call.

        """
        datasets = self.datasets_for_call(call_id)
        return self._join_datasets_models(datasets, self._models_by_hash)

    def example_to_call(self, example_id: str) -> ExampleLineage:
        """Resolve a training example to its source call, datasets and models.

        Args:
            example_id: Exact P2-05 example ID, or a key in the injected index.

        Raises:
            UnknownExampleError: If no exact example index was injected, or if
                the ID has no exact committed call lineage.

        """
        if self._resolver is None:
            raise UnknownExampleError("no exact example index was injected for this graph")
        call_id = self._resolver.resolve(example_id)
        try:
            datasets = self.datasets_for_call(call_id)
        except UnknownCallError as exc:
            raise UnknownExampleError("example has no committed dataset lineage") from exc
        return ExampleLineage(
            example_id=example_id,
            call_id=call_id,
            datasets=datasets,
            models=self._join_datasets_models(datasets, self._models_by_hash),
        )
