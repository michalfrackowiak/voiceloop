"""Deterministic fake LoRA-style adapter, artifact store and serving seam.

SYNTHETIC / FIXTURE TRAINING STAND-IN. Nothing here imports or runs torch,
Unsloth, TRL, LLaMA-Factory, GLiNER2 or vLLM. The adapter is a canonical JSON
descriptor of pseudo-weight deltas whose hash is reproducible from the input
examples and settings. Real backends implement :class:`SlotTrainingBackend`;
real serving layers implement :class:`ParserRuntime` and are injected.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from voiceloop_common import ValidationError

ADAPTER_VERSION = "1.0.0"
AdapterKind = Literal["fake-lora", "fake-gliner2"]
ExampleKind = Literal["gold", "noisy"]

_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_ADAPTER_KINDS: frozenset[str] = frozenset({"fake-lora", "fake-gliner2"})
_EXAMPLE_KINDS: frozenset[str] = frozenset({"gold", "noisy"})


def canonical_json(value: object) -> str:
    """Return deterministic compact JSON text for hashing and artifact export."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    """Return the lowercase SHA-256 hex digest of UTF-8 text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ParserExample:
    """One parser SFT example derived from a reviewed dataset row.

    ``gold`` examples come from a corrected transcript; ``noisy`` examples come
    from the original STT hypothesis. Both carry the reviewed booking slots.
    """

    example_id: str
    kind: ExampleKind
    source_text: str
    slots: dict[str, Any]
    call_id: str = ""
    synthetic: bool = True

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> ParserExample:
        """Validate an injected loader record and build a parser example.

        Args:
            row: Mapping with ``example_id``, ``kind`` (``gold``/``noisy``),
                non-empty ``source_text`` and non-empty ``slots``.

        Returns:
            A validated :class:`ParserExample`.

        Raises:
            ValidationError: If any required field is missing or malformed.

        """
        example_id = str(row.get("example_id", "")).strip()
        kind_raw = row.get("kind")
        source_text = row.get("source_text")
        slots_raw = row.get("slots")
        if not example_id:
            raise ValidationError("parser example requires a non-empty example_id")
        if kind_raw not in _EXAMPLE_KINDS:
            raise ValidationError(f"parser example kind must be gold or noisy, got {kind_raw!r}")
        if not isinstance(source_text, str) or not source_text.strip():
            raise ValidationError(f"parser example {example_id!r} requires non-empty source_text")
        if not isinstance(slots_raw, Mapping) or not slots_raw:
            raise ValidationError(f"parser example {example_id!r} requires non-empty slots")
        kind: ExampleKind = "gold" if kind_raw == "gold" else "noisy"
        slots: dict[str, Any] = {str(key): value for key, value in slots_raw.items()}
        return cls(
            example_id=example_id,
            kind=kind,
            source_text=source_text,
            slots=slots,
            call_id=str(row.get("call_id", "")),
            synthetic=bool(row.get("synthetic", True)),
        )


@dataclass(frozen=True, slots=True)
class AdapterDelta:
    """One deterministic pseudo-weight update, keyed by example lineage ID.

    Source text is stored only as a hash so the exported stand-in artifact does
    not copy transcript text (``context.md`` section 12).
    """

    example_id: str
    kind: ExampleKind
    source_hash: str
    slots: dict[str, Any]
    weight: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable, stable representation of this delta."""
        return {
            "example_id": self.example_id,
            "kind": self.kind,
            "source_hash": self.source_hash,
            "slots": self.slots,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> AdapterDelta:
        """Validate a decoded delta before it enters a loaded artifact.

        Raises:
            ValidationError: If any field is missing or has an unexpected type.

        """
        expected = {"example_id", "kind", "source_hash", "slots", "weight"}
        if set(raw) != expected:
            raise ValidationError("adapter delta has unexpected fields")
        example_id = raw["example_id"]
        kind = raw["kind"]
        source_hash = raw["source_hash"]
        slots = raw["slots"]
        weight = raw["weight"]
        if not isinstance(example_id, str) or not example_id:
            raise ValidationError("adapter delta example_id must be a non-empty string")
        if kind not in _EXAMPLE_KINDS:
            raise ValidationError(f"adapter delta kind is invalid: {kind!r}")
        if not isinstance(source_hash, str) or not _DIGEST_RE.fullmatch(source_hash):
            raise ValidationError("adapter delta source_hash must be a sha256 hex digest")
        if not isinstance(slots, dict) or not slots:
            raise ValidationError("adapter delta slots must be a non-empty object")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValidationError("adapter delta weight must be numeric")
        if not math.isfinite(weight):
            raise ValidationError("adapter delta weight must be finite")
        delta_kind: ExampleKind = "gold" if kind == "gold" else "noisy"
        return cls(
            example_id=example_id,
            kind=delta_kind,
            source_hash=source_hash,
            slots={str(key): value for key, value in slots.items()},
            weight=float(weight),
        )


@dataclass(frozen=True, slots=True)
class FakeLoraAdapter:
    """Deterministic, hash-addressed LoRA-style stand-in artifact.

    The same examples, seed and parameters always produce the same content
    hash, independent of input ordering. ``synthetic_training_standin`` makes
    the provenance label explicit to every consumer of the artifact.
    """

    dataset_hash: str
    deltas: tuple[AdapterDelta, ...]
    kind: AdapterKind = "fake-lora"
    schema_version: str = ADAPTER_VERSION
    base_model: str = ""
    seed: int = 0
    params: dict[str, Any] = field(default_factory=dict)
    synthetic_training_standin: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return canonical JSON data for export and hashing."""
        return {
            "base_model": self.base_model,
            "dataset_hash": self.dataset_hash,
            "deltas": [
                delta.to_dict() for delta in sorted(self.deltas, key=lambda d: d.example_id)
            ],
            "kind": self.kind,
            "params": self.params,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "synthetic_training_standin": self.synthetic_training_standin,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FakeLoraAdapter:
        """Rebuild and structurally validate a decoded artifact.

        Raises:
            ValidationError: If the payload is malformed or has extra fields.

        """
        expected = {
            "base_model",
            "dataset_hash",
            "deltas",
            "kind",
            "params",
            "schema_version",
            "seed",
            "synthetic_training_standin",
        }
        if set(raw) != expected:
            raise ValidationError("adapter artifact has unexpected fields")
        if raw["schema_version"] != ADAPTER_VERSION:
            raise ValidationError(f"unsupported adapter schema version: {raw['schema_version']!r}")
        if raw["kind"] not in _ADAPTER_KINDS:
            raise ValidationError(f"unsupported adapter kind: {raw['kind']!r}")
        dataset_hash = raw["dataset_hash"]
        if not isinstance(dataset_hash, str) or not _DIGEST_RE.fullmatch(dataset_hash):
            raise ValidationError("adapter dataset_hash must be a sha256 hex digest")
        if not isinstance(raw["base_model"], str):
            raise ValidationError("adapter base_model must be a string")
        if isinstance(raw["seed"], bool) or not isinstance(raw["seed"], int):
            raise ValidationError("adapter seed must be an integer")
        if not isinstance(raw["params"], dict):
            raise ValidationError("adapter params must be an object")
        if not isinstance(raw["synthetic_training_standin"], bool):
            raise ValidationError("adapter synthetic_training_standin must be a boolean")
        raw_deltas = raw["deltas"]
        if not isinstance(raw_deltas, list) or not raw_deltas:
            raise ValidationError("adapter deltas must be a non-empty list")
        deltas = tuple(AdapterDelta.from_dict(delta) for delta in raw_deltas)
        if len({delta.example_id for delta in deltas}) != len(deltas):
            raise ValidationError("adapter deltas contain duplicate example_id values")
        kind: AdapterKind = "fake-lora" if raw["kind"] == "fake-lora" else "fake-gliner2"
        return cls(
            dataset_hash=dataset_hash,
            deltas=deltas,
            kind=kind,
            schema_version=str(raw["schema_version"]),
            base_model=str(raw["base_model"]),
            seed=int(raw["seed"]),
            params=dict(raw["params"]),
            synthetic_training_standin=bool(raw["synthetic_training_standin"]),
        )

    def content_hash(self) -> str:
        """Return the SHA-256 content hash of the canonical artifact bytes."""
        return sha256_text(canonical_json(self.to_dict()))


class SlotTrainingBackend(Protocol):
    """Injectable training seam for LoRA SFT and optional GLiNER2 backends.

    A real implementation owns its heavy imports and returns adapter deltas in
    this stand-in descriptor format; production must additionally persist the
    native merged adapter through its own artifact writer.
    """

    def train(
        self,
        examples: Sequence[ParserExample],
        *,
        model: str,
        seed: int,
        params: Mapping[str, Any],
    ) -> tuple[AdapterDelta, ...]:
        """Train on gold and noisy examples and return adapter deltas."""
        ...


class FakeLoraTrainingBackend:
    """Deterministic pseudo-training backend: no GPU, weights or network I/O."""

    def train(
        self,
        examples: Sequence[ParserExample],
        *,
        model: str,
        seed: int,
        params: Mapping[str, Any],
    ) -> tuple[AdapterDelta, ...]:
        """Derive reproducible pseudo-weights from examples and settings.

        Args:
            examples: Gold and noisy parser examples.
            model: Base model ID from core settings (recorded in the digest).
            seed: Random seed from core settings.
            params: Hyperparameters from :class:`~.settings.ParserSettings`.

        Returns:
            Deltas sorted by example ID, independent of input ordering.

        """
        prefix = "|".join((str(seed), model, canonical_json(params)))
        deltas = [
            AdapterDelta(
                example_id=example.example_id,
                kind=example.kind,
                source_hash=sha256_text(example.source_text),
                slots=dict(example.slots),
                weight=round(
                    int(
                        sha256_text(
                            "|".join(
                                (
                                    prefix,
                                    example.example_id,
                                    example.kind,
                                    example.source_text,
                                    canonical_json(example.slots),
                                )
                            )
                        )[:8],
                        16,
                    )
                    / 0xFFFFFFFF,
                    6,
                ),
            )
            for example in examples
        ]
        return tuple(sorted(deltas, key=lambda delta: delta.example_id))


class AdapterStore(Protocol):
    """Artifact persistence seam for fake and real adapter stores."""

    def save(self, adapter: FakeLoraAdapter, *, prefix: str) -> str:
        """Persist an adapter and return its deployable artifact URI."""
        ...

    def load(self, uri: str) -> FakeLoraAdapter:
        """Load and validate a previously exported artifact."""
        ...


def _reject_json_constant(value: str) -> Any:
    """Reject non-standard JSON constants (NaN, Infinity) during artifact load."""
    raise ValueError(f"non-standard JSON constant {value!r}")


def _uri_digest(uri: str) -> str:
    """Return and validate the trailing SHA-256 digest of a fake artifact URI."""
    digest = uri.rsplit("/", 1)[-1]
    if not _DIGEST_RE.fullmatch(digest):
        raise ValidationError(f"invalid fake adapter URI: {uri!r}")
    return digest


class FakeAdapterStore:
    """In-memory, hash-addressed artifact store used by tests and local runs."""

    def __init__(self) -> None:
        """Create an empty artifact store."""
        self._objects: dict[str, str] = {}

    @property
    def uris(self) -> tuple[str, ...]:
        """Return all stored artifact URIs in insertion order."""
        return tuple(self._objects)

    def save(self, adapter: FakeLoraAdapter, *, prefix: str) -> str:
        """Export canonical adapter bytes and return a content-addressed URI.

        Raises:
            ValidationError: If the prefix is empty or the digest collides with
                different artifact bytes.

        """
        if not prefix:
            raise ValidationError("artifact URI prefix must be non-empty")
        digest = adapter.content_hash()
        uri = f"{prefix.rstrip('/')}/{adapter.kind}/{digest}"
        raw = canonical_json(adapter.to_dict())
        existing = self._objects.get(uri)
        if existing is not None and existing != raw:
            raise ValidationError(f"adapter content hash collision at {uri!r}")
        self._objects[uri] = raw
        return uri

    def load(self, uri: str) -> FakeLoraAdapter:
        """Load an artifact, rejecting missing, malformed or tampered bytes.

        Raises:
            ValidationError: If the URI is malformed, the artifact is absent or
                its content hash does not match the URI digest.

        """
        digest = _uri_digest(uri)
        try:
            raw = self._objects[uri]
        except KeyError as exc:
            raise ValidationError(f"adapter artifact not found: {uri!r}") from exc
        try:
            decoded = json.loads(raw, parse_constant=_reject_json_constant)
        except ValueError as exc:
            raise ValidationError(f"adapter artifact is not valid JSON: {exc}") from exc
        if not isinstance(decoded, dict):
            raise ValidationError("adapter artifact must be a JSON object")
        adapter = FakeLoraAdapter.from_dict(decoded)
        if adapter.content_hash() != digest:
            raise ValidationError("adapter artifact content hash does not match its URI")
        return adapter

    def put_raw(self, uri: str, raw: str) -> None:
        """Inject or tamper with artifact bytes for failure-injection tests.

        Args:
            uri: Content-addressed URI whose payload is replaced.
            raw: Raw canonical JSON text to store, valid or deliberately broken.

        """
        _uri_digest(uri)
        self._objects[uri] = raw


class ParserRuntime(Protocol):
    """Injectable vLLM-like serving seam for runtime LoRA load/unload."""

    def load_adapter(self, uri: str) -> str:
        """Load an adapter artifact and return a stable handle."""
        ...

    def unload_adapter(self, handle: str) -> None:
        """Unload a handle; repeated unloads are no-ops."""
        ...

    def generate(self, handle: str, prompt: str) -> str:
        """Generate raw candidate text for a loaded adapter handle."""
        ...


@dataclass(frozen=True, slots=True)
class RuntimeStats:
    """Counters proving load/unload idempotence in the fake runtime."""

    loads: int = 0
    unloads: int = 0
    generations: int = 0


class FakeVllmRuntime:
    """In-memory vLLM stand-in with idempotent load/unload and fail-closed load."""

    def __init__(
        self, store: AdapterStore, *, generation_overrides: Mapping[str, str] | None = None
    ) -> None:
        """Wire the runtime to an artifact store and optional failure injections.

        Args:
            store: Artifact store used to validate and load adapters.
            generation_overrides: Handle-to-text overrides that simulate bad
                generations (e.g. schema-invalid output) in constrained tests.

        """
        self._store = store
        self._generation_overrides = dict(generation_overrides or {})
        self._loaded: dict[str, FakeLoraAdapter] = {}
        self._loads = 0
        self._unloads = 0
        self._generations = 0

    @property
    def stats(self) -> RuntimeStats:
        """Return counters for successful loads, unloads and generations."""
        return RuntimeStats(loads=self._loads, unloads=self._unloads, generations=self._generations)

    @property
    def loaded_handles(self) -> tuple[str, ...]:
        """Return sorted handles currently loaded in the runtime."""
        return tuple(sorted(self._loaded))

    def load_adapter(self, uri: str) -> str:
        """Validate and load an artifact; repeated loads return the same handle.

        Raises:
            ValidationError: If the artifact is invalid; no handle is registered.

        """
        adapter = self._store.load(uri)
        handle = f"fake-vllm:{_uri_digest(uri)}"
        if handle in self._loaded:
            return handle
        self._loaded[handle] = adapter
        self._loads += 1
        return handle

    def unload_adapter(self, handle: str) -> None:
        """Unload a handle once; unknown or already-unloaded handles are no-ops."""
        if handle in self._loaded:
            del self._loaded[handle]
            self._unloads += 1

    def generate(self, handle: str, prompt: str) -> str:
        """Return reviewed slots as raw JSON text for a matching prompt.

        Args:
            handle: Handle returned by :meth:`load_adapter`.
            prompt: SFT-style prompt ending in ``Utterance: <text>``.

        Returns:
            Compact JSON text. An unmatched utterance returns ``{}`` so the
            downstream schema gate rejects it rather than inventing a repair.

        Raises:
            ValidationError: If the handle is not currently loaded.

        """
        if handle not in self._loaded:
            raise ValidationError(f"adapter handle is not loaded: {handle!r}")
        self._generations += 1
        if handle in self._generation_overrides:
            return self._generation_overrides[handle]
        adapter = self._loaded[handle]
        marker = "Utterance: "
        utterance = prompt.rsplit(marker, 1)[1].strip() if marker in prompt else prompt.strip()
        source_hash = sha256_text(utterance)
        for delta in adapter.deltas:
            if delta.source_hash == source_hash:
                return canonical_json(delta.slots)
        return "{}"
