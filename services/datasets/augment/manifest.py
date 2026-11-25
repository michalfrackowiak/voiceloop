"""Internal augmentation manifest, provenance checks and safe writes (P3-03.3).

The output is an **internal** artifact, not a ``dataset_manifest.schema.json``
document: schema 1.0.0 forbids extra properties and cannot express per-variant
transform params. The P2-05 manifest is consumed read-only and validated against
that contract; this module never edits it, the sealed golden stand-in or the
fixtures. Every generated row is tagged synthetic and excluded from evaluation
sets by construction and by validation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voiceloop_common import ConfigError, ValidationError, utcnow, validate

from ..manifest import canonical_json
from .settings import (
    ALLOWED_TRANSFORMS,
    ARTIFACT_TYPE,
    AUGMENTATION_SCHEMA_VERSION,
    STAND_IN_LABEL,
    AugmentSettings,
)
from .variants import AudioStandIn, VariantSpec

REPO_ROOT = Path(__file__).resolve().parents[3]
READ_ONLY_ROOTS: tuple[Path, ...] = (
    Path("services/datasets/golden/artifacts"),
    Path("tests/fixtures"),
)
_NON_TEXT_KEYS: frozenset[str] = frozenset(
    {
        "created_at",
        "content_hash",
        "source_manifest_content_hash",
        "digest",
        "token_digest",
    }
)
_AUDIO_BYTE_KEYS: frozenset[str] = frozenset({"samples", "audio_bytes", "pcm", "sample_values"})


def _resolve(path: str | Path) -> Path:
    """Resolve a repo-relative path against the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from disk with a structured failure."""
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read JSON at {path}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ConfigError(f"expected a JSON object at {path}")
    return decoded


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 of a read-only file's bytes."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class SourceHashes:
    """Immutable provenance snapshot of the read-only inputs."""

    golden_content_hash: str
    golden_file_sha256: str
    fixtures_sha256: str

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-safe snapshot for CLI output and tests."""
        return {
            "golden_content_hash": self.golden_content_hash,
            "golden_file_sha256": self.golden_file_sha256,
            "fixtures_sha256": self.fixtures_sha256,
        }


def snapshot_source_hashes(settings: AugmentSettings) -> SourceHashes:
    """Hash the sealed golden manifest and fixture pool without writing to them.

    Args:
        settings: Typed augmentation settings holding the expected anchors.

    Returns:
        The verified before/after snapshot values.

    Raises:
        ValidationError: If a committed hash no longer matches its anchor.

    """
    golden = _resolve(settings.golden_manifest_path)
    manifest = _read_json(golden)
    content_hash = manifest.get("content_hash")
    if content_hash != settings.expected_golden_content_hash:
        raise ValidationError("sealed golden content hash changed; refusing to augment")
    fixtures = _file_sha256(_resolve(settings.fixtures_path))
    if fixtures != settings.expected_fixture_sha256:
        raise ValidationError("fixture manifest hash changed; refusing to augment")
    return SourceHashes(
        golden_content_hash=str(content_hash),
        golden_file_sha256=_file_sha256(golden),
        fixtures_sha256=fixtures,
    )


def assert_source_hashes_unchanged(
    settings: AugmentSettings,
    before: SourceHashes,
) -> SourceHashes:
    """Re-hash read-only inputs and fail closed if anything moved.

    Args:
        settings: Typed augmentation settings.
        before: Snapshot taken before generation and writing.

    Returns:
        The matching after snapshot.

    Raises:
        ValidationError: If any before/after value differs.

    """
    after = snapshot_source_hashes(settings)
    if before != after:
        raise ValidationError("read-only golden/fixture inputs changed during augmentation")
    return after


def load_dataset_manifest(
    settings: AugmentSettings,
    *,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Load and schema-validate a read-only P2-05 dataset manifest.

    Args:
        settings: Typed augmentation settings.
        path: Optional override; defaults to ``settings.dataset_manifest_path``.

    Returns:
        The validated manifest object.

    Raises:
        ConfigError: If no manifest path was supplied or the file is unreadable.
        ValidationError: If the document violates ``dataset_manifest`` 1.0.0.

    """
    raw_path = path if path is not None else settings.dataset_manifest_path
    if not raw_path:
        raise ConfigError("a P2-05 dataset manifest path is required (--manifest)")
    document = _read_json(_resolve(raw_path))
    validate(document, "dataset_manifest")
    return document


@dataclass(frozen=True, slots=True)
class TransformRecord:
    """One transform name with PII-free, JSON-safe parameters."""

    name: str
    params: dict[str, int | str]

    def __post_init__(self) -> None:
        """Reject unknown transforms or non-scalar parameter values."""
        if self.name not in ALLOWED_TRANSFORMS:
            raise ValidationError(f"unknown augmentation transform: {self.name!r}")
        for key, value in self.params.items():
            if (
                not isinstance(key, str)
                or isinstance(value, bool)
                or not isinstance(value, (int, str))
            ):
                raise ValidationError("transform params must be string-to-scalar mappings")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe copy for the manifest."""
        return {"name": self.name, "params": dict(self.params)}


@dataclass(frozen=True, slots=True)
class AugmentedVariant:
    """One generated stand-in variant with full transform lineage."""

    spec: VariantSpec
    audio: AudioStandIn
    transforms: tuple[TransformRecord, ...]

    def __post_init__(self) -> None:
        """Require tagged audio, native-rate consistency and a TTS transform."""
        if self.audio.label != STAND_IN_LABEL:
            raise ValidationError("every variant must carry the stand-in label")
        if self.audio.sample_rate_hz != self.spec.sample_rate_hz:
            raise ValidationError("variant audio must keep the planned native sample rate")
        if self.audio.duration_ms != self.spec.duration_ms:
            raise ValidationError("variant audio duration differs from its plan")
        if not self.transforms or self.transforms[0].name != "fake_tts":
            raise ValidationError("variant must start with a fake_tts transform record")

    @property
    def source_call_id(self) -> str:
        """Return the source call ID used for exclusion checks."""
        return self.spec.source_call_id

    def to_dict(self) -> dict[str, object]:
        """Return a PII-free manifest record; samples are never emitted."""
        return {
            **self.spec.as_manifest_fields(),
            "transforms": [record.to_dict() for record in self.transforms],
            "audio": self.audio.metadata(),
            "synthetic": True,
        }


def _iter_strings(value: object, *, key: str | None = None) -> Iterable[str]:
    """Yield manifest strings that may carry caller-supplied text."""
    if isinstance(value, str):
        if key not in _NON_TEXT_KEYS:
            yield value
    elif isinstance(value, Mapping):
        for child_key, child in value.items():
            yield from _iter_strings(child, key=str(child_key))
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_strings(child, key=key)


def _assert_no_audio_bytes(value: object, *, key: str | None = None) -> None:
    """Reject raw sample arrays or codec-like fields in an output manifest."""
    if key in _AUDIO_BYTE_KEYS:
        raise ValidationError("audio bytes must never appear in an augmentation manifest")
    if isinstance(value, list) and len(value) > 64:
        if all(isinstance(item, int) and not isinstance(item, bool) for item in value):
            raise ValidationError("raw integer sample arrays are not allowed in manifests")
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            _assert_no_audio_bytes(child, key=str(child_key))
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_audio_bytes(child, key=key)


def build_manifest(
    variants: Sequence[AugmentedVariant],
    *,
    source_manifest: Mapping[str, Any],
    golden_call_ids: Iterable[str],
    regression_call_ids: Iterable[str],
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic internal manifest for one augmentation batch.

    Args:
        variants: Generated stand-in variants, each carrying transform lineage.
        source_manifest: Read-only, schema-valid P2-05 dataset manifest.
        golden_call_ids: Sealed evaluation call IDs that must never appear.
        regression_call_ids: Regression call IDs that must never appear.
        created_at: Optional UTC timestamp override for deterministic tests.

    Returns:
        A JSON-serialisable internal manifest with a content hash.

    Raises:
        ValidationError: If lineage, exclusion or PII checks fail.

    """
    from services.datasets.filters import detect_pii

    validate(dict(source_manifest), "dataset_manifest")
    golden = tuple(sorted(set(golden_call_ids)))
    regression = tuple(sorted(set(regression_call_ids) - set(golden)))
    excluded = set(golden) | set(regression)

    records = [variant.to_dict() for variant in variants]
    identifiers = [str(record["variant_id"]) for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValidationError("augmentation variant IDs must be unique")
    source_calls = sorted({variant.source_call_id for variant in variants})
    manifest_calls = set(source_manifest["source_call_ids"])
    if not set(source_calls).issubset(manifest_calls):
        raise ValidationError("variant call IDs must come from the P2-05 source manifest")
    leaked = excluded.intersection(source_calls)
    if leaked:
        raise ValidationError("synthetic variants leaked into golden or regression calls")

    timestamp = created_at or utcnow().isoformat(timespec="seconds").replace("+00:00", "Z")
    transform_names = sorted({record.name for variant in variants for record in variant.transforms})
    payload: dict[str, Any] = {
        "artifact_type": ARTIFACT_TYPE,
        "augmentation_schema_version": AUGMENTATION_SCHEMA_VERSION,
        "dataset_id": str(source_manifest["dataset_id"]),
        "source_manifest_content_hash": str(source_manifest["content_hash"]),
        "source_manifest_schema_version": str(source_manifest["schema_version"]),
        "created_at": timestamp,
        "synthetic": True,
        "stand_in_label": STAND_IN_LABEL,
        "source_call_ids": source_calls,
        "transform_names": transform_names,
        "variants": sorted(records, key=lambda record: str(record["variant_id"])),
        "evaluation_set_exclusion": {
            "golden_call_ids": list(golden),
            "regression_call_ids": list(regression),
            "synthetic_variants_in_golden": 0,
            "synthetic_variants_in_regression": 0,
        },
        "counts": {
            "variants": len(records),
            "source_calls": len(source_calls),
            "transforms": sum(len(variant.transforms) for variant in variants),
        },
        "promotion_decision": "not_evaluated",
    }
    manifest = {
        **payload,
        "content_hash": hashlib.sha256(canonical_json(payload).encode()).hexdigest(),
    }
    validate_manifest(manifest)

    for text in _iter_strings(manifest):
        if detect_pii(text):
            raise ValidationError(
                "manifest strings must not contain names, emails, phones or cards"
            )
    _assert_no_audio_bytes(manifest)
    return manifest


def validate_manifest(manifest: object) -> None:
    """Validate the internal shape, hash, tagging and exclusion proof.

    Args:
        manifest: Candidate augmentation manifest object.

    Raises:
        ValidationError: If any required field, hash or invariant is wrong.

    """
    if not isinstance(manifest, Mapping):
        raise ValidationError("augmentation manifest must be an object")
    expected = {
        "artifact_type",
        "augmentation_schema_version",
        "dataset_id",
        "source_manifest_content_hash",
        "source_manifest_schema_version",
        "created_at",
        "synthetic",
        "stand_in_label",
        "source_call_ids",
        "transform_names",
        "variants",
        "evaluation_set_exclusion",
        "counts",
        "promotion_decision",
        "content_hash",
    }
    if set(manifest) != expected:
        raise ValidationError("augmentation manifest has missing or extra fields")
    if manifest["artifact_type"] != ARTIFACT_TYPE:
        raise ValidationError("wrong augmentation artifact type")
    if manifest["augmentation_schema_version"] != AUGMENTATION_SCHEMA_VERSION:
        raise ValidationError("unsupported augmentation manifest version")
    if manifest["synthetic"] is not True or manifest["stand_in_label"] != STAND_IN_LABEL:
        raise ValidationError("manifest must be tagged synthetic with the stand-in label")
    if manifest["promotion_decision"] != "not_evaluated":
        raise ValidationError("augmentation must not make a promotion decision")
    payload = {key: value for key, value in manifest.items() if key != "content_hash"}
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    if manifest["content_hash"] != digest:
        raise ValidationError("augmentation content hash does not match the manifest")

    variants = manifest["variants"]
    excluded = manifest["evaluation_set_exclusion"]
    if not isinstance(variants, list) or not isinstance(excluded, Mapping):
        raise ValidationError("variants and exclusion proof must be a list and object")
    identifiers: set[str] = set()
    variant_calls: set[str] = set()
    for item in variants:
        if not isinstance(item, Mapping):
            raise ValidationError("variant records must be objects")
        if item.get("synthetic") is not True:
            raise ValidationError("each variant must carry synthetic: true")
        audio = item.get("audio")
        if not isinstance(audio, Mapping) or audio.get("label") != STAND_IN_LABEL:
            raise ValidationError("variant audio must carry the stand-in label")
        variant_id = str(item.get("variant_id", ""))
        if not variant_id or variant_id in identifiers:
            raise ValidationError("variant IDs must be non-empty and unique")
        identifiers.add(variant_id)
        variant_calls.add(str(item.get("source_call_id", "")))
        transforms = item.get("transforms")
        if not isinstance(transforms, list) or not transforms:
            raise ValidationError("variants need at least one transform record")
        for record in transforms:
            if not isinstance(record, Mapping) or record.get("name") not in ALLOWED_TRANSFORMS:
                raise ValidationError("unknown transform in variant record")
            if not isinstance(record.get("params"), Mapping):
                raise ValidationError("transform params must be an object")
        if "samples" in audio or any(key in audio for key in _AUDIO_BYTE_KEYS):
            raise ValidationError("raw audio bytes must not be serialized")
    if manifest["source_call_ids"] != sorted(variant_calls):
        raise ValidationError("source_call_ids must equal the sorted variant call IDs")
    golden_calls = set(excluded.get("golden_call_ids", ()))
    regression_calls = set(excluded.get("regression_call_ids", ()))
    if variant_calls & (golden_calls | regression_calls):
        raise ValidationError("synthetic variants are present in an evaluation set")
    if excluded.get("synthetic_variants_in_golden") != 0:
        raise ValidationError("golden synthetic variant count must be zero")
    if excluded.get("synthetic_variants_in_regression") != 0:
        raise ValidationError("regression synthetic variant count must be zero")
    if set(manifest["transform_names"]) != {
        str(record["name"]) for item in variants for record in item["transforms"]
    }:
        raise ValidationError("transform_names must cover exactly the recorded transforms")
    counts = manifest["counts"]
    if not isinstance(counts, Mapping):
        raise ValidationError("counts must be an object")
    if counts.get("variants") != len(variants) or counts.get("source_calls") != len(variant_calls):
        raise ValidationError("manifest counts do not match variant records")


def write_manifest(
    manifest: object,
    path: str | Path,
    *,
    settings: AugmentSettings,
    forbidden_paths: Sequence[str | Path] = (),
) -> Path:
    """Validate and write one augmentation manifest to a safe output path.

    Args:
        manifest: Manifest produced by :func:`build_manifest`.
        path: Output JSON path, created with parents when needed.
        settings: Typed settings used to identify read-only roots.
        forbidden_paths: Input paths that must never be overwritten.

    Returns:
        The resolved output path.

    Raises:
        ValidationError: If validation fails or the destination is read-only.

    """
    validate_manifest(manifest)
    destination = _resolve(path)
    guarded = tuple(_resolve(item) for item in READ_ONLY_ROOTS) + tuple(
        _resolve(item) for item in forbidden_paths
    )
    if settings.dataset_manifest_path:
        guarded += (_resolve(settings.dataset_manifest_path),)
    if any(destination == item or item in destination.parents for item in guarded):
        raise ValidationError("augmentation output must not overwrite read-only inputs")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot write augmentation manifest at {destination}: {exc}") from exc
    return destination
