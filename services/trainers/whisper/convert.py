"""P3-01.3 merge-to-convert seam and fake CTranslate2 runtime.

SYNTHETIC / FIXTURE STAND-IN - NOT A REAL FINE-TUNE. No CTranslate2,
faster-whisper or audio decoder is imported. The fake converted artifact is a
canonical JSON descriptor with a content address, the exact decoding settings
used by the fake runtime, and digest-only fixture transcript mappings. Real
implementations are injected through the Protocols below. This package emits
no promotion, alias or threshold side effect (P3-04/P2-09 own those seams).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from voiceloop_common import ValidationError

from .dataset import (
    STANDIN_WARNING,
    DataManifest,
    canonical_json,
    sha256_text,
)
from .lora import FakeLoraAdapter

SCHEMA_VERSION = "1.0.0"
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class DecodingSettings:
    """PII-free decoding settings exercised against the converted artifact.

    Hotwords and initial prompts are deliberately empty in the synthetic
    artifact: P2-08's boost config is a separate runtime seam, and real
    CTranslate2 hotword/initial-prompt behaviour must be validated in P3-04.
    ``condition_on_previous_text`` defaults to false per ``context.md`` §10.
    """

    language: str = "en"
    task: Literal["transcribe", "translate"] = "transcribe"
    beam_size: int = 5
    temperature: float = 0.0
    condition_on_previous_text: bool = False
    initial_prompt: str = ""
    hotwords: tuple[str, ...] = ()
    word_timestamps: bool = False
    vad_filter: bool = False

    def validate(self) -> None:
        """Fail closed on PII-bearing or malformed decoding settings."""
        if not self.language:
            raise ValidationError("decoding language must be non-empty")
        if self.task not in ("transcribe", "translate"):
            raise ValidationError(f"unsupported decoding task: {self.task!r}")
        if (
            isinstance(self.beam_size, bool)
            or not isinstance(self.beam_size, int)
            or self.beam_size < 1
        ):
            raise ValidationError("beam_size must be a positive integer")
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
            raise ValidationError("temperature must be numeric")
        if not 0.0 <= float(self.temperature) <= 1.0:
            raise ValidationError("temperature must be between 0 and 1")
        if self.initial_prompt:
            raise ValidationError("initial_prompt must be empty in a synthetic artifact")
        if self.hotwords:
            raise ValidationError("hotwords must be empty in a synthetic artifact; use P2-08")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable decoding settings object."""
        return {
            "beam_size": self.beam_size,
            "condition_on_previous_text": self.condition_on_previous_text,
            "hotwords": list(self.hotwords),
            "initial_prompt": self.initial_prompt,
            "language": self.language,
            "task": self.task,
            "temperature": self.temperature,
            "vad_filter": self.vad_filter,
            "word_timestamps": self.word_timestamps,
        }


@dataclass(frozen=True, slots=True)
class MergedModel:
    """Fake merged weights produced from a stand-in LoRA adapter."""

    adapter_hash: str
    base_model: str
    lora: dict[str, Any]
    schema_version: str = SCHEMA_VERSION
    synthetic_training_standin: bool = True
    warning: str = STANDIN_WARNING

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical merged-model descriptor."""
        return {
            "adapter_hash": self.adapter_hash,
            "base_model": self.base_model,
            "lora": dict(self.lora),
            "schema_version": self.schema_version,
            "synthetic_training_standin": self.synthetic_training_standin,
            "warning": self.warning,
        }

    def content_hash(self) -> str:
        """Return the SHA-256 content hash of the merged descriptor."""
        return sha256_text(canonical_json(self.to_dict()))


class MergeBackend(Protocol):
    """Injectable merge seam: adapter plus base model to merged weights."""

    def merge(self, adapter: FakeLoraAdapter, *, model: str) -> MergedModel:
        """Merge a LoRA adapter into its base model descriptor."""
        ...


class FakeMergeBackend:
    """Deterministic stand-in that only hashes the adapter descriptor."""

    def merge(self, adapter: FakeLoraAdapter, *, model: str) -> MergedModel:
        """Return a labelled merged descriptor without touching neural weights."""
        return MergedModel(
            adapter_hash=adapter.content_hash(),
            base_model=model,
            lora=dict(adapter.lora),
        )


@dataclass(frozen=True, slots=True)
class ConvertedModel:
    """Content-addressed fake CTranslate2 artifact plus decoding settings."""

    merged_hash: str
    base_model: str
    data_manifest: dict[str, Any]
    decoding: dict[str, Any]
    transcript_index: dict[str, str]
    schema_version: str = SCHEMA_VERSION
    synthetic_training_standin: bool = True
    warning: str = STANDIN_WARNING

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical converted-artifact payload."""
        return {
            "base_model": self.base_model,
            "data_manifest": dict(self.data_manifest),
            "decoding": dict(self.decoding),
            "merged_hash": self.merged_hash,
            "schema_version": self.schema_version,
            "synthetic_training_standin": self.synthetic_training_standin,
            "transcript_index": dict(self.transcript_index),
            "warning": self.warning,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ConvertedModel:
        """Validate a decoded artifact before loading it into the fake runtime."""
        expected = {
            "base_model",
            "data_manifest",
            "decoding",
            "merged_hash",
            "schema_version",
            "synthetic_training_standin",
            "transcript_index",
            "warning",
        }
        if set(raw) != expected:
            raise ValidationError("converted model artifact has unexpected fields")
        if raw["schema_version"] != SCHEMA_VERSION:
            raise ValidationError(
                f"unsupported converted artifact version: {raw['schema_version']!r}"
            )
        merged_hash = raw["merged_hash"]
        if not isinstance(merged_hash, str) or not _DIGEST_RE.fullmatch(merged_hash):
            raise ValidationError("converted artifact merged_hash must be a sha256 digest")
        if not isinstance(raw["base_model"], str):
            raise ValidationError("converted artifact base_model must be a string")
        if not isinstance(raw["data_manifest"], Mapping) or not isinstance(
            raw["decoding"], Mapping
        ):
            raise ValidationError("converted artifact data_manifest/decoding must be objects")
        if not isinstance(raw["transcript_index"], Mapping):
            raise ValidationError("converted artifact transcript_index must be an object")
        for audio_hash, text_hash in raw["transcript_index"].items():
            if not isinstance(audio_hash, str) or not _DIGEST_RE.fullmatch(audio_hash):
                raise ValidationError("transcript_index audio digest is invalid")
            if not isinstance(text_hash, str) or not _DIGEST_RE.fullmatch(text_hash):
                raise ValidationError("transcript_index text digest is invalid")
        decoded = DecodingSettings(
            language=str(raw["decoding"].get("language", "")),
            task=raw["decoding"].get("task", "transcribe"),
            beam_size=int(raw["decoding"].get("beam_size", 5)),
            temperature=float(raw["decoding"].get("temperature", 0.0)),
            condition_on_previous_text=bool(
                raw["decoding"].get("condition_on_previous_text", False)
            ),
            initial_prompt=str(raw["decoding"].get("initial_prompt", "")),
            hotwords=tuple(str(word) for word in raw["decoding"].get("hotwords", ())),
            word_timestamps=bool(raw["decoding"].get("word_timestamps", False)),
            vad_filter=bool(raw["decoding"].get("vad_filter", False)),
        )
        decoded.validate()
        return cls(
            merged_hash=str(merged_hash),
            base_model=str(raw["base_model"]),
            data_manifest=dict(raw["data_manifest"]),
            decoding=decoded.to_dict(),
            transcript_index=dict(raw["transcript_index"]),
            schema_version=str(raw["schema_version"]),
            synthetic_training_standin=bool(raw["synthetic_training_standin"]),
            warning=str(raw["warning"]),
        )

    def content_hash(self) -> str:
        """Return the SHA-256 content hash of the canonical artifact bytes."""
        return sha256_text(canonical_json(self.to_dict()))


class Ctranslate2ConvertBackend(Protocol):
    """Injectable merge-to-convert seam for CTranslate2/faster-whisper."""

    def convert(
        self,
        merged: MergedModel,
        *,
        model: str,
        data_manifest: DataManifest,
        decoding: DecodingSettings,
    ) -> ConvertedModel:
        """Convert a merged model and return a deployable artifact descriptor."""
        ...


class FakeCtranslate2Backend:
    """Deterministic conversion stand-in; no native model format is written."""

    def convert(
        self,
        merged: MergedModel,
        *,
        model: str,
        data_manifest: DataManifest,
        decoding: DecodingSettings,
    ) -> ConvertedModel:
        """Build a digest-only transcript index from the mixed training rows.

        Args:
            merged: Merged descriptor from :class:`FakeMergeBackend`.
            model: Base model ID from core settings.
            data_manifest: Tagged mix manifest with sample-rate/replay counts.
            decoding: PII-free decoding settings recorded on the artifact.

        Returns:
            A labelled fake converted artifact with a content address.

        Raises:
            ValidationError: If decoding settings carry PII-bearing text.

        """
        decoding.validate()
        index = {
            example.audio_uri_sha256: example.target_text_sha256
            for example in data_manifest.examples
        }
        if len(index) != len(data_manifest.examples):
            raise ValidationError("duplicate audio digest in Whisper data manifest")
        return ConvertedModel(
            merged_hash=merged.content_hash(),
            base_model=model,
            data_manifest=data_manifest.to_dict(),
            decoding=decoding.to_dict(),
            transcript_index=index,
        )


class ConvertedModelStore(Protocol):
    """Content-addressed artifact persistence seam."""

    def save(self, model: ConvertedModel, *, prefix: str) -> str:
        """Persist a converted artifact and return its URI."""
        ...

    def load(self, uri: str) -> ConvertedModel:
        """Load and structurally validate a converted artifact."""
        ...


def _uri_digest(uri: str) -> str:
    """Return and validate the trailing SHA-256 digest of a fake artifact URI."""
    digest = uri.rsplit("/", 1)[-1]
    if not _DIGEST_RE.fullmatch(digest):
        raise ValidationError(f"invalid fake converted-model URI: {uri!r}")
    return digest


class FakeConvertedModelStore:
    """In-memory, hash-addressed converted-artifact store for local runs/tests."""

    def __init__(self) -> None:
        """Create an empty artifact store."""
        self._objects: dict[str, str] = {}

    @property
    def uris(self) -> tuple[str, ...]:
        """Return all stored artifact URIs in insertion order."""
        return tuple(self._objects)

    def save(self, model: ConvertedModel, *, prefix: str) -> str:
        """Export canonical converted bytes and return a content-addressed URI.

        Raises:
            ValidationError: If the prefix is empty or the digest collides with
                different bytes.

        """
        if not prefix:
            raise ValidationError("artifact URI prefix must be non-empty")
        digest = model.content_hash()
        uri = f"{prefix.rstrip('/')}/whisper-ctranslate2/{digest}"
        raw = canonical_json(model.to_dict())
        existing = self._objects.get(uri)
        if existing is not None and existing != raw:
            raise ValidationError(f"converted artifact hash collision at {uri!r}")
        self._objects[uri] = raw
        return uri

    def load(self, uri: str) -> ConvertedModel:
        """Load an artifact, rejecting missing, malformed or tampered bytes."""
        digest = _uri_digest(uri)
        try:
            raw = self._objects[uri]
        except KeyError as exc:
            raise ValidationError(f"converted artifact not found: {uri!r}") from exc
        try:
            decoded = json.loads(raw)
        except ValueError as exc:
            raise ValidationError(f"converted artifact is not valid JSON: {exc}") from exc
        if not isinstance(decoded, dict):
            raise ValidationError("converted artifact must be a JSON object")
        model = ConvertedModel.from_dict(decoded)
        if model.content_hash() != digest:
            raise ValidationError("converted artifact content hash does not match its URI")
        return model

    def put_raw(self, uri: str, raw: str) -> None:
        """Inject or tamper with artifact bytes for failure-injection tests."""
        _uri_digest(uri)
        self._objects[uri] = raw


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Digest-only transcription returned by the fake CTranslate2 runtime."""

    audio_uri_sha256: str
    text_sha256: str
    sample_rate_hz: int
    no_speech_prob: float = 0.0
    matched: bool = True
    synthetic: bool = True


class Ctranslate2Runtime(Protocol):
    """Injectable inference seam over the exact deployable artifact."""

    def transcribe(self, uri: str, audio_uri: str) -> TranscriptionResult:
        """Return a transcription for one audio URI using the converted model."""
        ...


class FakeCtranslate2Runtime:
    """Fake runtime that replays digest transcriptions from the converted index.

    No audio bytes are read: the audio URI is hashed and looked up in the
    converted artifact's transcript index. Unknown URIs fail closed.
    """

    def __init__(self, store: ConvertedModelStore) -> None:
        """Wire the runtime to a converted-artifact store."""
        self._store = store

    def transcribe(self, uri: str, audio_uri: str) -> TranscriptionResult:
        """Return the digest transcription for one fixture-style audio URI.

        Raises:
            ValidationError: If the artifact is invalid, unlabelled, or has no
                transcript for the supplied URI.

        """
        model = self._store.load(uri)
        if not model.synthetic_training_standin:
            raise ValidationError("fake runtime refuses artifacts without the stand-in label")
        audio_hash = sha256_text(audio_uri)
        if audio_hash not in model.transcript_index:
            raise ValidationError("audio URI is not present in the fake converted model")
        rates: dict[str, int] = {}
        for example in model.data_manifest.get("examples", []):
            if isinstance(example, Mapping):
                digest = example.get("audio_uri_sha256")
                rate = example.get("sample_rate_hz")
                if isinstance(digest, str) and isinstance(rate, int) and not isinstance(rate, bool):
                    rates[digest] = rate
        return TranscriptionResult(
            audio_uri_sha256=audio_hash,
            text_sha256=model.transcript_index[audio_hash],
            sample_rate_hz=rates.get(audio_hash, 0),
        )
