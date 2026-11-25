"""P3-01 dataset seam: hashed examples, deterministic replay and 8 kHz tags.

The package consumes P2-05 dataset manifests read-only. HF-export rows
(``kind == "stt"``) are converted into compact examples that carry digests,
not transcripts or audio bytes. Replay selection is seeded and explicit, and
8 kHz telephony rows are tagged as bandwidth-limited without resampling, per
``context.md`` §16. Every emitted mix manifest is labelled
``SYNTHETIC/FIXTURE STAND-IN - NOT A REAL FINE-TUNE``.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol

from voiceloop_common import ValidationError

EXPECTED_GOLDEN_HASH = "662cdaf43ee2d5fde03faf5e69fde196114d4f312406d94c00bf53714d03da8e"
EXPECTED_FIXTURE_HASH = "aca23a303954d035245a97f78319df3694d63611b717a2ff2131bc2e0a4453fd"
STANDIN_WARNING = "SYNTHETIC/FIXTURE STAND-IN - NOT A REAL FINE-TUNE"
DATA_MANIFEST_VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[3]
_DIGEST_LENGTH = 64


def canonical_json(value: object) -> str:
    """Return stable compact JSON text for hashing and artifact export."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    """Return the lowercase SHA-256 hex digest of UTF-8 text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class WhisperTrainingExample:
    """One PII-free STT training example used by the fake backend.

    Text and audio URIs are stored only as SHA-256 digests. ``sample_rate_hz``
    is the *source* rate: the fake preprocessor preserves it verbatim and tags
    bandwidth rather than fabricating higher-frequency content.
    """

    example_id: str
    call_id: str
    audio_uri_sha256: str
    target_text_sha256: str
    duration_ms: int
    sample_rate_hz: int
    replay: bool
    synthetic: bool
    source: Literal["primary", "replay"] = "primary"

    @property
    def telephony_band(self) -> bool:
        """Return whether the source rate is 8 kHz or narrower."""
        return self.sample_rate_hz <= 8000

    @property
    def effective_bandwidth_hz(self) -> int:
        """Return the Nyquist bandwidth implied by the preserved source rate."""
        return self.sample_rate_hz // 2

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable, PII-free example record."""
        return {
            "audio_uri_sha256": self.audio_uri_sha256,
            "call_id": self.call_id,
            "duration_ms": self.duration_ms,
            "effective_bandwidth_hz": self.effective_bandwidth_hz,
            "example_id": self.example_id,
            "replay": self.replay,
            "resampled": False,
            "sample_rate_hz": self.sample_rate_hz,
            "sample_rate_policy": "preserve_native",
            "source": self.source,
            "synthetic": self.synthetic,
            "target_text_sha256": self.target_text_sha256,
            "telephony_band": self.telephony_band,
        }


class SampleRateResolver(Protocol):
    """Injectable source-sample-rate seam for fake and real preprocessors."""

    def resolve(self, audio_uri: str) -> int:
        """Return the source sample rate in Hz for one audio URI."""
        ...


class StaticSampleRateResolver:
    """Sample-rate resolver backed by an injected URI-to-Hz mapping."""

    def __init__(self, rates: Mapping[str, int], *, default_hz: int = 16000) -> None:
        """Hold a mapping of audio URIs to their source sample rates.

        Args:
            rates: Audio URI to positive sample rate in Hz.
            default_hz: Rate used for URIs absent from ``rates``.

        Raises:
            ValidationError: If any rate is not a positive integer.

        """
        for uri, rate in rates.items():
            if isinstance(rate, bool) or not isinstance(rate, int) or rate < 1000:
                raise ValidationError(f"invalid sample rate for {uri!r}: {rate!r}")
        if isinstance(default_hz, bool) or not isinstance(default_hz, int) or default_hz < 1000:
            raise ValidationError(f"invalid default sample rate: {default_hz!r}")
        self._rates = dict(rates)
        self._default_hz = default_hz

    def resolve(self, audio_uri: str) -> int:
        """Return the mapped or default source rate for one audio URI."""
        return self._rates.get(audio_uri, self._default_hz)


class EventSampleRateResolver:
    """Fixture stand-in resolver that reads sample rates from turn-event JSON.

    The audio URI is translated from a repo-relative ``.../audio/...wav`` path
    to its ``.../events/...json`` sibling. Only event metadata is read: audio
    bytes are never opened, decoded or resampled.
    """

    def __init__(self, root: Path | None = None) -> None:
        """Set the repository root used to resolve repo-relative URIs."""
        self._root = root or REPO_ROOT

    def resolve(self, audio_uri: str) -> int:
        """Read ``audio.sample_rate_hz`` from the matching turn event.

        Raises:
            ValidationError: If the URI is not a WAV path, the event is missing
                or malformed, or the rate is not a positive integer.

        """
        if not audio_uri.endswith(".wav") or "/audio/" not in audio_uri:
            raise ValidationError(f"cannot resolve sample rate for URI {audio_uri!r}")
        event_uri = audio_uri.replace("/audio/", "/events/")[: -len(".wav")] + ".json"
        path = Path(event_uri)
        if not path.is_absolute():
            path = self._root / path
        try:
            decoded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValidationError(f"cannot read turn event at {event_uri}: {exc}") from exc
        audio = decoded.get("audio") if isinstance(decoded, dict) else None
        rate = audio.get("sample_rate_hz") if isinstance(audio, Mapping) else None
        if isinstance(rate, bool) or not isinstance(rate, int) or rate < 1000:
            raise ValidationError(f"invalid audio.sample_rate_hz in {event_uri}: {rate!r}")
        return rate


def example_from_hf_row(
    row: Mapping[str, Any], resolver: SampleRateResolver
) -> WhisperTrainingExample:
    """Convert one P2-05 HF-export STT row into a PII-free training example.

    Args:
        row: A decoded ``DatasetExample.to_dict()`` row from the HF export.
        resolver: Source sample-rate seam.

    Returns:
        A validated example whose text and audio URI are digest-only.

    Raises:
        ValidationError: If required fields are missing or malformed, or the
            resolved sample rate is invalid.

    """
    if row.get("kind") != "stt":
        raise ValidationError("Whisper training accepts only P2-05 kind='stt' rows")
    example_id = str(row.get("example_id", "")).strip()
    call_id = str(row.get("call_id", "")).strip()
    audio_uri = row.get("audio_uri")
    target_text = row.get("target_text")
    duration_ms = row.get("duration_ms")
    if not example_id or not call_id:
        raise ValidationError("Whisper example requires example_id and call_id")
    if not isinstance(audio_uri, str) or not audio_uri:
        raise ValidationError(f"Whisper example {example_id!r} requires an audio URI")
    if not isinstance(target_text, str) or not target_text.strip():
        raise ValidationError(f"Whisper example {example_id!r} requires a non-empty target text")
    if isinstance(duration_ms, bool) or not isinstance(duration_ms, int) or duration_ms < 0:
        raise ValidationError(f"Whisper example {example_id!r} requires a non-negative duration_ms")
    sample_rate_hz = resolver.resolve(audio_uri)
    if (
        isinstance(sample_rate_hz, bool)
        or not isinstance(sample_rate_hz, int)
        or sample_rate_hz < 1000
    ):
        raise ValidationError(f"invalid sample rate for {example_id!r}: {sample_rate_hz!r}")
    replay = bool(row.get("replay", False))
    return WhisperTrainingExample(
        example_id=example_id,
        call_id=call_id,
        audio_uri_sha256=sha256_text(audio_uri),
        target_text_sha256=sha256_text(target_text),
        duration_ms=duration_ms,
        sample_rate_hz=sample_rate_hz,
        replay=replay,
        synthetic=bool(row.get("synthetic", True)),
        source="replay" if replay else "primary",
    )


class ReplayProvider(Protocol):
    """Injectable, deterministic replay sampler over earlier verified turns."""

    def select(
        self,
        primary: Sequence[WhisperTrainingExample],
        pool: Sequence[WhisperTrainingExample],
        fraction: float,
        seed: int,
        excluded_call_ids: frozenset[str],
    ) -> tuple[WhisperTrainingExample, ...]:
        """Return replay examples selected without call or example leakage."""
        ...


class FakeReplayProvider:
    """Seeded stand-in replay sampler; no randomness outside ``random.Random``."""

    def select(
        self,
        primary: Sequence[WhisperTrainingExample],
        pool: Sequence[WhisperTrainingExample],
        fraction: float,
        seed: int,
        excluded_call_ids: frozenset[str],
    ) -> tuple[WhisperTrainingExample, ...]:
        """Select a deterministic floor(fraction * primary) replay subset.

        Calls already present in primary and all excluded calls are skipped so
        a call never spans replay and primary rows (§2 invariant 7).
        """
        if not primary or fraction <= 0.0 or not pool:
            return ()
        primary_calls = {example.call_id for example in primary}
        primary_ids = {example.example_id for example in primary}
        candidates = sorted(
            (
                example
                for example in pool
                if example.call_id not in primary_calls
                and example.call_id not in excluded_call_ids
                and example.example_id not in primary_ids
            ),
            key=lambda example: example.example_id,
        )
        count = min(len(candidates), int(fraction * len(primary)))
        if count <= 0:
            return ()
        chosen = random.Random(seed).sample(candidates, count)
        return tuple(
            sorted(
                (replace(example, replay=True, source="replay") for example in chosen),
                key=lambda example: example.example_id,
            )
        )


def mix_replay(
    primary: Sequence[WhisperTrainingExample],
    pool: Sequence[WhisperTrainingExample],
    fraction: float,
    seed: int,
    *,
    provider: ReplayProvider | None = None,
    excluded_call_ids: frozenset[str] = frozenset(),
) -> tuple[WhisperTrainingExample, ...]:
    """Combine primary rows with explicitly tagged, deterministic replay rows.

    Raises:
        ValidationError: If primary rows are duplicated, belong to an excluded
            call, or replay selection duplicates an example ID.

    """
    if not 0.0 <= fraction <= 1.0:
        raise ValidationError("replay_fraction must be between 0 and 1")
    seen: set[str] = set()
    for example in primary:
        if example.example_id in seen:
            raise ValidationError(f"duplicate Whisper example_id: {example.example_id!r}")
        seen.add(example.example_id)
        if example.call_id in excluded_call_ids:
            raise ValidationError(f"Whisper example {example.example_id!r} uses an excluded call")
    pool_ids = [example.example_id for example in pool]
    if len(pool_ids) != len(set(pool_ids)):
        raise ValidationError("replay pool has duplicate Whisper example_id values")
    if seen.intersection(pool_ids):
        raise ValidationError("duplicate Whisper example_id between primary and replay pool")
    sampler = provider or FakeReplayProvider()
    replay = sampler.select(primary, pool, fraction, seed, excluded_call_ids)
    combined = tuple(primary) + replay
    identifiers = [example.example_id for example in combined]
    if len(identifiers) != len(set(identifiers)):
        raise ValidationError("replay selection duplicated a Whisper example_id")
    if any(example.call_id in excluded_call_ids for example in replay):
        raise ValidationError("replay selection used an excluded call")
    return tuple(sorted(combined, key=lambda example: example.example_id))


@dataclass(frozen=True, slots=True)
class DataManifest:
    """Task-local, PII-free mix manifest consumed by the fake conversion seam.

    This is an internal artifact descriptor, not a new cross-service contract;
    the P2-05 dataset manifest schema is consumed unchanged.
    """

    source_dataset_hash: str
    examples: tuple[WhisperTrainingExample, ...]
    mix: dict[str, Any]
    schema_version: str = DATA_MANIFEST_VERSION
    synthetic_training_standin: bool = True
    warning: str = STANDIN_WARNING

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical, JSON-serialisable manifest payload."""
        return {
            "examples": [example.to_dict() for example in self.examples],
            "mix": dict(self.mix),
            "schema_version": self.schema_version,
            "source_dataset_hash": self.source_dataset_hash,
            "synthetic_training_standin": self.synthetic_training_standin,
            "warning": self.warning,
        }

    def content_hash(self) -> str:
        """Return the SHA-256 content hash of the canonical manifest bytes."""
        return sha256_text(canonical_json(self.to_dict()))


def build_data_manifest(
    source_dataset_hash: str,
    examples: Sequence[WhisperTrainingExample],
    *,
    replay_fraction: float,
    preserve_native_sample_rate: bool,
) -> DataManifest:
    """Build the tagged mix manifest for one stand-in training run.

    Args:
        source_dataset_hash: ``content_hash`` from the P2-05 dataset manifest.
        examples: Primary plus replay examples, in any order.
        replay_fraction: Requested replay fraction (recorded verbatim).
        preserve_native_sample_rate: Whether 8 kHz rows are preserved as-is.

    Returns:
        A content-addressed manifest with replay, sample-rate and synthetic
        counts; no transcript text, audio bytes or contact details are stored.

    """
    if not source_dataset_hash:
        raise ValidationError("source_dataset_hash must be non-empty")
    if not examples:
        raise ValidationError("Whisper training requires at least one example")
    if not preserve_native_sample_rate:
        raise ValidationError(
            "the fake preprocessor only supports preserve-native sample rate; "
            "inject a real backend for resampling"
        )
    rates: dict[str, int] = {}
    for example in examples:
        key = str(example.sample_rate_hz)
        rates[key] = rates.get(key, 0) + 1
    replay_count = sum(1 for example in examples if example.replay)
    mix: dict[str, Any] = {
        "primary_examples": sum(1 for example in examples if not example.replay),
        "preserve_native_sample_rate": True,
        "replay_examples": replay_count,
        "replay_fraction_realised": replay_count / len(examples),
        "replay_fraction_requested": replay_fraction,
        "sample_rates_hz": dict(sorted(rates.items(), key=lambda item: int(item[0]))),
        "synthetic_examples": sum(1 for example in examples if example.synthetic),
        "telephony_8k_examples": sum(1 for example in examples if example.telephony_band),
    }
    ordered = tuple(sorted(examples, key=lambda example: example.example_id))
    return DataManifest(
        source_dataset_hash=source_dataset_hash,
        examples=ordered,
        mix=mix,
    )


@dataclass(frozen=True, slots=True)
class HashCheck:
    """Read-only verification result for the sealed golden and fixture hashes."""

    golden_content_hash: str
    fixture_sha256: str
    golden_ok: bool
    fixture_ok: bool

    @property
    def ok(self) -> bool:
        """Return whether both expected hashes matched."""
        return self.golden_ok and self.fixture_ok

    def to_dict(self) -> dict[str, Any]:
        """Return a PII-free, loggable hash report."""
        return {
            "fixture_ok": self.fixture_ok,
            "fixture_sha256": self.fixture_sha256,
            "golden_content_hash": self.golden_content_hash,
            "golden_ok": self.golden_ok,
            "ok": self.ok,
        }


@dataclass(frozen=True, slots=True)
class ReadOnlyHashGuard:
    """Verifies protected artifact hashes before and after a training run.

    Only reads the sealed golden manifest/seal and ``tests/fixtures/turns.jsonl``.
    It never calls the P1-06 seal verification API (which appends an audit
    entry) and never writes to protected paths.
    """

    root: Path = REPO_ROOT
    golden_seal_path: Path = Path("services/datasets/golden/artifacts/golden.v1.seal.json")
    golden_manifest_path: Path = Path("services/datasets/golden/artifacts/golden.v1.json")
    fixture_path: Path = Path("tests/fixtures/turns.jsonl")
    expected_golden_hash: str = EXPECTED_GOLDEN_HASH
    expected_fixture_hash: str = EXPECTED_FIXTURE_HASH

    def _resolve(self, path: Path) -> Path:
        """Resolve a repo-relative path against the configured root."""
        return path if path.is_absolute() else self.root / path

    def _golden_hash(self) -> tuple[str, bool]:
        """Read both golden content hashes and compare them with the expectation."""
        hashes: list[str] = []
        for path in (self.golden_seal_path, self.golden_manifest_path):
            try:
                decoded = json.loads(self._resolve(path).read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValidationError(f"cannot read sealed golden artifact {path}: {exc}") from exc
            value = decoded.get("content_hash") if isinstance(decoded, dict) else None
            if not isinstance(value, str):
                raise ValidationError(f"sealed golden artifact {path} has no content_hash")
            hashes.append(value)
        return hashes[0], len(set(hashes)) == 1 and hashes[0] == self.expected_golden_hash

    def _fixture_hash(self) -> tuple[str, bool]:
        """Hash the fixture labels manifest bytes without modifying it."""
        try:
            digest = hashlib.sha256(self._resolve(self.fixture_path).read_bytes()).hexdigest()
        except OSError as exc:
            raise ValidationError(f"cannot read fixture manifest: {exc}") from exc
        return digest, digest == self.expected_fixture_hash

    def check(self) -> HashCheck:
        """Return a fresh read-only hash report."""
        golden_hash, golden_ok = self._golden_hash()
        fixture_hash, fixture_ok = self._fixture_hash()
        return HashCheck(
            golden_content_hash=golden_hash,
            fixture_sha256=fixture_hash,
            golden_ok=golden_ok,
            fixture_ok=fixture_ok,
        )

    def verify(self) -> HashCheck:
        """Fail closed if either protected artifact hash differs from expected."""
        check = self.check()
        if not check.ok:
            raise ValidationError(f"protected artifact hashes changed: {check.to_dict()}")
        return check

    def assert_unchanged(self, before: HashCheck, after: HashCheck) -> None:
        """Fail closed if a run altered either protected artifact hash.

        Raises:
            ValidationError: If the before/after reports differ in any field.

        """
        if before != after:
            raise ValidationError(
                "protected artifact hashes changed during training: "
                f"before={before.to_dict()} after={after.to_dict()}"
            )
