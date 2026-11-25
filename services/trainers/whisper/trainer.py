"""P3-01 Whisper LoRA trainer and dataset/mix orchestration.

Implements the P2-06 ``TrainerPlugin`` contract read-only and returns a
contract 1.0.0 result whose ``artifact_uri`` addresses the *converted* artifact,
never a raw checkpoint (``context.md`` §2 invariants 3 and 9). Every shipped
component is a deterministic fake; integration with the CP-2 parser path is
marked CP-2-pending because CP-2 has not merged at implementation time.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from services.trainers.core import (
    TrainerPlugin,
    TrainerResult,
    TrainerSettings,
    build_trainer_result,
)
from voiceloop_common import ConfigError, ValidationError

from .convert import (
    ConvertedModelStore,
    Ctranslate2ConvertBackend,
    Ctranslate2Runtime,
    DecodingSettings,
    FakeConvertedModelStore,
    FakeCtranslate2Backend,
    FakeCtranslate2Runtime,
    FakeMergeBackend,
    MergeBackend,
    TranscriptionResult,
)
from .dataset import (
    DataManifest,
    EventSampleRateResolver,
    FakeReplayProvider,
    ReadOnlyHashGuard,
    ReplayProvider,
    SampleRateResolver,
    WhisperTrainingExample,
    build_data_manifest,
    example_from_hf_row,
    mix_replay,
)
from .lora import (
    FakeLoraAdapter,
    FakeTransformersPeftBackend,
    LoraConfig,
    LoraTrainingBackend,
    validate_conservative_lora,
)
from .settings import WhisperSettings

_DIGEST_LENGTH = 64


class ExampleLoader(Protocol):
    """Injectable bridge from a P2-05 dataset manifest to STT examples."""

    def load_examples(self, manifest: Mapping[str, Any]) -> tuple[WhisperTrainingExample, ...]:
        """Return digest-only STT examples for one manifest."""
        ...


class InMemoryExampleLoader:
    """Loader for injected fake examples; ignores manifest contents by design."""

    def __init__(self, examples: Sequence[WhisperTrainingExample] = ()) -> None:
        """Hold an immutable snapshot of examples for later runs."""
        self._examples = tuple(examples)

    def load_examples(self, manifest: Mapping[str, Any]) -> tuple[WhisperTrainingExample, ...]:
        """Return the injected examples (the manifest is accepted but unused)."""
        return self._examples


class HfJsonExampleLoader:
    """Loader for the P2-05 Hugging Face JSON export (``kind == "stt"`` rows).

    Golden/regression call IDs must be supplied by the caller from P1-06/P2-05;
    this loader never opens the sealed set itself. Source sample rates come
    from the injected resolver; audio bytes are never read.
    """

    def __init__(
        self,
        resolver: SampleRateResolver | None = None,
        *,
        path: str | Path | None = None,
        exclude_call_ids: Iterable[str] = (),
    ) -> None:
        """Configure a loader path override, sample-rate seam and exclusions.

        Args:
            resolver: Source-rate seam; fixture event resolver by default.
            path: Optional JSON path; defaults to ``manifest['hf_path']``.
            exclude_call_ids: Golden/regression calls never trained on.

        """
        self._resolver = resolver or EventSampleRateResolver()
        self._path = Path(path) if path is not None else None
        self._excluded = frozenset(exclude_call_ids)

    def load_examples(self, manifest: Mapping[str, Any]) -> tuple[WhisperTrainingExample, ...]:
        """Parse and validate HF-export rows into Whisper examples.

        Raises:
            ValidationError: If the path is missing/unreadable or a row is
                malformed, duplicated or belongs to an excluded call.

        """
        raw_path = self._path if self._path is not None else manifest.get("hf_path")
        if not isinstance(raw_path, (str, Path)) or not str(raw_path):
            raise ValidationError("Whisper loader needs a path or manifest['hf_path']")
        try:
            rows = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValidationError(f"cannot read Whisper HF JSON at {raw_path}: {exc}") from exc
        if not isinstance(rows, list):
            raise ValidationError("Whisper HF export must be a JSON array")
        examples: list[WhisperTrainingExample] = []
        seen: set[str] = set()
        for number, row in enumerate(rows, start=1):
            if not isinstance(row, Mapping) or row.get("kind") != "stt":
                continue
            try:
                example = example_from_hf_row(row, self._resolver)
            except ValidationError as exc:
                raise ValidationError(f"invalid Whisper HF row {number}: {exc}") from exc
            if example.example_id in seen:
                raise ValidationError(f"duplicate Whisper example_id: {example.example_id!r}")
            if example.call_id in self._excluded:
                raise ValidationError(
                    f"Whisper example {example.example_id!r} uses an excluded call"
                )
            seen.add(example.example_id)
            examples.append(example)
        return tuple(examples)


def validate_training_examples(
    examples: Sequence[WhisperTrainingExample],
    excluded_call_ids: Iterable[str] = (),
) -> None:
    """Fail closed on data that would violate dataset or lineage invariants.

    Args:
        examples: Digest-only Whisper examples to check.
        excluded_call_ids: Sealed golden/regression calls that must never train.

    Raises:
        ValidationError: If examples are empty, duplicated, carry an excluded
            call, or have malformed digests/sample rates.

    """
    if not examples:
        raise ValidationError("Whisper training requires at least one example")
    identifiers = [example.example_id for example in examples]
    if len(set(identifiers)) != len(identifiers):
        raise ValidationError("Whisper training examples have duplicate example_id values")
    excluded = frozenset(excluded_call_ids)
    for example in examples:
        if example.call_id in excluded:
            raise ValidationError(f"Whisper example {example.example_id!r} uses an excluded call")
        if (
            len(example.audio_uri_sha256) != _DIGEST_LENGTH
            or len(example.target_text_sha256) != _DIGEST_LENGTH
        ):
            raise ValidationError(f"Whisper example {example.example_id!r} has invalid digests")
        if isinstance(example.sample_rate_hz, bool) or example.sample_rate_hz < 1000:
            raise ValidationError(f"Whisper example {example.example_id!r} has invalid sample rate")


class WhisperLoraTrainer(TrainerPlugin):
    """Conservative LoRA Whisper plugin with fake merge and CTranslate2 export."""

    def __init__(
        self,
        *,
        settings: WhisperSettings | None = None,
        example_loader: ExampleLoader | None = None,
        examples: Sequence[WhisperTrainingExample] | None = None,
        replay_pool: Sequence[WhisperTrainingExample] = (),
        replay_provider: ReplayProvider | None = None,
        training_backend: LoraTrainingBackend | None = None,
        merge_backend: MergeBackend | None = None,
        convert_backend: Ctranslate2ConvertBackend | None = None,
        store: ConvertedModelStore | None = None,
        runtime: Ctranslate2Runtime | None = None,
        sample_rate_resolver: SampleRateResolver | None = None,
        decoding: DecodingSettings | None = None,
        hash_guard: ReadOnlyHashGuard | None = None,
        excluded_call_ids: Iterable[str] | None = None,
    ) -> None:
        """Wire settings and injectable fake/real seams.

        Args:
            settings: Whisper hyperparameters; defaults to conservative values.
            example_loader: Dataset seam; defaults to the P2-05 HF JSON export.
            examples: In-memory fake examples used when no loader is supplied.
            replay_pool: Earlier verified examples available for replay mixing.
            replay_provider: Deterministic replay sampler; seeded fake default.
            training_backend: LoRA backend; a non-fake settings label requires
                an injected implementation (fails closed otherwise).
            merge_backend: Adapter-merge seam; fake default.
            convert_backend: Merge-to-CTranslate2 seam; fake default.
            store: Content-addressed artifact store; fake default.
            runtime: Converted-artifact transcription seam; fake default.
            sample_rate_resolver: Source-rate seam; event JSON fixture default.
            decoding: Recorded decoding settings; PII-free defaults.
            hash_guard: Read-only sealed golden/fixture hash verifier.
            excluded_call_ids: Golden/regression calls; settings value by default.

        Raises:
            ConfigError: If settings request a real backend without injection.

        """
        config = settings or WhisperSettings()
        if config.backend != "fake" and training_backend is None:
            raise ConfigError(
                f"Whisper backend {config.backend!r} requires an injected LoraTrainingBackend"
            )
        self._settings = config
        self._excluded = frozenset(
            config.exclude_call_ids if excluded_call_ids is None else excluded_call_ids
        )
        resolver = sample_rate_resolver or EventSampleRateResolver()
        if example_loader is not None:
            self._loader: ExampleLoader = example_loader
        elif examples is not None:
            self._loader = InMemoryExampleLoader(examples)
        else:
            self._loader = HfJsonExampleLoader(resolver, exclude_call_ids=self._excluded)
        self._replay_pool = tuple(replay_pool)
        self._replay_provider = replay_provider or FakeReplayProvider()
        self._backend: LoraTrainingBackend = training_backend or FakeTransformersPeftBackend()
        self._merge_backend: MergeBackend = merge_backend or FakeMergeBackend()
        self._convert_backend: Ctranslate2ConvertBackend = (
            convert_backend or FakeCtranslate2Backend()
        )
        self._store: ConvertedModelStore = store or FakeConvertedModelStore()
        self._runtime: Ctranslate2Runtime = runtime or FakeCtranslate2Runtime(self._store)
        self._decoding = decoding or DecodingSettings(**config.as_decoding_values())
        self._guard = hash_guard or ReadOnlyHashGuard()
        self._artifact_uri: str | None = None

    @property
    def name(self) -> str:
        """Return the stable registered plugin name."""
        return "whisper-lora"

    def train(self, manifest: Mapping[str, Any], settings: TrainerSettings) -> TrainerResult:
        """Run the deterministic stand-in fine-tune, merge, convert and log.

        Args:
            manifest: P2-05 dataset manifest with a non-empty ``content_hash``.
            settings: P2-06 job settings supplying model, seed and artifact URI.

        Returns:
            A validated contract 1.0.0 result whose ``artifact_uri`` addresses
            the converted artifact, with hyperparameters and scaling in metrics.

        Raises:
            ConfigError: If a real backend label was selected without injection.
            ValidationError: If the manifest, data mix or protected hashes fail.

        """
        before = self._guard.verify()
        content_hash = manifest.get("content_hash")
        if not isinstance(content_hash, str) or not content_hash:
            raise ValidationError("dataset_manifest.content_hash must be a non-empty string")

        primary = self._loader.load_examples(manifest)
        validate_training_examples(primary, self._excluded)
        mixed = mix_replay(
            primary,
            self._replay_pool,
            self._settings.replay_fraction,
            self._settings.replay_seed,
            provider=self._replay_provider,
            excluded_call_ids=self._excluded,
        )
        validate_training_examples(mixed, self._excluded)
        data_manifest = build_data_manifest(
            content_hash,
            mixed,
            replay_fraction=self._settings.replay_fraction,
            preserve_native_sample_rate=self._settings.preserve_native_sample_rate,
        )

        lora_config = validate_conservative_lora(
            self._settings.lora_r,
            self._settings.lora_alpha,
            self._settings.max_lora_scaling,
        )
        params = self._settings.as_training_params()
        adapter: FakeLoraAdapter = self._backend.train(
            mixed,
            model=settings.model,
            seed=settings.seed,
            dataset_hash=content_hash,
            data_manifest_hash=data_manifest.content_hash(),
            params=params,
            lora=lora_config,
        )
        merged = self._merge_backend.merge(adapter, model=settings.model)
        converted = self._convert_backend.convert(
            merged,
            model=settings.model,
            data_manifest=data_manifest,
            decoding=self._decoding,
        )
        artifact_uri = self._store.save(converted, prefix=settings.artifact_uri_prefix)

        after = self._guard.verify()
        self._guard.assert_unchanged(before, after)
        self._artifact_uri = artifact_uri
        metrics = self._metrics(data_manifest, lora_config)
        return build_trainer_result(manifest, settings, artifact_uri=artifact_uri, metrics=metrics)

    def transcribe(self, audio_uri: str) -> TranscriptionResult:
        """Transcribe one digest-only audio URI with the last converted artifact.

        Args:
            audio_uri: Repo-relative or fake audio URI; only its hash is used.

        Returns:
            A digest-only transcription result from the fake runtime.

        Raises:
            ConfigError: If :meth:`train` has not produced an artifact yet.

        """
        if self._artifact_uri is None:
            raise ConfigError("train() must produce a converted artifact before transcribing")
        return self._runtime.transcribe(self._artifact_uri, audio_uri)

    def _metrics(self, data_manifest: DataManifest, lora: LoraConfig) -> dict[str, float]:
        """Return numeric hyperparameters, scaling and mix counts for logging."""
        mix = data_manifest.mix
        return {
            "beam_size": float(self._settings.beam_size),
            "condition_on_previous_text": float(self._decoding.condition_on_previous_text),
            "epochs": float(self._settings.epochs),
            "examples_8k": float(mix["telephony_8k_examples"]),
            "examples_replay": float(mix["replay_examples"]),
            "examples_total": float(len(data_manifest.examples)),
            "fake_train_loss": 0.0,
            "learning_rate": float(self._settings.learning_rate),
            "lora_alpha": float(lora.alpha),
            "lora_max_scaling": float(lora.max_scaling),
            "lora_rank": float(lora.r),
            "lora_scaling": float(lora.scaling),
            "replay_fraction": float(mix["replay_fraction_realised"]),
            "synthetic_examples": float(mix["synthetic_examples"]),
            "temperature": float(self._settings.temperature),
        }
