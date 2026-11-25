"""P3-01 Whisper LoRA trainer: deterministic SYNTHETIC/FIXTURE STAND-IN only.

No transformer, PEFT, CTranslate2, torch, GPU, network or model download is
imported or executed by this package. Real integrations are injectable
Protocols documented in the README. Integration with the CP-2 parser path is
marked CP-2-pending: the P2-05/P2-06 contracts are consumed read-only.
"""

from __future__ import annotations

from .convert import (
    ConvertedModel,
    ConvertedModelStore,
    Ctranslate2ConvertBackend,
    Ctranslate2Runtime,
    DecodingSettings,
    FakeConvertedModelStore,
    FakeCtranslate2Backend,
    FakeCtranslate2Runtime,
    FakeMergeBackend,
    MergeBackend,
    MergedModel,
    TranscriptionResult,
)
from .dataset import (
    EXPECTED_FIXTURE_HASH,
    EXPECTED_GOLDEN_HASH,
    DataManifest,
    EventSampleRateResolver,
    FakeReplayProvider,
    HashCheck,
    ReadOnlyHashGuard,
    ReplayProvider,
    SampleRateResolver,
    StaticSampleRateResolver,
    WhisperTrainingExample,
    build_data_manifest,
    canonical_json,
    example_from_hf_row,
    mix_replay,
    sha256_text,
)
from .lora import (
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_R,
    MAX_LORA_SCALING,
    STANDIN_WARNING,
    FakeLoraAdapter,
    FakeTransformersPeftBackend,
    LoraConfig,
    LoraDelta,
    LoraTrainingBackend,
    validate_conservative_lora,
)
from .settings import WhisperBackend, WhisperSettings
from .trainer import (
    ExampleLoader,
    HfJsonExampleLoader,
    InMemoryExampleLoader,
    WhisperLoraTrainer,
    validate_training_examples,
)

__all__ = [
    "DEFAULT_LORA_ALPHA",
    "DEFAULT_LORA_R",
    "EXPECTED_FIXTURE_HASH",
    "EXPECTED_GOLDEN_HASH",
    "MAX_LORA_SCALING",
    "STANDIN_WARNING",
    "Ctranslate2ConvertBackend",
    "Ctranslate2Runtime",
    "ConvertedModel",
    "ConvertedModelStore",
    "DataManifest",
    "DecodingSettings",
    "EventSampleRateResolver",
    "ExampleLoader",
    "FakeCtranslate2Backend",
    "FakeCtranslate2Runtime",
    "FakeConvertedModelStore",
    "FakeLoraAdapter",
    "FakeMergeBackend",
    "FakeReplayProvider",
    "FakeTransformersPeftBackend",
    "HashCheck",
    "HfJsonExampleLoader",
    "InMemoryExampleLoader",
    "LoraConfig",
    "LoraDelta",
    "LoraTrainingBackend",
    "MergeBackend",
    "MergedModel",
    "ReadOnlyHashGuard",
    "ReplayProvider",
    "SampleRateResolver",
    "StaticSampleRateResolver",
    "TranscriptionResult",
    "WhisperBackend",
    "WhisperLoraTrainer",
    "WhisperSettings",
    "WhisperTrainingExample",
    "build_data_manifest",
    "canonical_json",
    "example_from_hf_row",
    "mix_replay",
    "sha256_text",
    "validate_conservative_lora",
    "validate_training_examples",
]
