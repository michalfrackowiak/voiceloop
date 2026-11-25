"""Parser trainer plugin, deterministic stand-in adapter and schema gate (P2-07).

All shipped backends are SYNTHETIC / FIXTURE TRAINING STAND-INS. Real Unsloth,
TRL, LLaMA-Factory, GLiNER2 and vLLM implementations are injected through the
documented Protocols; no heavy dependency is imported by default.
"""

from __future__ import annotations

from .adapter import (
    ADAPTER_VERSION,
    AdapterDelta,
    AdapterKind,
    AdapterStore,
    ExampleKind,
    FakeAdapterStore,
    FakeLoraAdapter,
    FakeLoraTrainingBackend,
    FakeVllmRuntime,
    ParserExample,
    ParserRuntime,
    RuntimeStats,
    SlotTrainingBackend,
    canonical_json,
    sha256_text,
)
from .constrained import (
    BookingSchemaDecoder,
    ConstrainedDecoder,
    ConstrainedParserRuntime,
    decode_booking_output,
)
from .settings import ParserBackend, ParserSettings
from .trainer import (
    ExampleLoader,
    Gliner2SlotTrainer,
    InMemoryExampleLoader,
    ParserLoraTrainer,
    SftJsonlExampleLoader,
    validate_training_examples,
)

__all__ = [
    "ADAPTER_VERSION",
    "AdapterDelta",
    "AdapterKind",
    "AdapterStore",
    "BookingSchemaDecoder",
    "ConstrainedDecoder",
    "ConstrainedParserRuntime",
    "ExampleKind",
    "ExampleLoader",
    "FakeAdapterStore",
    "FakeLoraAdapter",
    "FakeLoraTrainingBackend",
    "FakeVllmRuntime",
    "Gliner2SlotTrainer",
    "InMemoryExampleLoader",
    "ParserBackend",
    "ParserExample",
    "ParserLoraTrainer",
    "ParserRuntime",
    "ParserSettings",
    "RuntimeStats",
    "SftJsonlExampleLoader",
    "SlotTrainingBackend",
    "canonical_json",
    "decode_booking_output",
    "sha256_text",
    "validate_training_examples",
]
