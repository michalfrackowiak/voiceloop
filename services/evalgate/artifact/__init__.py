"""P3-04 conversion and artifact safety checks (synthetic fakes only)."""

from __future__ import annotations

from .artifact import (
    ARTIFACT_URI_PREFIX,
    FakeArtifact,
    ReadOnlyArtifactStore,
    artifact_hash_from_uri,
    create_fake_artifact,
)
from .decoding import (
    DecodeCombination,
    DecodingSettings,
    decode_combinations,
    decoding_hash_from_fields,
    describe_combination,
    language_combinations,
    production_combination,
)
from .probes import (
    ProbeError,
    ProbeInputs,
    ProbeResult,
    ProbeSuiteResult,
    hallucination_summary,
    load_probe_inputs,
    multilingual_summary,
    run_probe_suite,
    verify_inputs,
)
from .report import (
    EXTENSION_KEY,
    EXTENSION_VERSION,
    ArtifactEvaluation,
    ExactArtifactConverter,
    HashedReportEvaluator,
    evaluate_artifact,
    evaluate_checkpoint_uris_do_not_change_result,
    run_exact_artifact_gate,
    validate_stt_report,
    write_stt_report,
)
from .settings import (
    EXPECTED_FIXTURE_SHA256,
    EXPECTED_GOLDEN_CONTENT_HASH,
    ArtifactBehaviour,
    ArtifactSettings,
)

__all__ = [
    "ARTIFACT_URI_PREFIX",
    "EXPECTED_FIXTURE_SHA256",
    "EXPECTED_GOLDEN_CONTENT_HASH",
    "EXTENSION_KEY",
    "EXTENSION_VERSION",
    "ArtifactBehaviour",
    "ArtifactEvaluation",
    "ArtifactSettings",
    "DecodeCombination",
    "DecodingSettings",
    "ExactArtifactConverter",
    "FakeArtifact",
    "HashedReportEvaluator",
    "ProbeError",
    "ProbeInputs",
    "ProbeResult",
    "ProbeSuiteResult",
    "ReadOnlyArtifactStore",
    "artifact_hash_from_uri",
    "create_fake_artifact",
    "decode_combinations",
    "decoding_hash_from_fields",
    "describe_combination",
    "evaluate_artifact",
    "evaluate_checkpoint_uris_do_not_change_result",
    "hallucination_summary",
    "language_combinations",
    "load_probe_inputs",
    "multilingual_summary",
    "production_combination",
    "run_exact_artifact_gate",
    "run_probe_suite",
    "validate_stt_report",
    "verify_inputs",
    "write_stt_report",
]
