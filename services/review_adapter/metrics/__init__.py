"""P2-04 review UX metrics: fixture-only, injected, synthetic stand-in.

Nothing in this package starts a review tool, decodes audio, opens a network
connection or reads the system clock. Every result is a P2-03 fake replayed from
committed synthetic fixtures, and every report is labelled as a fixture
stand-in for the later P4-03 go/no-go discussion only.
"""

from __future__ import annotations

from .agreement import (
    GOLDEN_CANDIDATE_SOURCE,
    NOT_HUMAN_RELIABILITY,
    SCRIPTED_GOLDEN_REVIEWERS,
    AgreementMetrics,
    GoldenCandidate,
    GoldenCandidateSource,
    SealedGoldenCandidateSource,
    compute_inter_reviewer_agreement,
)
from .anchoring import (
    NO_CAUSAL_CLAIM,
    AnchoringMetrics,
    answers_agree,
    compare_prefill_anchoring,
)
from .edit_distance import (
    EditDistanceSummary,
    ItemEditDistance,
    compute_edit_distances,
    levenshtein,
    summarize_edit_distances,
)
from .events import (
    BLIND_REVIEWER_ID,
    PREFILLED_REVIEWER_ID,
    CandidateAnswer,
    CandidatePair,
    CandidatePairSource,
    Clock,
    FixedClock,
    FixtureReviewResultSource,
    InMemoryReportSink,
    JsonFileReportSink,
    ReportSink,
    ResultPairSource,
    ReviewResultLike,
    ReviewResultSource,
    SequenceClock,
    SequenceReviewResultSource,
    StoreReviewResultSource,
    parse_utc,
    resolve_path,
    rfc3339,
    sha256_directory,
    sha256_file,
)
from .report import (
    ARTIFACT_TYPE,
    REPORT_SCHEMA_VERSION,
    STATUS,
    MetricsInputs,
    build_metrics_report,
    generate_metrics_report,
    label_distribution,
    validate_report_safety,
    write_metrics_report,
)
from .settings import (
    EXPECTED_GOLDEN_CONTENT_HASH,
    SCRIPTED_DURATION_SECONDS,
    ReviewMetricsSettings,
)
from .timing import (
    ItemTiming,
    TimingSummary,
    compute_item_times,
    elapsed_seconds,
    summarize_times,
)

__all__ = [
    "ARTIFACT_TYPE",
    "BLIND_REVIEWER_ID",
    "EXPECTED_GOLDEN_CONTENT_HASH",
    "GOLDEN_CANDIDATE_SOURCE",
    "NO_CAUSAL_CLAIM",
    "NOT_HUMAN_RELIABILITY",
    "PREFILLED_REVIEWER_ID",
    "REPORT_SCHEMA_VERSION",
    "SCRIPTED_DURATION_SECONDS",
    "SCRIPTED_GOLDEN_REVIEWERS",
    "STATUS",
    "AgreementMetrics",
    "AnchoringMetrics",
    "CandidateAnswer",
    "CandidatePair",
    "CandidatePairSource",
    "Clock",
    "EditDistanceSummary",
    "FixtureReviewResultSource",
    "FixedClock",
    "GoldenCandidate",
    "GoldenCandidateSource",
    "InMemoryReportSink",
    "ItemEditDistance",
    "ItemTiming",
    "JsonFileReportSink",
    "MetricsInputs",
    "ReportSink",
    "ResultPairSource",
    "ReviewResultLike",
    "ReviewResultSource",
    "ReviewMetricsSettings",
    "SealedGoldenCandidateSource",
    "SequenceClock",
    "SequenceReviewResultSource",
    "StoreReviewResultSource",
    "TimingSummary",
    "answers_agree",
    "build_metrics_report",
    "compare_prefill_anchoring",
    "compute_edit_distances",
    "compute_inter_reviewer_agreement",
    "compute_item_times",
    "elapsed_seconds",
    "generate_metrics_report",
    "label_distribution",
    "levenshtein",
    "parse_utc",
    "resolve_path",
    "rfc3339",
    "sha256_directory",
    "sha256_file",
    "summarize_edit_distances",
    "summarize_times",
    "validate_report_safety",
    "write_metrics_report",
]
