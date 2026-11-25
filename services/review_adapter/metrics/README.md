# Review UX measurement harness (P2-04)

> [!WARNING]
> **SYNTHETIC/FIXTURE STAND-IN — NOT A PRODUCTION REVIEW AND NOT A UI DECISION.**
> This package replays committed synthetic fixtures through the P2-03 in-memory
> fake review tool. There are no real reviewers, no Label Studio/Argilla service,
> no audio decoding, no network call and no wall-clock benchmark. The report is
> an input to the later P4-03 go/no-go discussion only: it makes no custom-UI
> decision, no promotion, no alias/threshold change and no production-readiness
> claim.

## Layout

| File | Responsibility |
|---|---|
| `__init__.py` | Public package API and explicit exports. |
| `settings.py` | Typed `VOICELOOP_REVIEW_METRICS_` settings; expected seal hash and scripted duration. |
| `events.py` | Injectable protocols and read-only, fixture-backed fakes: result source, clock, candidate pairs, report sink. |
| `timing.py` | Per-item elapsed time from recorded fake start/completion timestamps. |
| `edit_distance.py` | Unit-cost Levenshtein distance from stored hypothesis to scripted correction. |
| `anchoring.py` | Descriptive agreement on explicit blind vs prefilled fixture pairs. |
| `agreement.py` | Scripted inter-reviewer agreement on read-only sealed golden candidates. |
| `report.py` | Internal report assembly, provenance hashes, privacy allowlist and sink handoff. |
| `cli.py` | One-command runner wiring the fixture-only fakes. |
| `README.md` | This file: run, test, contract, public API and fake-only seam notes. |

## Run

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m services.review_adapter.metrics.cli `
  --output C:\Users\asusj\AppData\Local\Temp\opencode\review_metrics_report.json
```

The default output path is `reports/review_metrics/report.json`. An explicit
temporary `--output` is used in verification so no report artifact is added
outside this task's assigned paths. The CLI prints:

- the destination label;
- the number of P2-03 fake review results measured;
- the routing-label distribution;
- an explicit SYNTHETIC/FIXTURE STAND-IN warning and P4-03-only scope note.

`--fixtures`, `--golden-artifact-dir`, `--golden-version`, `--start-at` and
`--duration-seconds` override the typed settings for one run. All environment
variables use the prefix `VOICELOOP_REVIEW_METRICS_`.

## What is measured

| Metric | Definition | Provenance |
|---|---|---|
| Per-item time | `completed_at - started_at` from each P2-03 stored fake result. | Injected fake clock plus the committed scripted duration; never the system clock. |
| Edit distance | Unit-cost Levenshtein insert/delete/substitute distance from `hypothesis` to `corrected_transcript`. | Text stays in memory; only lengths and distances enter the report. |
| Routing distribution | Counts of `stt_wrong`, `parser_wrong`, `both_fine`, `unusable_audio`; absent labels are zero. | Stored result `routing` from the read-only fixture failure type. |
| Prefill anchoring | Exact transcript/slot agreement for explicit blind/prefilled pairs of the same fixture event. | Paired fake results only; no partner is inferred or invented. Descriptive, never causal. |
| Inter-reviewer agreement | Exact transcript/slot agreement between two pseudonymous scripted reviewers on sealed golden candidates. | Sealed `golden.v1` synthetic stand-in; no human reliability claim and no label mutation. |

Expected committed inputs for the default run:

- `tests/fixtures/turns.jsonl` SHA-256:
  `aca23a303954d035245a97f78319df3694d63611b717a2ff2131bc2e0a4453fd`.
- Sealed golden content hash:
  `662cdaf43ee2d5fde03faf5e69fde196114d4f312406d94c00bf53714d03da8e`.
- 36 synthetic fixture turns, each replayed as one blind and one prefilled fake
  review: 72 results, all with the committed 72-second scripted duration.

The report writer hashes the fixture file and every golden artifact file before
and after computation and fails closed if any input changed during the run.
`SealedGoldenCandidateSource` calls the committed P1-06 `verify_seal()` without
audit context, then checks the expected content hash and the synthetic
manifest flags. Golden artifacts are read only.

## Public API

The package exports these symbols from `services.review_adapter.metrics`:

- Constants: `ARTIFACT_TYPE`, `REPORT_SCHEMA_VERSION`, `STATUS`,
  `EXPECTED_GOLDEN_CONTENT_HASH`, `SCRIPTED_DURATION_SECONDS`,
  `BLIND_REVIEWER_ID`, `PREFILLED_REVIEWER_ID`, `NO_CAUSAL_CLAIM`,
  `NOT_HUMAN_RELIABILITY`, `SCRIPTED_GOLDEN_REVIEWERS`,
  `GOLDEN_CANDIDATE_SOURCE`.
- Injectable seams: `ReviewResultSource`, `ReviewResultLike`, `Clock`,
  `CandidatePairSource`, `ReportSink`, `GoldenCandidateSource`.
- Fakes/adapters: `FixtureReviewResultSource`, `SequenceReviewResultSource`,
  `StoreReviewResultSource`, `ResultPairSource`, `FixedClock`, `SequenceClock`,
  `SealedGoldenCandidateSource`, `JsonFileReportSink`, `InMemoryReportSink`.
- Value types: `MetricsInputs`, `CandidateAnswer`, `CandidatePair`,
  `GoldenCandidate`, `ItemTiming`, `TimingSummary`, `ItemEditDistance`,
  `EditDistanceSummary`, `AnchoringMetrics`, `AgreementMetrics`.
- Functions: `levenshtein`, `compute_item_times`, `elapsed_seconds`,
  `summarize_times`, `compute_edit_distances`, `summarize_edit_distances`,
  `answers_agree`, `compare_prefill_anchoring`,
  `compute_inter_reviewer_agreement`, `label_distribution`,
  `validate_report_safety`, `build_metrics_report`, `write_metrics_report`,
  `generate_metrics_report`, `parse_utc`, `rfc3339`, `resolve_path`,
  `sha256_file`, `sha256_directory`.

Substitution points are structural protocols; callers can inject another
result source, fake clock, explicit pair source, golden candidate source or
report sink without changing computation code. No protocol imports a real
review backend.

## Report shape and safety

The internal artifact is `artifact_type = voiceloop.review.metrics` with
`report_schema_version = internal-1.0.0`. It is not a versioned cross-service
contract and does not reuse `eval_report` 1.0.0. Top-level keys are `status`,
`scope`, `provenance`, `settings`, `sections` and `limitations`; sections hold
per-item timing/distance records and summaries, routing-label distribution,
prefill anchoring counts/rates and golden agreement counts/rates.

`validate_report_safety()` runs before every sink write. Keys must be on a fixed
allowlist; values are either fixed explanatory strings or pseudonymous IDs,
paths and hex digests. Hypothesis text, corrected transcripts, slot values,
audio URIs, names, emails, phone numbers, card digits, credentials and warning
free text cannot enter the report through an injected source.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/review_adapter_metrics -q
.\.venv\Scripts\python.exe -m ruff check services/review_adapter/metrics tests/unit/review_adapter_metrics
.\.venv\Scripts\python.exe -m ruff format --check services/review_adapter/metrics tests/unit/review_adapter_metrics
.\.venv\Scripts\python.exe -m mypy services/review_adapter/metrics tests/unit/review_adapter_metrics
```

## Known gaps

- Timing is a scripted-duration rehearsal, not human interaction or clickstream
  measurement.
- Blind and prefilled variants use the same fixture expectation, so their 100%
  scripted agreement is not evidence about a real anchoring effect.
- Sealed golden labels all carry `agreed: true`; the agreement metric verifies a
  scripted rehearsal and cannot estimate human inter-reviewer reliability.
- The report is intentionally only an input to P4-03. It does not decide whether
  a custom UI should be built and does not mark any model/tool production-ready.
