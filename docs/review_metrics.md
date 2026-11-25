# Review UX metrics (P2-04)

> [!WARNING]
> **SYNTHETIC/FIXTURE STAND-IN — NOT HUMAN REVIEW EVIDENCE.** Every measurement
> comes from committed synthetic fixtures replayed through the P2-03 in-memory
> fake review tool with an injected clock. No real reviewer, UI, clickstream,
> Label Studio/Argilla service, network call, audio decode or wall-clock timing
> is involved. The report is an input to the later P4-03 go/no-go discussion
> only. It makes no custom-UI decision, no promotion, no alias/threshold change
> and no production-readiness claim.

## Purpose and scope

P2-04 measures the *shape* of the review workflow using P2-03 fake results so
that a later lead-owned decision has deterministic evidence. It does not build
or select a review UI and does not decide whether one is worth building.

| Roadmap subtask | Implementation | Output |
|---|---|---|
| P2-04.1 | `timing.py`, `edit_distance.py`, `report.py` | Per-item time, hypothesis-to-correction Levenshtein distance, routing-label distribution. |
| P2-04.2 | `anchoring.py`, `events.py` | Blind vs prefilled agreement on explicit paired fixture items, with counts, rates and a no-causation note. |
| P2-04.3 | `agreement.py` | Scripted inter-reviewer agreement on synthetic golden-set candidates; sealed artifacts read only. |

## Generating the report

```powershell
.\.venv\Scripts\python.exe -m services.review_adapter.metrics.cli `
  --output C:\Users\asusj\AppData\Local\Temp\opencode\review_metrics_report.json
```

Typed settings use the prefix `VOICELOOP_REVIEW_METRICS_`. The default output
path is `reports/review_metrics/report.json`; an explicit temporary output was
used in verification so this task adds no report artifact outside its assigned
paths. Inputs are read only: `tests/fixtures/turns.jsonl` and
`services/datasets/golden/artifacts/`.

Pinned input hashes:

| Input | SHA-256 |
|---|---|
| `tests/fixtures/turns.jsonl` | `aca23a303954d035245a97f78319df3694d63611b717a2ff2131bc2e0a4453fd` |
| Sealed golden `content_hash` | `662cdaf43ee2d5fde03faf5e69fde196114d4f312406d94c00bf53714d03da8e` |

The package verifies the seal through the committed P1-06 verifier and checks
the expected content hash before using golden labels. It hashes the fixture file
and every golden artifact file before and after computation; a mismatch fails
the whole report. Golden artifacts are never amended, tuned, trained on or
written to.

## Metric definitions

### Per-item time

For each stored P2-03 result, elapsed seconds are
`completed_at - started_at`. Both timestamps come from the injected fake clock:
the start is a deterministic sequence value and the completion is start plus the
committed 72-second scripted duration. The summary reports count, total, mean,
median, minimum and maximum. No system clock is consulted.

### Edit distance

`levenshtein(hypothesis, corrected_transcript)` uses unit-cost character
insertion, deletion and substitution over Unicode code points, without case
folding or normalisation. Per-item records keep pseudonymous lineage, the raw
distance and the two string lengths; no transcript text is emitted.

### Routing-label distribution

Counts cover the four OpenAPI 1.0.0 labels from
`schemas/review_adapter.openapi.yaml`: `stt_wrong`, `parser_wrong`,
`both_fine`, `unusable_audio`. Labels with no fixtures still appear with zero.
An unknown label fails closed.

### Prefill anchoring comparison

`ResultPairSource` groups fake results by pseudonymous `event_id` and yields a
pair only when exactly one blind and one prefilled result exist. Agreement means
exact equality of scripted corrected transcripts and slot objects. Reported
counts, rates (count / paired items) and `unpaired_events` are descriptive. The
fixed interpretation string is `descriptive_only_no_causal_claim`: identical
scripted answers cannot show that a UI prefill caused, reduced or increased
anything.

### Inter-reviewer agreement on golden candidates

`SealedGoldenCandidateSource` reads the sealed synthetic `golden.v1` labels,
creates two pseudonymous scripted answers per candidate from the sealed label,
and compares transcripts and slots exactly. The manifest must remain
`synthetic`, a `workflow_stand_in` and in `simulated_synthetic_labels` mode.
The fixed interpretation is `not_human_inter_reviewer_reliability`.

## Deterministic default-run evidence

The verification run produced these fixture-only values:

| Section | Value |
|---|---|
| Fake P2-03 results | 72 (36 fixture turns × blind/prefilled variants) |
| Per-item time | 72.0 s each; total 5184.0 s; mean/median/min/max 72.0 s |
| Edit distance | total 106; mean 1.4722222222222223; median 0; max 22 |
| Routing labels | `both_fine` 52, `parser_wrong` 8, `unusable_audio` 8, `stt_wrong` 4 |
| Prefill anchoring | 36 paired items, 0 unpaired; transcript/slot/both agreement 36/36 (1.0) |
| Golden agreement | 36 candidates; transcript/slot/both agreement 36/36 (1.0) |

These numbers describe scripted fixture replays. In particular, both anchoring
variants are driven from the same committed fixture expectation, and every
sealed golden label is marked `agreed: true`; neither rate is a human-quality
estimate.

## Report contract and privacy

The artifact is internal: `artifact_type = voiceloop.review.metrics`,
`report_schema_version = internal-1.0.0`. It is not a versioned cross-service
contract and does not reuse `eval_report` 1.0.0. Top-level keys are `status`,
`scope`, `provenance`, `settings`, `sections` and `limitations`.

`validate_report_safety()` allows only known keys, fixed explanatory strings and
pseudonymous IDs/paths/hex digests. The report cannot contain hypothesis or
corrected text, slot values, audio URIs, names, emails, phone numbers, card
digits, credentials or warning text. The sink is injectable; the default sink
writes one sorted UTF-8 JSON file and is the only filesystem write in the
package.

## Fake-only seams

| Protocol | Default fake | Replacement point |
|---|---|---|
| `ReviewResultSource` | `FixtureReviewResultSource` replays P2-03 `InMemoryReviewTool` results. | Supply already-captured P2-03-style fake results; no real tool is called. |
| `Clock` | `SequenceClock` for result generation; `FixedClock` for report timestamp. | Supply deterministic timestamps only. |
| `CandidatePairSource` | `ResultPairSource` from explicit result variants. | Supply explicit pairs; no pair is fabricated. |
| `GoldenCandidateSource` | `SealedGoldenCandidateSource` reads sealed synthetic labels. | Supply scripted synthetic candidates; sealed artifacts stay read only. |
| `ReportSink` | `JsonFileReportSink`, with `InMemoryReportSink` for tests. | Direct the validated report to another test/documentation destination. |

## Interpretation limits and follow-ups

- No clickstream, UI timing, reviewer workload or real error-rate claim is
  available from this report.
- 100% scripted anchoring agreement reflects fixture scripting, not absence of
  anchoring bias in a future UI.
- 100% scripted golden agreement is a fixture rehearsal, not inter-annotator
  reliability.
- P4-03 owns the later go/no-go decision; this task records
  `custom_ui_decision: not_made`, `promotion: not_evaluated`,
  `alias_threshold_change: none` and `production_readiness: not_claimed`.
- No conflict with `context.md` §2 invariants was found. The roadmap's phrase
  "used later to decide whether a custom UI is worth building" is treated as
  lead/P4-03 input, not an authorisation for P2-04 to make that decision.
- Final full-repo verification was clean; intermediate unrelated redaction-lane
  findings were resolved externally and were not edited here. Exact outputs are
  recorded in `docs/handoffs/P2-04.md`.
