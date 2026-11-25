# P4-04 — Offline teacher triage signal (SYNTHETIC/FIXTURE STAND-IN)

This package is an **offline rehearsal** for a teacher-model disagreement signal. It contains **no real teacher model, GPU, network, audio decoding, production queue, alias state or gate logic**. Fixtures are tone/noise/silence carriers with invented labels, so every number produced here is labelled `SYNTHETIC/FIXTURE STAND-IN - NOT A PRODUCTION PRECISION CLAIM`.

## What it does

1. Reads the committed synthetic fixture manifest (`tests/fixtures/turns.jsonl`) read-only and uses it only to locate turn-event JSON files and check lineage. Event STT text is referenced as the production transcript; no audio bytes are read.
2. Runs an injected `TeacherTranscriber` over the turns. The default `DeterministicFixtureTeacher` derives a stand-in transcript from event metadata (`no_speech_prob` / per-word confidence), never from fixture labels or WAV audio. Each transcript gets a SHA-256 content hash.
3. Computes one deterministic token-level disagreement signal, `teacher_transcript_disagreement`, between the teacher and production transcripts. It does not modify P2-01 thresholds, use P2-01's disabled teacher seam, or alter any queue/sampling code.
4. Computes **exploratory synthetic queue arithmetic** with two injected `QueueSelector` policies over the same fixture annotations. These are invented labels, not human-reviewed labels; they are consumed only after selection. Counts, precision-like ratios, recalls, deltas, Wilson 95% intervals and explicit insufficient-`n` notes are included. No improvement is claimed and **P4-04 is NOT DONE**.
5. Verifies the committed fixture text hash and the sealed synthetic golden stand-in seal/hash before and after the run. Golden labels are never loaded; the seal is verified without an audit entry.

## Run / verify

From the repository root (using the existing virtualenv):

```powershell
.\.venv\Scripts\python.exe -m services.triage.teacher.cli --output reports/triage/teacher/report.json
```

Useful flags: `--fixtures`, `--queue-size`, `--signal-weight`, `--teacher-model-id`. On success it writes one internal JSON artifact and prints turns/calls/counts, both synthetic ratios, an `EXPLORATORY ONLY` warning and the `P4-04 is NOT DONE` stand-in warning. It never prints an improvement claim. Known config/validation failures exit 2 with a structured JSON error on stderr.

Scoped checks:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/triage_teacher -q
.\.venv\Scripts\python.exe -m ruff check services/triage/teacher tests/unit/triage_teacher
.\.venv\Scripts\python.exe -m ruff format --check services/triage/teacher tests/unit/triage_teacher
.\.venv\Scripts\python.exe -m mypy services/triage/teacher tests/unit/triage_teacher
```

Full checks:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy .
```

## Public API

Import from `services.triage.teacher`:

- Settings/constants: `TeacherSettings`, `SIGNAL_NAME`, `RUN_LABEL`, `EXPECTED_GOLDEN_CONTENT_HASH`.
- Batch seam: `TurnSource`, `TeacherTranscriber`, `TurnRecord`, `TeacherTranscript`, `BatchResult`, `run_batch`, `canonical_hash`, `fixture_inputs_digest`.
- Fakes: `FixtureTurnSource`, `InMemoryTurnSource`, `DeterministicFixtureTeacher`, `EventProductionTranscriptSource`, `InMemoryProductionTranscriptSource`, `InMemoryBaselineScorer`, `BaselineQueueSelector`, `TeacherQueueSelector`, `TeacherOnlySelector`, `InMemoryReviewedLabelSource`, `FixtureReviewedLabelSource`, `InMemoryReportSink`.
- Production/baseline seams: `ProductionTranscriptSource`, `BaselineScorer`, `QueueSelector`, `ReviewedLabelSource`, `ReportSink`, `P2_01BaselineScorer`.
- Scoring/report: `score_disagreement`, `build_candidates`, `evaluate_precision`, `build_report`, `validate_report`, `verify_read_only_inputs`, `JsonFileReportSink`.
- Orchestration/CLI: import `run_teacher_rehearsal` and `main` from `services.triage.teacher.cli` (the package `__init__` deliberately does not import the CLI module, so `python -m services.triage.teacher.cli` stays clean).

## Injectable seams (fake-only by default)

| Protocol | Method | Default fake | Real seam intent |
| --- | --- | --- | --- |
| `TurnSource` | `iter_turns()` | `FixtureTurnSource` | Caller-supplied, label-free turn events |
| `TeacherTranscriber` | `transcribe(record)`, `model_id` | `DeterministicFixtureTeacher` | Offline batch adapter using exact deployed decoding settings and native sample rate (including 8 kHz); not implemented here |
| `ProductionTranscriptSource` | `transcript_for(record)` | `EventProductionTranscriptSource` | Store of the transcript that actually served production |
| `BaselineScorer` | `score(record)` | `P2_01BaselineScorer` (read-only) | Existing baseline scoring store |
| `QueueSelector` | `select(candidates, limit)`, `name` | baseline / teacher-enhanced fakes | Policy code that consumes the optional signal |
| `ReviewedLabelSource` | `label_for(record)` | `FixtureReviewedLabelSource` | Human-reviewed labels, used only for evaluation |
| `ReportSink` | `write(artifact)` | `JsonFileReportSink` | Internal report storage; not a cross-service schema |

Every fake uses a `fake://` / `memory://` identifier or the explicit `SYNTHETIC/FIXTURE STAND-IN` status. `run_teacher_rehearsal(...)` accepts all seams as keyword arguments for tests; `main(...)` is a thin CLI wrapper over it.

## Observed synthetic result (default run; EXPLORATORY ONLY — not a precision claim)

The committed 36-turn fixture run produced **identical overall queue ratios for both selectors at every reported depth**: `1.0` at depths 1–10 and unchanged thereafter. Both selectors selected 10 overall positives in 10 items at the default cap 10, and 6 transcription-review positives in 10 items, so both headline deltas are `0.0 / 0.0`.

The `depth_curve` is disclosed in full, **post hoc**, with per-depth `n`, TP counts, Wilson 95% intervals and a `confidence_note` stating that `n` is insufficient for any claim. Transcription-review ratios are higher for the enhanced selector at depths **1–9** and equal at depth 10:

| Depth `n` | Baseline TP / n | Enhanced TP / n | Baseline ratio | Enhanced ratio | Δ | Baseline Wilson 95% | Enhanced Wilson 95% |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0/1 | 1/1 | 0.0 | 1.0 | +1.0 | [0.0, 0.793457] | [0.206543, 1.0] |
| 5 | 2/5 | 5/5 | 0.4 | 1.0 | +0.6 | [0.117618, 0.76928] | [0.565509, 1.0] |
| 6 | 3/6 | 5/6 | 0.5 | 0.833333 | +0.333333 | [0.187613, 0.812387] | [0.436491, 0.969947] |
| 9 | 5/9 | 6/9 | 0.555556 | 0.666667 | +0.111111 | [0.266647, 0.811225] | [0.354197, 0.879418] |
| 10 | 6/10 | 6/10 | 0.6 | 0.6 | 0.0 | [0.31267, 0.831822] | [0.31267, 0.831822] |

Every interval pair overlaps. No operating point was pre-declared, the cap `10` is not P2-02's production default (`30`), and the labels are invented fixture annotations rather than human-reviewed labels. These numbers are synthetic arithmetic only and **not a precision-improvement claim**. See "P4-04 status" below.

## P4-04 status — NOT DONE

This package is a **harness-only synthetic rehearsal**. P4-04 is **NOT COMPLETE**
and no production readiness or promotion claim is made. Completion requires all
of the following, none of which exists here:

1. A **pre-declared operating point** (queue cap and selector) fixed before labels are consulted, matching the real review-queue capacity rather than an exploratory fixture cap.
2. An **adequately sized human-reviewed label set** injected through `ReviewedLabelSource`; someone must actually listen to the audio (context.md §2 invariant 2). Fixture `failure_type` values are invented annotations, not reviewed labels.
3. A **real offline teacher model** injected through `TeacherTranscriber`, resolving `audio.uri` at native sample rate (including 8 kHz) with exact deployed decoding settings. The default teacher is a metadata rule.

Until then, words such as "precision", "delta" and "improvement" are reserved for explicitly synthetic arithmetic. No `improved_*` flag or CLI improvement line remains; the depth curve is labelled `EXPLORATORY ONLY` in the report and stdout.

## Honesty and limitations

- The WAV fixtures contain no intelligible speech; the default teacher is a metadata rule, not a transcription model.
- Reviewed labels are invented fixture annotations, not human-reviewed production outcomes. Every ratio and delta is stand-in arithmetic, not a precision or improvement claim.
- The queue-size cap and selectors are fakes; P2-02's real clustering, sampling and blind-item logic is untouched.
- The sealed golden v1 stand-in hash is checked, but its labels are never loaded or tuned on. No public leaderboard is consulted.
- `context.md` §2 invariants 4–8 are respected: synthetic queue arithmetic is not a gate metric, no golden/alias promotion is claimed, and only synthetic fixtures are used. No conflict with `context.md` was identified.
