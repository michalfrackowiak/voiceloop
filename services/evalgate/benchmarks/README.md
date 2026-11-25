# Base-model benchmark harness (P3-05)

> [!WARNING]
> **SYNTHETIC/FIXTURE STAND-IN — NOT A PRODUCTION BENCHMARK.** The built-in
> candidates replay or deterministically perturb committed synthetic fixtures.
> No Whisper, Parakeet, Qwen3-ASR or other real model is imported, decoded,
> downloaded or promoted here. No GPU, network or heavy dependency is needed.
> No public leaderboard number is cited. `e2e_booking_accuracy` measured from
> audio is the primary metric; **a better WER alone never promotes**
> (`context.md` §2 invariant 4). This harness writes **no gate decision**.

One command compares two candidates on our own labels: the sealed v1 synthetic
golden stand-in (36 turns / 6 calls, content hash
`662cdaf43ee2d5fde03faf5e69fde196114d4f312406d94c00bf53714d03da8e`) and the
general fixture pool from `tests/fixtures/turns.jsonl`. The P1-07 evaluation
harness is reused **read-only** for fixture loading, per-turn metrics, critical
slices, paired bootstrap CIs and contract-valid per-candidate reports.

## Layout

| File | Responsibility |
|---|---|
| `runners.py` | `CandidateRunner` protocol, registry, deterministic `identity` / `degraded` fakes |
| `settings.py` | Typed `VOICELOOP_BENCH_*` settings |
| `compare.py` | Read-only golden/fixture loading, candidate scoring, `A - B` paired deltas |
| `report.py` | Internal comparison artifact builder, validation and writer |
| `cli.py` | One-command CLI and structured errors |

## Public API

```python
from services.evalgate.benchmarks import (
    BenchmarkSettings,
    CandidateRunner,
    compare_candidates,
    get_candidate,
    register_candidate,
)
```

- `CandidateRunner` — protocol with `name`, `artifact_uri` and
  `predict(row, event) -> services.evalgate.metrics.Prediction`.
- `register_candidate(runner)` / `get_candidate(name)` / `available_candidates()`
  — pluggable runner registry; built-in names are reserved.
- `load_benchmark_inputs(settings)` — verifies and loads the sealed stand-in and
  fixture pool without writing to them.
- `compare_candidates(settings) -> ComparisonData` — scores two candidates on
  both sets with paired bootstrap intervals.
- `build_comparison_artifact(data)` / `validate_comparison_artifact(artifact)` /
  `write_comparison_artifact(artifact, path)` — internal artifact lifecycle.
- `services.evalgate.benchmarks.cli.main(argv)` — CLI entry point.

### Real-adapter seam (documentation only)

A future Whisper variant, Parakeet/NeMo model, Qwen3-ASR model or other STT
runtime can implement `CandidateRunner` against its **converted/quantised
artifact**, read `row["audio_path"]` at its native sample rate (including 8 kHz),
use the exact production decoding settings and call `register_candidate(...)`.
That adapter belongs in its own task/runtime with optional heavy dependencies;
this package neither imports nor installs it. Until then every `artifact_uri`
is a `fake://benchmark/...` stand-in (`context.md` §2 invariant 3).

## Run

From the repository root (`.venv\Scripts\python.exe` is this environment's
`python`):

```powershell
.\.venv\Scripts\python.exe -m services.evalgate.benchmarks.cli --candidates identity,degraded --output reports/evalgate/benchmarks/comparison.json
```

Optional flags: `--fixtures`, `--golden-artifact-dir`, `--golden-version`,
`--level`, `--iterations`, `--seed`. Candidate order defines the delta sign:
the first name is candidate A, the second is candidate B. Exactly two distinct
names are required. Built-ins:

| Name | Behaviour |
|---|---|
| `identity` | Replays each fixture event's recorded STT text and parser slots via the P1-07 fake |
| `degraded` | Deterministically appends a hallucinated `zero` token and removes the `guests` slot |

The CLI writes one JSON artifact, prints the primary metric rule and the
synthetic limitation, and exits 2 with `{"code","message"}` on stderr for a
known failure. Example stdout:

```text
Wrote comparison artifact: ...\comparison.json
Candidates compared (A - B): identity - degraded
Primary metric: e2e_booking_accuracy (end-to-end booking-field accuracy from audio)
SYNTHETIC/FIXTURE STAND-IN: not a production benchmark; no public leaderboards used.
WER alone never justifies promotion (context.md section 2 invariant 4).
```

## Settings

Every field reads `VOICELOOP_BENCH_*` (pydantic-settings, `.env` supported):

| Variable | Default | Meaning |
|---|---|---|
| `VOICELOOP_BENCH_FIXTURES_PATH` | `tests/fixtures/turns.jsonl` | Read-only labelled pool |
| `VOICELOOP_BENCH_GOLDEN_ARTIFACT_DIR` | `services/datasets/golden/artifacts` | Read-only sealed stand-in directory |
| `VOICELOOP_BENCH_GOLDEN_VERSION` | `golden.v1` | Sealed version to verify |
| `VOICELOOP_BENCH_EXPECTED_GOLDEN_CONTENT_HASH` | committed v1 hash | Refuse a changed seal |
| `VOICELOOP_BENCH_OUTPUT_PATH` | `reports/evalgate/benchmarks/comparison.json` | Only written path |
| `VOICELOOP_BENCH_CANDIDATES_CSV` | `identity,degraded` | Candidate pair A,B |
| `VOICELOOP_BENCH_BOOTSTRAP_LEVEL` | `0.95` | CI level |
| `VOICELOOP_BENCH_BOOTSTRAP_ITERATIONS` | `1000` | Paired resamples |
| `VOICELOOP_BENCH_BOOTSTRAP_SEED` | `42` | Local RNG seed (deterministic) |
| `VOICELOOP_BENCH_HOTWORDS`, `_INITIAL_PROMPT`, `_CONDITION_ON_PREVIOUS_TEXT`, `_LANGUAGE` | empty / `false` / `en` | Placeholder decoding settings recorded in reports |
| `VOICELOOP_BENCH_ENTITY_PATTERN_OVERRIDES`, `_OTHER_LANGUAGE_PATTERN` | P1-07 defaults | Entity/slice extraction |

## Artifact contract

The outer artifact is an **internal shape**
(`artifact_type: "voiceloop.benchmark.comparison"`,
`comparison_schema_version: "internal-1.0.0"`). It does **not** pretend to be
`schemas/eval_report.schema.json` 1.0.0; no schema or contract file was changed.
It contains:

- `provenance` — fixture SHA-256, verified golden hash/flags, 36 turns / 6
  calls, and explicit `false` flags for training/tuning use, public
  leaderboards and real adapters.
- `sets.golden_standin` and `sets.general_capability` — both synthetic. Each has
  `turns`, `calls`, `metrics`, `slice_metrics`, `deltas` and two embedded
  `candidate_reports`.
- `candidate_reports` — each is a full `eval_report` 1.0.0 payload built by
  P1-07 and validated with `voiceloop_common.validate(report, "eval_report")`.
  Its `confidence_intervals` bound **candidate − identity** (the internal
  reference fake), not absolute values; `eval_sets` stays
  `["general_capability"]` so a synthetic run is never mislabelled as the
  sealed set. `gate_decision.outcome` is `not_evaluated`.
- `deltas` — for all five core metrics: `point`, `lower`, `upper`, `level`.
  These bound **candidate A − candidate B** with the same paired-bootstrap
  semantics as P1-07 (shared turn indices, nearest-rank percentiles, seeded
  local RNG). Negative WER/entity-error deltas favor A; positive slot-F1,
  schema-valid-rate and E2E deltas favor A.
- `metric_semantics.primary_metric` — `e2e_booking_accuracy`, defined as
  end-to-end booking-field accuracy from audio. Promotion is
  `not_evaluated`; no threshold or gate rule is applied.

`validate_comparison_artifact` checks the internal fields and validates every
embedded report against the committed schema before writing.

## Metrics and statistics

Metric definitions and slice rules are inherited unchanged from P1-07: WER,
entity error (names/dates/numbers), parser slot F1, schema-valid rate and the
all-or-nothing primary E2E booking-field accuracy. Critical slices are `names`,
`dates`, `numbers`, `silence`, `other_languages`. Turn counts may overlap and
`other_languages` has **0 synthetic turns**: its neutral values are no evidence,
not a pass.

## Golden-set handling

`load_benchmark_inputs` calls the committed P1-06 `verify_seal` **without** audit
context, so no audit entry is appended. It checks the seal, the committed
content hash, synthetic/workflow-stand-in flags, the simulated review mode and
1:1 `(call_id, turn_index)` alignment before using labels. Nothing writes to
`services/datasets/golden/`; the set is not trained on or tuned on
(`context.md` §2 invariant 5). The golden stand-in and the general pool are
derived from the same 36 synthetic turns, so their numbers are expected to
match; they are not two independent evidence sources. No human listened to the
tone/noise/silence WAVs, so §2 invariant 2 is **not** satisfied.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/evalgate_benchmarks -q
.\.venv\Scripts\python.exe -m ruff check services/evalgate/benchmarks tests/unit/evalgate_benchmarks
.\.venv\Scripts\python.exe -m ruff format --check services/evalgate/benchmarks tests/unit/evalgate_benchmarks
.\.venv\Scripts\python.exe -m mypy services/evalgate/benchmarks tests/unit/evalgate_benchmarks
```

Tests cover the runner protocol and registry, settings/env overrides, seal
verification with unchanged golden file hashes, tamper rejection on a temp
copy, delta direction, embedded report schema validation, CLI round-trip and
structured errors.

## Known gaps

- All results are fixture stand-ins. A real candidate adapter and a real,
  human-reviewed sealed golden set are required before any promotion use.
- The `degraded` fake is a deterministic error injector on five metrics, not a
  model of a specific base-model failure profile.
- The synthetic pool is English-only; no multilingual evidence exists.
- No shadow evidence and no promotion threshold change are made here
  (P3-04/P3-08 remain follow-ups).
