# Evaluation harness v0 (P1-07)

> [!IMPORTANT]
> **`e2e_booking_accuracy` measured from audio is the PRIMARY promotion metric**
> (`context.md` §2 invariant 4 and §9). **A better WER never justifies promotion.**
> This harness writes `gate_decision.outcome = "not_evaluated"`: it reports numbers,
> it does not promote anything. Gate logic is P2-09 and may only use the primary
> metric plus critical-slice checks.

Pure-Python (3.11+) evaluation of the **converted artifact that will actually
serve** (§2 invariant 3). This v0 runs settings-selectable in-memory fakes over
the synthetic fixtures; there are no ML, network or GPU dependencies.

## Layout

| File | Responsibility |
|---|---|
| `metrics.py` | WER, entity error, slot F1, schema-valid rate, E2E booking accuracy, per-turn container |
| `bootstrap.py` | Deterministic paired bootstrap CIs on `challenger - champion` |
| `harness.py` | Fixture loader, fake artifacts, slices, `evaluate(...)`, CLI |
| `report.py` | `report.json` builder/writer validated against `eval_report` 1.0.0 |
| `settings.py` | Typed `VOICELOOP_EVALGATE_*` settings |

No `gate/`, `artifact/` or `benchmarks/` subpackages live here: P2-09, P3-04 and
P3-05 own those paths.

## Run

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m services.evalgate.harness
.\.venv\Scripts\python.exe -m services.evalgate.harness --champion identity --challenger oracle --seed 7 --iterations 2000 --output reports/evalgate/oracle-report.json
```

Inputs are always read-only: the default fixture manifest is
`tests/fixtures/turns.jsonl`, with each label pointing at a schema-valid event
and a synthetic WAV. The default output is `reports/evalgate/report.json`.
Settings are prefixed `VOICELOOP_EVALGATE_` (e.g.
`VOICELOOP_EVALGATE_BOOTSTRAP_ITERATIONS=2000`); CLI flags override them.

Fake artifacts (`--champion` / `--challenger`):

| Kind | STT transcript | Parser slots |
|---|---|---|
| `identity` (default) | Replays the recorded event's `stt.text` | Replays the recorded event's `parser.output`, including its known errors |
| `oracle` | `expected_transcript` from the label | `expected_slots` from the label (an upper bound, not a real model) |
| `no_slots` | `""` | `{}` |

`artifact_uri` defaults to `fake://evalgate/identity`; it is a stand-in and must
be replaced with the real converted-artifact URI before any gate decision.

## Metric definitions

- **WER** — Levenshtein alignment over case-folded `\\w+` tokens:
  `(S + D + I) / max(reference_tokens, 1)`. Empty reference with hallucinated
  text yields a value > 0 (4 tokens on silence → 4.0), as the contract allows.
- **Entity error** — deterministic regex multisets for `names`, `dates` and
  `numbers`. Missing and extra entities are errors; per-type error is
  `(missing + extra) / max(ref, hyp, 1)` capped at 1, and the aggregate is
  occurrence-weighted. Patterns come from `EntityPatterns` and can be replaced
  per type with `VOICELOOP_EVALGATE_ENTITY_PATTERN_OVERRIDES` (JSON object).
- **Slot F1** — exact-match micro F1 over `(field, canonical JSON value)` pairs
  from gold and predicted slot dicts; two empty dicts score 1.0.
- **Schema-valid rate** — fraction of parser outputs passing
  `schemas.booking.validate_booking`. On turns labelled with no booking fields,
  an empty output is exempt (documented no-slot rule); otherwise `{}` is invalid.
- **E2E booking accuracy (primary)** — predicted slots are the artifact's parser
  output derived from its STT transcript of the turn's audio; gold is the
  fixture label's `expected_slots`. A field is correct only if present on both
  sides with an exactly equal canonical value; a turn is correct only if **all**
  fields in the union of gold and predicted keys match. No-booking turns must
  be empty on both sides. The report also carries `e2e_field_*` breakdowns and
  `entity_error_{names,dates,numbers}` diagnostics.

## Statistics

All five core metrics get a paired-bootstrap CI. Resampling draws turn indices
with a local seedable `random.Random` and uses **the same indices for champion
and challenger**; bounds are the nearest-rank percentiles of
`challenger_metric - champion_metric`. Therefore:

- `confidence_intervals.<metric>.lower/upper` are **deltas**, not absolute values.
- Negative WER / entity-error deltas are improvements; positive slot F1,
  schema-valid-rate and E2E deltas are improvements.
- `level`, `iterations` and `seed` are configurable; identical settings and
  input order reproduce identical bounds, and `lower <= upper` is enforced.

`slice_metrics` is keyed by `names`, `dates`, `numbers`, `silence` and
`other_languages`; a turn may belong to several slices. Memberships come from
entities in the reference transcript, `audio_kind == "silence"`, and a
configurable `other_language_pattern`. The synthetic fixtures are English-only,
so `other_languages` has `turns: 0` and neutral no-evidence values (1.0 for
higher-is-better metrics, 0.0 for error metrics). The default `eval_sets` entry
is `general_capability`: **these synthetic fixtures are not the sealed golden
set** (§2 invariant 5) and a fixture report is never a baseline or a promotion
decision (P1-08 owns the baseline).

## Contract and verification

`report.json` is validated with
`voiceloop_common.validate(report, "eval_report")` before being written. It
conforms to `schemas/eval_report.schema.json` 1.0.0: version 1.0.0,
`artifact_uri`, `generated_at` (`voiceloop_common.utcnow()`), `eval_sets`, all
five required metrics, all five required CIs with `lower`/`upper`/`level`, the
five critical slices and `decoding_settings` (hotwords, initial prompt,
`condition_on_previous_text`, language). No contract file was changed.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy .
```

Unit tests live in `tests/unit/evalgate/` and include hand-computed metric and
bootstrap cases.

## Known gaps

- The WAV fixtures are tone/noise/silence carriers, not intelligible speech;
  the fakes replay labels/events rather than decoding audio. Real artifact
  runners arrive in P3-04, and gate/promotion logic in P2-09.
- The critical `other_languages` slice has no fixture turns; a real
  multilingual set is needed before slice regression checks are meaningful.
- No shadow evidence is emitted; P3-08 requires it before STT promotion.
