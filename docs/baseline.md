# Baseline measurement (P1-08) — **SYNTHETIC STAND-IN / NOT A PRODUCTION BASELINE**

> [!WARNING]
> **This is a SYNTHETIC STAND-IN baseline, NOT a production baseline.** No real
> deployed STT or parser artifact was available in this environment (no GPU,
> network or heavy ML runtime). The P1-07 harness was run with its deterministic
> `identity` fake on the **synthetic fixture set** and on the sealed v1
> **synthetic golden stand-in** derived from that same pool. The numbers below
> describe fixture replay behaviour only. They are **not** measurements of a
> live deployment, and they must **not** be used for a real promotion decision.
> Re-running the harness against the real converted/quantised artifacts that
> will serve traffic is a **required follow-up before CP-1 can be called
> production-ready** (`context.md` §2 invariant 3).

## Status

- **Provisional champion reference for the synthetic loop rehearsal only.**
- Not a production champion reference; `gate_decision.outcome` is
  `not_evaluated`. No promotion or rollback was performed.
- No legal/compliance conclusion is drawn here, and no numeric retention
  decision is made (`context.md` §12; `docs/compliance.md` is a questionnaire
  for counsel only).
- No promotion threshold or gate rule was changed (`context.md` §14 escalation
  item).

## Provenance and inputs

| Item | Value |
|---|---|
| Harness | `services.evalgate.harness` (P1-07, read-only) |
| Champion / challenger fake | `identity` / `identity` (self-comparison) |
| `artifact_uri` in report | `fake://evalgate/identity` — a stand-in URI, **not** a converted artifact |
| Fixtures | `tests/fixtures/turns.jsonl` (read-only), 36 synthetic turns / 6 calls |
| Fixture SHA-256 | `aca23a303954d035245a97f78319df3694d63611b717a2ff2131bc2e0a4453fd` |
| Sealed golden stand-in | `services/datasets/golden/artifacts/golden.v1.*`, dataset `golden.v1.synthetic` |
| Golden stand-in content hash (from `docs/handoffs/P1-06.md`) | `662cdaf43ee2d5fde03faf5e69fde196114d4f312406d94c00bf53714d03da8e` |
| Stand-in properties | `synthetic: true`, `workflow_stand_in: true`, `review_mode: simulated_synthetic_labels`; 36 labels, 6 calls, 100% simulated second-review coverage |
| Source-pool check | All 36 fixture `(call_id, turn_index)` pairs match all 36 sealed stand-in labels; seal verification passed (`{"dir":"services/datasets/golden/artifacts","errors":[],"ok":true,"version":"golden.v1"}`) |

The sealed set was **not trained on and not used for tuning**. That rule holds
for the real sealed golden set (`context.md` §2 invariant 5) and for this
synthetic stand-in. No human listened to these tone/noise/silence WAVs; labels
are simulated from fixture expectations, so this artifact does **not** satisfy
the human-listening requirement for a real golden set (`context.md` §2
invariant 2). The harness's `eval_sets` field remains `general_capability`:
the P1-07 CLI labels `golden`/`regression` reports without filtering rows
(`docs/LEAD_BRIEFS.md`), so labelling this run `golden` would misrepresent it as
the sealed set. The run scores the fixture pool that the sealed v1 stand-in was
derived from; it is not a filtered set-membership evaluation.

## Exact command, seed and report

From the repository root (`.venv\Scripts\python.exe` is the environment's
`python`):

```powershell
.\.venv\Scripts\python.exe -m services.evalgate.harness --output docs/baseline/synthetic-standin-report.json --iterations 200 --seed 42
```

- Bootstrap: `iterations = 200`, `seed = 42`, `level = 0.95` (default).
- Report JSON: `docs/baseline/synthetic-standin-report.json`
  (`schema_version: 1.0.0`, `generated_at: 2026-11-24T23:51:39Z`).
- Decoding settings recorded: `hotwords: []`, `initial_prompt: ""`,
  `condition_on_previous_text: false`, `language: "en"`. These are fake-artifact
  placeholders, not verified production decoding settings.
- Harness stdout:

```text
Wrote report: C:\r\model-training-api\docs\baseline\synthetic-standin-report.json
Primary metric (e2e_booking_accuracy from audio): 0.833
WER alone never justifies promotion (context.md section 2 invariant 4).
```

## Baseline metrics (challenger = `identity`)

| Metric | Value | Direction |
|---|---|---|
| WER | 0.2236 (22.36%) | lower is better |
| Entity error (names/dates/numbers aggregate) | 0.0556 | lower is better |
| Parser slot F1 | 0.9518 | higher is better |
| Schema-valid rate | 0.9444 | higher is better |
| **E2E booking accuracy from audio (PRIMARY)** | **0.8333 (30/36 turns)** | **higher is better** |

Diagnostics in the report: entity errors by type are `names = 0.0278`,
`dates = 0.0`, `numbers = 0.0278`; per-field E2E accuracy is
`check_in = 0.9615`, `check_out = 0.8846`, `guests = 0.8846`,
`guest_name = 0.9615`, `room_type = 1.0`, `contact = 1.0`.

**`e2e_booking_accuracy` measured end to end from audio is the PRIMARY
promotion metric (`context.md` §2 invariant 4 and §9). WER alone never
promotes.** This baseline's silence slice illustrates why: WER is 2.0 on two
silence turns (inserted text on empty references, permitted by the schema), yet
E2E booking accuracy is 1.0 because no booking fields were emitted. Promotion
requires a required margin on the primary metric plus no critical-slice
regression, evaluated on the deployed artifact; that rule is implemented later
in P2-09, not here.

## Paired-bootstrap confidence intervals — **challenger − champion deltas**

`confidence_intervals.*.lower/upper` bound the **delta between challenger and
champion**, not the absolute metric value (`docs/LEAD_BRIEFS.md`, P1-07 review
follow-up; `docs/contracts.md`). Negative WER/entity-error deltas are
improvements; positive slot-F1, schema-valid-rate and E2E deltas are
improvements. Gate code must not read these bounds as absolute intervals.

| Metric | lower | upper | level |
|---|---:|---:|---:|
| WER | 0.0 | 0.0 | 0.95 |
| Entity error | 0.0 | 0.0 | 0.95 |
| Slot F1 | 0.0 | 0.0 | 0.95 |
| Schema-valid rate | 0.0 | 0.0 | 0.95 |
| E2E booking accuracy | 0.0 | 0.0 | 0.95 |

All bounds are exactly zero **by construction**: this baseline run compares the
`identity` fake against itself, so every per-turn paired difference is zero.
These are valid delta intervals for the self-comparison; they are **not**
evidence of model quality or of a promotion margin. A future challenger run
must resample shared turn indices and report its own deltas; an absolute
interval would require a separate field and a schema version bump.

## Critical-slice breakdown (challenger = `identity`)

Slices are derived from the reference transcript and `audio_kind`; a turn may
belong to more than one slice, so turn counts do not sum to 36.

| Slice | Turns | WER | Entity error | Slot F1 | Schema-valid rate | E2E accuracy |
|---|---:|---:|---:|---:|---:|---:|
| names | 8 | 0.1000 | 0.1250 | 0.9792 | 1.0000 | 0.8750 |
| dates | 8 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| numbers | 15 | 0.0167 | 0.0667 | 0.9497 | 0.9333 | 0.7333 |
| silence | 2 | 2.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| other_languages | **0** | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |

`other_languages` has **zero evidence**: its neutral values are placeholders,
not a passing slice result. P2-09 must gate on `turns > 0` before treating any
slice as "no regression" (`docs/LEAD_BRIEFS.md`). The synthetic pool is
English-only; a real multilingual set is needed.

## Verification

- Harness command and stdout are recorded above; report written to
  `docs/baseline/synthetic-standin-report.json`.
- Contract validation with
  `jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())`
  against `schemas/eval_report.schema.json`:
  `schema_version: 1.0.0`, `validation_errors: 0`, `RESULT: VALID` (exact output
  in `docs/handoffs/P1-08.md`).
- Golden stand-in seal verification passed (exact output above). No golden,
  schema, fixture, service or roadmap file was modified by this task.

## Required follow-up before CP-1 is production-ready

1. Obtain a real, human-reviewed, sealed golden set (`context.md` §2 invariants
   2 and 5); the P1-06 stand-in is not sufficient (see its handoff).
2. Re-run the harness on the **actual converted/quantised STT + parser
   artifacts** with their production decoding settings (§2 invariant 3), first
   against synthetic fixtures and then against the real sealed set.
3. Replace this provisional champion reference with that report; have the lead
   decide whether CP-1 is green. Do not promote on this document's numbers.
4. Later phases add gate logic (P2-09), conversion/artifact tests (P3-04),
   shadow evidence (P3-08) and real multilingual/silence slice sets.

## Conflicts with `context.md`

None when this document is read as a synthetic rehearsal artifact. Treating it
as a production baseline would violate §2 invariants 2 (a human listens to
audio), 3 (evaluate the deployed artifact) and 5 (golden set sealed and
human-reviewed), and would require escalation to the lead before any gate or
promotion use.
