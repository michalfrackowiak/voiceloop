# P3-07 shadow mode (backend)

> [!IMPORTANT]
> **FAKE-ONLY REFERENCE IMPLEMENTATION — NO LIVE TRAFFIC, NETWORK, GPU OR REAL
> SERVING STACK.** Every injected protocol has a deterministic in-memory fake.
> No Whisper/CTranslate2/ONNX/vLLM/MLflow/Postgres/S3 client is imported. The
> CLI reads the committed synthetic fixture manifest read-only and writes one
> internal comparison artifact. All results are **SYNTHETIC/FIXTURE STAND-IN**
> evidence for P3-08, never a promotion decision.

Shadow mode mirrors a **reference** to live turn audio into a bounded queue and
runs a candidate and the champion silently on a background worker. It never
feeds a result back to a call, never flips an alias and never changes a gate or
threshold (`context.md` §2 invariants 1, 4 and 10; §14). The call path sees only
a bounded enqueue, a deadline check and a receipt.

## Layout

| File | Responsibility |
|---|---|
| `settings.py` | Typed `VOICELOOP_RELEASE_SHADOW_` settings |
| `mirror.py` | Call-path protocols (`Clock`, `TurnSource`, `MirrorSink`, `AuditSink`), bounded fail-open `ShadowMirror`, clock/queue/source/audit fakes |
| `runner.py` | Background `CandidateRunner` / `ChampionRunner` protocols, `ShadowRunner.drain_once()`, digest-only `ModelResult`, scripted runner fakes |
| `scoring.py` | `OutcomeSource` protocol, digest projection, `call_id`-only join, outcome-based deltas and sample counts |
| `report.py` | Internal comparison artifact builder, strict validator, `ReportSink`, file/collecting sinks |
| `cli.py` | One-command fake-only rehearsal over `tests/fixtures/turns.jsonl` |

## Run and test

From the repository root (this environment uses `.venv\Scripts\python.exe`):

```powershell
.\.venv\Scripts\python.exe -m services.release.shadow.cli --output reports/release/shadow/comparison.json
.\.venv\Scripts\python.exe -m pytest tests/unit/release_shadow -q
.\.venv\Scripts\python.exe -m ruff check services/release/shadow tests/unit/release_shadow
.\.venv\Scripts\python.exe -m ruff format --check services/release/shadow tests/unit/release_shadow
.\.venv\Scripts\python.exe -m mypy services/release/shadow tests/unit/release_shadow
```

CLI flags: `--fixtures` (read-only labelled JSONL), `--output`, `--max-pending`,
`--budget-ms`. Successful output includes turn/call counts, candidate-vs-champion
`outcome_call_accuracy` delta, the primary-metric rule and two explicit
limitations lines. Known errors exit 2 with `{"code","message"}` on stderr.

The CLI fixture rehearsal yields 36 turns mirrored, 0 drops, 6 calls scored,
candidate accuracy `1.0000`, champion `0.5000`, delta `+0.5000`; the champion
fake deterministically drops `guests` on even-numbered synthetic calls. This is
a plumbing demonstration, not evidence about any real model.

## Public API

```python
from services.release.shadow import (
    BoundedMirrorSink,
    CollectingAuditSink,
    FakeClock,
    ShadowMirror,
    ShadowRunner,
    ShadowSettings,
    ShadowTurn,
    score_comparison,
)

settings = ShadowSettings()  # VOICELOOP_RELEASE_SHADOW_*
clock = FakeClock()  # deterministic; advance(), never sleeps
audit = CollectingAuditSink()  # labelled in-memory fake
sink = BoundedMirrorSink(clock, settings.max_pending)
mirror = ShadowMirror(settings, sink, clock, audit)
receipt = mirror.mirror(turn)  # the ONLY call-path entry point
# candidate / champion / runner / outcome source injected in the background:
# runner.drain_once(); score_comparison(...); build_shadow_report(...)
```

`ShadowTurn` (and the internal `MirroredTurn`) carry only `call_id`,
`turn_index`, `audio_uri`, a synthetic flag and timestamps — **no audio bytes,
transcripts, names, emails, phone numbers, credentials or card digits**.
`MirrorReceipt` exposes `accepted`, `reason` (`queued`/`full`/
`deadline_exceeded`/`rejected`/`sink_error`), deadline and elapsed clock time.
`MirrorStats` exposes observed/accepted/dropped/sink-error counters and the
maximum observed elapsed time. `ModelResult` stores SHA-256 digests of canonical
booking-field values, never the values themselves.

`score_comparison(...)` returns `ComparisonData` with three outcome metrics per
side and candidate-minus-champion deltas. `build_shadow_report(data, uri)` and
`write_shadow_report(report, path)` produce the internal artifact;
`validate_shadow_report` rejects promotion claims, malformed counts and any
field whose name could carry raw guest data.

## Settings (`VOICELOOP_RELEASE_SHADOW_`)

| Variable | Default | Meaning |
|---|---|---|
| `VOICELOOP_RELEASE_SHADOW_TASK_ID` | `P3-07` | Task ID stamped on audit events and reports |
| `VOICELOOP_RELEASE_SHADOW_MAX_PENDING` | `64` | Bounded queue capacity (`>= 1`) |
| `VOICELOOP_RELEASE_SHADOW_MIRROR_BUDGET_MS` | `2.0` | Enqueue deadline (`> 0`) |
| `VOICELOOP_RELEASE_SHADOW_MAX_TURNS_PER_DRAIN` | `128` | Background batch bound (`>= 1`) |
| `VOICELOOP_RELEASE_SHADOW_SCORABLE_STATUSES_CSV` | `confirmed,edited_by_staff` | Call statuses eligible for scoring |
| `VOICELOOP_RELEASE_SHADOW_IGNORED_OUTCOME_FIELDS_CSV` | `call_id,status,updated_at` | Final-record keys excluded from comparison |
| `VOICELOOP_RELEASE_SHADOW_FIXTURES_PATH` | `tests/fixtures/turns.jsonl` | Read-only CLI input |
| `VOICELOOP_RELEASE_SHADOW_OUTPUT_PATH` | `reports/release/shadow/comparison.json` | Only path this package writes |
| `VOICELOOP_RELEASE_SHADOW_REPORT_STATUS_LABEL` | `SYNTHETIC/FIXTURE STAND-IN - NOT PRODUCTION SHADOW EVIDENCE` | Artifact label |
| `VOICELOOP_RELEASE_SHADOW_LOG_LEVEL` | `INFO` | Startup logging hint |

No secrets, base-model names, promotion thresholds or serving addresses belong
here. Real adapters take their credentials from their own deployment config.

## Contract and call-path isolation

1. A caller constructs an immutable `ShadowTurn` and calls
   `ShadowMirror.mirror()`. The mirror copies its safe fields, checks
   `Clock.now() + budget`, performs **one O(1) in-memory handoff**, returns a
   receipt and never calls a model runner. A full queue, elapsed deadline, sink
   exception or audit exception yields a dropped receipt with a stable reason;
   nothing is raised into the caller and the turn is never mutated.
2. `ShadowRunner.drain_once()` is the only model-work entry point. It runs on a
   background worker, not in the call path, processes at most
   `max_turns_per_drain` queued turns and converts every candidate/champion
   exception into a failed result. A failing candidate cannot prevent the
   champion result from being recorded.
3. **Verified bound (fake clock, no real sleeps):** an injected slow/failing
   candidate that advances the fake clock by 30 s per call leaves the clock
   unchanged, both runner call lists empty and the receipt inside the 2 ms
   budget at `mirror()` return; only the explicit `drain_once()` advances time
   and records results. `tests/unit/release_shadow/test_runner.py` asserts this.
4. Every `shadow.result` audit event carries source (`candidate`/`champion`),
   artifact URI, model ID, completion timestamp and the synthetic flag. Only
   allowlisted fields reach `AuditSink`; exception *bodies* are never logged,
   only exception type names.
5. `OutcomeSource.for_call(call_id)` is the sole join. No transcript, audio,
   turn index, event ID or guest name is used to match an outcome. Both sides'
   final successful result per call is compared with the call's final booking
   record as canonical-value SHA-256 digests. Metrics:
   `outcome_call_accuracy` (primary; all fields match), `outcome_error_rate`
   (`1 - accuracy`, or `0.0` with no scorable calls), and
   `outcome_field_accuracy` (exact matches over the union of field names).
   Deltas are candidate minus champion. Sample counts include observed/mirrored
   turns, drops, per-side results, outcome coverage and exclusion reasons.
6. The internal artifact (`artifact_type: voiceloop.shadow.comparison`,
   `comparison_schema_version: internal-1.0.0`) is **not** an
   `eval_report` 1.0.0 claim. It carries
   `promotion_decision: not_evaluated`, `alias_writes: 0`, `champion_flips: 0`,
   `gate_edits: 0`, `thresholds_changed: false` and a `shadow_evidence` block
   with `report_uri`, `turns_observed` and `outcome_error_rate` for P3-08 to
   consume structurally. No gate code, registry or alias store is imported.

## Real adapter seams (documentation only, not implemented here)

- `TurnSource.next_turn()` → the ingest/SDK turn queue, carrying only an audio
  URI and pseudonymous call/turn IDs. The SDK and ingest paths are owned by
  other tasks; this package does not wire itself into them.
- `MirrorSink.try_enqueue()` → an O(1) non-blocking in-process bounded queue or
  a producer that hands the reference to a background worker. A real adapter
  must never do model inference, network I/O or waiting on the call path; keep
  the configured deadline and drop counter, and expose `dequeue()`/`pending()`
  to the worker.
- `CandidateRunner` / `ChampionRunner` → resolve the audio URI at its native
  sample rate (including 8 kHz), run the exact converted/quantised artifact
  with production decoding settings on a non-serving worker, and return only
  `ModelResult` digests plus allowlisted metadata (`context.md` §2 #3, §10).
- `OutcomeSource.for_call()` → the P1-05 booking connector's outcome store;
  outcomes arrive late and are keyed by `call_id` alone.
- `AuditSink.record()` → structlog/OpenTelemetry with a bounded, non-blocking
  writer. `StructlogAuditSink` is provided; production wiring must preserve the
  allowlist and must not perform synchronous remote I/O in the call path.
- `ReportSink.write()` → object store or local path for P3-08; this package
  only offers a local `FileReportSink` and an in-memory collecting fake.
- `Clock.now()` → `SystemClock` delegates to the shared fail-open UTC clock.

## Known gaps

- All inputs and artifacts are synthetic/fixture stand-ins. No human listened
  to the tone/noise/silence WAVs, no real artifact served traffic, and the
  fixture outcomes are labels, not a booking-system response. §2 invariants 2
  and 3 are **not** satisfied by this rehearsal.
- The CLI's champion degradation is a deterministic two-field error injector,
  not a model of a real regression. Ticket-level fake delays use an injected
  clock; they prove scheduling isolation, not wall-clock latency under load.
- No scheduler, thread, SDK hook or ingest wiring ships here; callers own the
  background loop that calls `drain_once()` and the fail-open call-path adapter
  that calls `mirror()`.
- The report has point deltas and sample counts, not paired-bootstrap CIs. Any
  promotion use (P3-08) needs real shadow traffic, CIs and §14 human sign-off
  on thresholds; this package supplies neither.
- `models_invoked_by_call_path: false` is a structural property of this module,
  verified by injected slow/failing runner tests. A real adapter must preserve
  it; integration wiring is outside this task's ownership.
- Conflicts with `context.md`: none found. §9's release path puts shadow before
  canary and alias flip; this package stops at shadow evidence and touches no
  alias, threshold or gate. §14 escalation items were left untouched.
