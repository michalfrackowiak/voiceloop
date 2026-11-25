# P4-01 canary with automatic rollback (backend)

> [!IMPORTANT]
> **FAKE-ONLY REFERENCE IMPLEMENTATION — NO LIVE TRAFFIC, NETWORK, GPU,
> REGISTRY OR MLflow CONNECTION.** All event/outcome sources, the clock, the
> alias store and the audit sinks are deterministic in-memory fakes. Every
> report and audit event is labelled
> **`SYNTHETIC/FIXTURE STAND-IN - NOT LIVE CANARY EVIDENCE`**. This package
> makes **no promotion decision**, claims no champion, changes no gate rule and
> touches no threshold (`context.md` §2 invariants 4/10, §9 and §14).

The canary observes a bounded, explicitly configured share of *calls* (never
utterances, §2 invariant 7). Each canary-served call's predicted booking fields
are paired with that call's final booking record through the injected P3-07
outcome seam. A same-call champion baseline from replay/shadow makes the
comparison paired; missing or malformed outcome evidence fails closed. If a
paired outcome-metric regression (or the window boundary with too little
evidence) is observed, a single one-step alias rollback restores the version
recorded in an injected P2-09 promotion record.

## Layout

| File | Responsibility |
|---|---|
| `settings.py` | Typed `VOICELOOP_CANARY_` settings with explicit fraction/window bounds |
| `split.py` | Deterministic, salted, call-level traffic split and arm audit fields |
| `metrics.py` | Paired, descriptive outcome metrics, sample counts and fail-closed verdict |
| `rollback.py` | Bounded, idempotent, hash-chained one-step alias rollback |
| `runner.py` | Background-only tick orchestration, audit events, fixture rehearsal CLI |

## Run and test

From the repository root (this environment uses `.venv\Scripts\python.exe`):

```powershell
.\.venv\Scripts\python.exe -m services.release.canary.runner
.\.venv\Scripts\python.exe -m pytest tests/unit/release_canary -q
.\.venv\Scripts\python.exe -m ruff check services/release/canary tests/unit/release_canary
.\.venv\Scripts\python.exe -m ruff format --check services/release/canary tests/unit/release_canary
.\.venv\Scripts\python.exe -m mypy services/release/canary tests/unit/release_canary
```

The one-command rehearsal reads `tests/fixtures/turns.jsonl` read-only, injects
a deterministic booking-field regression and prints one JSON report whose
`rollback.status` is `applied`, with `alias_writes: 1` and the synthetic label.
Exit code is 0 only when the bounded rollback was applied (or was already
applied), 2 otherwise. No fixture file is written.

## Public API

```python
from services.evalgate.gate.aliases import InMemoryAliasStore
from services.evalgate.gate.audit import PromotionRecord
from services.release.canary import (
    CanaryRunner,
    CanarySettings,
    CollectingCanaryAudit,
    FakeEventSource,
    InMemoryRollbackAudit,
    run_fixture_rehearsal,
)
from services.release.shadow.mirror import FakeClock
from services.release.shadow.scoring import FakeOutcomeSource

result = run_fixture_rehearsal()  # labelled, deterministic demonstration
assert result.rollback.status == "applied"
assert result.rollback.writes == 1
assert result.promotion_decision == "not_evaluated"
assert "SYNTHETIC" in result.evidence_label

# Explicit wiring uses the same injected seams:
settings = CanarySettings(canary_fraction=0.05, window_seconds=300.0)
promotion = PromotionRecord(
    actor_id="fake-gate-promoter",
    action="promote",
    reason="synthetic canary rehearsal",
    before_version="v1",
    after_version="v2",
    task_id="P4-01",
)
runner = CanaryRunner(
    settings=settings,
    promotion=promotion,
    event_source=FakeEventSource(()),
    outcome_source=FakeOutcomeSource({}),
    alias_store=InMemoryAliasStore({"@champion": "v2"}),
    audit=CollectingCanaryAudit(),
    clock=FakeClock(),
    rollback_audit=InMemoryRollbackAudit(),
)
tick = runner.tick()  # no regression yet -> no alias write
```

`CanaryTickResult.to_report()` emits an internal
`voiceloop.canary.rehearsal` artifact (schema `internal-1.0.0`). It is **not**
an `eval_report` 1.0.0 claim. `CanaryMetrics.to_dict()` and
`RegressionVerdict.to_dict()` are labelled fragments carrying rates, paired
deltas and integer counts only. `RollbackResult.to_dict()`/`to_event()` carry
the status, reason, alias, before/after versions, write count, exception *type*
only and the evidence label.

## Settings (`VOICELOOP_CANARY_`)

| Variable | Default | Meaning |
|---|---|---|
| `VOICELOOP_CANARY_TASK_ID` | `P4-01` | Task ID stamped on audit and reports |
| `VOICELOOP_CANARY_CANARY_FRACTION` | `0.05` | Canary share of calls, bounded to `[0, 0.20]` |
| `VOICELOOP_CANARY_SPLIT_SALT` | `voiceloop-canary-v1` | Non-secret salt for deterministic buckets |
| `VOICELOOP_CANARY_WINDOW_SECONDS` | `300.0` | Rollback window, bounded to `(0, 3600]` |
| `VOICELOOP_CANARY_MIN_PAIRED_CALLS` | `3` | Sample-count floor, not a promotion threshold |
| `VOICELOOP_CANARY_MAX_CALLS_PER_TICK` | `16` | Background batch bound (1–128) |
| `VOICELOOP_CANARY_MAX_PAIRS` | `256` | In-memory evidence bound (1–10000) |
| `VOICELOOP_CANARY_SCORABLE_STATUSES_CSV` | `confirmed,edited_by_staff` | Eligible final-record statuses |
| `VOICELOOP_CANARY_IGNORED_OUTCOME_FIELDS_CSV` | `call_id,status,updated_at` | Non-booking final-record keys |
| `VOICELOOP_CANARY_ROLLBACK_ACTOR_ID` | `canary-auto-rollback` | Pseudonymous rollback actor |
| `VOICELOOP_CANARY_FIXTURES_PATH` | `tests/fixtures/turns.jsonl` | Read-only rehearsal input |
| `VOICELOOP_CANARY_ROLLBACK_AUDIT_PATH` | `reports/release/canary/rollback_audit.jsonl` | Default file audit path |
| `VOICELOOP_CANARY_EVIDENCE_LABEL` | `SYNTHETIC/FIXTURE STAND-IN - NOT LIVE CANARY EVIDENCE` | Mandatory provenance label |
| `VOICELOOP_CANARY_LOG_LEVEL` | `INFO` | Startup logging hint |

No secrets, base-model names, serving addresses or promotion thresholds belong
here. `min_paired_calls` is a sample-count safety floor for descriptive
evidence; it is not a gate margin and is never used to promote.

## Contract and safety properties

1. **Split:** one pseudonymous `call_id` is hashed with the configured salt into
   a bucket; the arm is stable across ticks and is never chosen per utterance.
   Raw digit runs of 13–19 characters are rejected as unredacted identifiers.
2. **Metrics:** the primary metric is end-to-end booking-field accuracy from
   the final record (`e2e` call accuracy and field accuracy), compared per call
   against a same-call champion baseline. Deltas are descriptive means of
   paired per-call differences; wins/losses/ties and every exclusion are
   counted. No WER input or output exists.
3. **Verdict:** missing baseline/outcome, empty final records or identity
   mismatches mean `incomplete_outcome_evidence` and require rollback.
   Non-scorable statuses and too few pairs are `insufficient`; the runner still
   fails closed at the window boundary. A strictly negative paired delta on
   either outcome metric is `outcome_metric_regression`.
4. **Rollback:** one `set(@champion, previous_version)` from the injected P2-09
   promotion record, cross-checked against the P3-06 read-only registry
   protocol. It is refused outside the window, on an unbound/stale/mismatched
   alias and on read failures; repeated calls are idempotent with no extra
   write. A successful flip appends one hash-chained audit record; an audit
   failure restores the pre-flip alias and reports `audit_write_failed`.
5. **Promotion:** never performed here. Reports carry
   `promotion_decision: not_evaluated`, `champion_claim: false`,
   `gate_edits: 0` and `thresholds_changed: false`.
6. **PII:** only pseudonymous call IDs, SHA-256 digests of booking-field values,
   field names, rates, counts and exception type names are emitted. No audio
   bytes, transcripts, names, emails, phone numbers, card digits or exception
   bodies appear in logs or reports.

## Consumed contracts (read-only, no edits)

- P2-09 gate: `PromotionRecord` (`before_version` is the sole rollback target),
  `CHAMPION_ALIAS`, `InMemoryAliasStore` and the one-`set` alias-store seam.
  `eval_report` 1.0.0 is **not** produced or modified.
- P3-06 sync: `ReadOnlyAliasStore.get` is the poll-only consistency check; this
  package never calls `set` through that interface.
- P3-07 shadow: `Clock`/`FakeClock`, `OutcomeSource`/`FakeOutcomeSource`,
  `OutcomeRecord.from_payload` and `field_digests` are reused read-only.
- `tests/fixtures/turns.jsonl` is read-only; the sealed synthetic golden
  stand-in (`golden.v1`, content hash
  `662cdaf43ee2d5fde03faf5e69fde196114d4f312406d94c00bf53714d03da8e`) is never
  read for tuning or changed.

## Real adapter seams (documentation only, not implemented)

- `EventSource.next_call()` → a bounded background queue fed by ingest/SDK;
  carry only a pseudonymous `call_id`, safe artifact/baseline references and
  timestamps, never audio bytes or transcripts.
- `OutcomeSource.for_call()` → the P1-05 booking connector, joined by
  `call_id` alone; outcomes arrive late and must be complete before scoring.
- Champion baseline prediction → P3-07 shadow runner or a deterministic replay
  of the same call; without a same-call baseline the canary fails closed.
- Clock → P3-07 `SystemClock` (or an OTel-compatible UTC clock) in production;
  the injected fake clock is the only source of time in tests.
- Alias store / registry → MLflow `get/set_model_version_by_alias` for the
  champion alias, with write-ahead audit and cross-check against P3-06's
  polled snapshot. This package supplies only fakes.
- Audit → `RollbackAuditLog` is a local JSONL hash chain; production should use
  the gate's append-only audit with bounded, non-blocking writes.

## Known gaps / conflicts

- All seams are fakes. No real traffic, human review, converted artifact,
  MLflow alias, booking system or latency measurement is involved; §2
  invariants 2 and 3 are not satisfied by this rehearsal.
- Descriptive paired deltas have no bootstrap CIs; a real promotion judgment
  still needs P3-08 shadow evidence, a gate pass and §14 human sign-off.
- `RollbackAuditLog` writes locally and the compensating restore on an audit
  failure is not transactionally atomic with MLflow; hard-crash atomicity
  remains a lead-owned write-ahead-audit follow-up (P3-06 note).
- Calls with non-scorable statuses are excluded from evidence; if exclusions
  leave too few pairs, rollback happens only at the configured window boundary.
- No scheduler or service wiring ships here; callers own the background loop
  that calls `tick()`.
- Conflicts with `context.md`: none found. §9 places auto-rollback after shadow;
  this package consumes shadow/read-only gate seams and performs no promotion,
  so §2 invariants 1/3/4/7/9/10 remain intact.
