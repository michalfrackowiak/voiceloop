# Scheduled and drift-triggered retraining (P4-02)

Offline, deterministic drift evaluation for **replayed synthetic windows**. It
answers one question: are there enough newly verified examples **and** either a
descriptive drift signal or an audited manual override? A calendar tick is
recorded for audit and never starts training.

Every decision and report is labelled:

`SYNTHETIC/FIXTURE STAND-IN - NOT PRODUCTION DRIFT EVIDENCE`

There is no real scheduler, training, MLflow, network, GPU, or audio access in
this package. The downstream retraining step is an injected protocol whose
default fake only records the intended invocation.

## Layout

| File | Responsibility |
|---|---|
| `settings.py` | `VOICELOOP_DRIFT_*` typed settings, expected fixture/golden hashes, synthetic label. |
| `detectors.py` | Score-distribution and outcome-rate detectors, window protocols, report safety screen. |
| `policy.py` | `data volume AND (drift OR audited manual override)`; write-ahead audit seam. |
| `runner.py` | Fake/injected pipeline invocation, fixture replay adapter, in-process CLI. |
| `__init__.py` | Public API re-exports. |

Only `infra/pipelines/drift/` is owned by P4-02. Existing `infra/pipelines/*.py`
(P2-10), services, schemas, fixtures, and other docs are read-only here.

## Run

From the repository root (PowerShell):

```powershell
# One policy evaluation on read-only synthetic fixtures. A calendar tick alone
# will not trigger: default volume floor is 200 and fixture drift is stand-in only.
.\.venv\Scripts\python.exe -m infra.pipelines.drift.runner --scheduled-tick

# Replay with enough new data but no drift/manual override: still no pipeline call.
.\.venv\Scripts\python.exe -m infra.pipelines.drift.runner --new-verified 250 --min-new-verified 200

# Explicit, audited manual override (pseudonymous actor + coded reason required).
.\.venv\Scripts\python.exe -m infra.pipelines.drift.runner `
  --new-verified 250 --manual-actor operator-1a2b3c4d --manual-reason incident_replay

# Optionally write one sorted JSON report; no file is written by default.
.\.venv\Scripts\python.exe -m infra.pipelines.drift.runner `
  --new-verified 250 --manual-actor operator-1a2b3c4d --manual-reason incident_replay `
  --output reports/drift/decision.json
```

Output is `DECISION: {...}`, zero or more `AUDIT: {...}` lines, then
`REPORT: {...}`. Exit codes: `0` = decision recorded (triggered or skipped),
`2` = invalid configuration/input (for example `--min-new-verified 0` or a
manual actor without a reason). There is no exit-1 training path: the default
pipeline is a recorder. The `-m` invocation can emit a harmless
`RuntimeWarning` about importing the module after its package; output is
unaffected.

`crontab.example` in P2-10 remains the only example scheduler. A real cron,
Prefect or Dagster deployment should call this CLI (or `run_once`) on a tick,
with real data sources and a real injected downstream pipeline. The scheduler
is a **seam**, not an implementation: no polling loop, daemon or alerting is
added here.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/pipelines_drift -q
.\.venv\Scripts\python.exe -m ruff check infra/pipelines/drift tests/unit/pipelines_drift
.\.venv\Scripts\python.exe -m ruff format --check infra/pipelines/drift tests/unit/pipelines_drift
.\.venv\Scripts\python.exe -m mypy infra/pipelines/drift tests/unit/pipelines_drift
```

The done-criterion test injects replay windows whose current scores are shifted,
verifies that both detectors alarm, and asserts that the fake pipeline was
invoked exactly once. Another test checks that a calendar tick, even with
enough new data, never invokes it.

## Detectors (P4-02.1)

| Detector | Inputs | Reported deltas | Alarm when |
|---|---|---|---|
| `triage_score_distribution` | Baseline/current P2-01-style scores in `[0, 1]` | Baseline/current sample counts, mean, median, decile bucket deltas, total-variation distance, max bucket delta | Enough samples AND mean-delta or distribution-distance threshold crossed |
| `outcome_rates` | Baseline/current late-outcome statuses | Baseline/current counts, per-status rates and signed deltas, degraded/outcome rate deltas, max absolute status delta | Enough samples AND status-rate or degraded-rate threshold crossed |

Reports contain **counts and descriptive signed deltas only**. They explicitly
carry `causal_claim: "none"` and `promotion_decision: "not_evaluated"`. The
detectors do not infer why a distribution changed, do not consume fixture
failure labels, and do not touch P2-09 promotion thresholds (context.md §2.4,
§9, §14). Alarm thresholds are local, unapproved engineering placeholders for
`VOICELOOP_DRIFT_*`; they are not production acceptance criteria.

A window below `min_window_samples` still reports its deltas but sets
`enough_data: false`, `drifted: false` and reason
`insufficient_window_samples`. Absence of evidence is never reported as drift.
`turn_event` statuses are the only outcome values accepted; unknown strings
fail closed with `ValidationError`.

## Trigger policy (P4-02.2)

| Input | Effect |
|---|---|
| `new_verified_count >= min_new_verified` | Required data-volume leg |
| `drifted` from detectors | Sufficient second leg |
| `ManualOverride(actor_id, reason_code)` | Sufficient second leg, always audited |
| `scheduled_tick` / `last_run_at` | Audit fields only; never a reason |

`triggered = data_volume_ok AND (drifted OR manual_override_present)`.
Satisfied reasons use priority manual → drift → data volume; a skipped decision
records `blocked_reasons`. The policy is stricter than P2-10's OR policy; this
is a P4-02 scope decision, not a change to the P2-10 file. See "Conflicts and
scope notes" below.

Manual overrides use a pseudonymous actor token (`operator-1a2b3c4d`,
`reviewer-…`, `admin-…` or `lead-…`) and one of three coded reasons
(`incident_replay`, `review_backlog`, `lead_authorized`). Free text, names,
emails and phone-like actor IDs are rejected. `decide_trigger` writes the
actor, reason code and human-readable fixed reason description to the injected
`AuditSink` **before** the pipeline can be invoked, including when the volume
leg fails. The fake sink is `InMemoryAuditSink`; a real sink seam must preserve
this write-ahead behaviour.

## Public API

```python
from infra.pipelines.drift import (
    DriftSettings,
    InMemoryAuditSink,
    InMemoryOutcomeWindowSource,
    InMemoryScoreWindowSource,
    ManualOverride,
    OutcomeWindow,
    RecordingPipeline,
    TriageScoreWindow,
    build_fixture_replay_sources,
    detect_all,
    run_once,
)

settings = DriftSettings(min_new_verified=250, min_window_samples=10)
scores, outcomes = build_fixture_replay_sources(settings=settings)
detectors = detect_all(scores.load(), outcomes.load(), settings)
report = run_once(
    score_source=scores,
    outcome_source=outcomes,
    pipeline=RecordingPipeline(),  # inject the real downstream chain here
    new_verified_count=250,
    settings=settings,
    manual_override=ManualOverride("lead-a1b2c3d4", "lead_authorized"),
    audit_sink=InMemoryAuditSink(),
)
payload = report.to_dict()  # labelled, JSON-serialisable, PII-free
```

Protocols: `ScoreWindowSource.load()`, `OutcomeWindowSource.load()`,
`AuditSink.record()`, and `RetrainingPipeline.invoke(decision, evidence)`.
`RecordingPipeline` and the in-memory window/audit sources are the default
fakes. `build_fixture_replay_sources()` reads the read-only fixture manifest
and events, splits by **call ID** (context.md §2.7), and uses an injected
`TurnScorer`; the default adapter calls P2-01's pure-Python scoring engine and
retains only the numeric score. It hash-checks the fixture file and the sealed
golden stand-in manifest read-only before replay and refuses a mismatch.

Real seams for the lead: inject production window sources (triage score store
and late-outcome store), a durable audit sink, and a pipeline implementation
that calls the P2-10 chain (or its real replacement). Keep the calendar tick in
the scheduler; it remains audit-only here.

## Report safety and contracts

`assert_no_sensitive_text()` runs before any report, audit entry or pipeline
result is serialised. It rejects byte payloads, email-like strings, phone-like
digit groups, 13–19 digit card/number runs and audio file references. Names are
excluded structurally: sources supply only numbers, allowlisted statuses, and
coded actor/reason values — never raw event text. No audio bytes or raw scores
are embedded in reports.

Contracts touched: **none**. No JSON Schema, OpenAPI, `CHANGELOG.md`, golden
set, promotion threshold or P2-10 file was changed. Local internal shapes are
`DetectorSignal.to_dict()`, `DetectorReport.to_dict()`,
`TriggerDecision.to_dict()` and `RunnerReport.to_dict()`; all are internal
unless a lead-owned contract bump adopts them.

## Conflicts and scope notes

- No conflict with any `context.md` §2 invariant was found. §2.11 (cheapest fix
  first) is respected: a descriptive score shift alone never starts retraining,
  and this package makes no adaptation-tier or promotion decision.
- P2-10 `triggers.should_trigger()` uses OR semantics; P4-02.2 requires the
  stricter `volume AND (drift OR manual)` rule. Both policies coexist: P4-02
  owns only its own package and does not edit P2-10. The lead should decide
  whether to unify them at a later checkpoint.
- The roadmap's "no calendar-only retraining" rule lives in P2-10.2/P4-02.2,
  not in `context.md` §9 (the evaluation-gate section). This README follows the
  explicit prohibition without changing §9.
- Fixture outcomes and scores are synthetic stand-ins. The sealed golden
  stand-in is read-only and never used for detector tuning. No §14 escalation
  (legal, golden change, promotion threshold) was triggered.

## Known gaps

- Detector thresholds (`min_window_samples`, mean/TV distance and rate deltas)
  are engineering placeholders, not human-approved production alarm limits.
- Outcome windows on the 36 fixtures contain only six late outcomes, split
  3/3 by call; they are used for determinism/PII tests, not as rate evidence.
  The drift-trigger test uses explicitly injected shifted windows.
- The default scorer depends on P2-01's engine; real score-store adapters are
  a lead integration seam. No real scheduler or downstream chain is wired.
- `assert_no_sensitive_text()` is a structural/PII-pattern screen, not a
  general-purpose name detector; maintain the allowlisted inputs when adding
  new report fields.
