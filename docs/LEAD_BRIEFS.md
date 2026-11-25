# Lead-owned worker briefs for Phase 1 (pre-drafted)

These are briefs for the worker subagents to be spawned after CP-0. They are planning notes, not contracts; the roadmap and `context.md` remain authoritative.

## Common preamble for every Phase 1 worker

Work at `C:\r\model-training-api`. Read fully first: `context.md` (§2 invariants, §4 architecture, §5 event schema, §9 eval rules, §13 conventions, §14 agreement), `roadmap.md` (your task), `docs/conventions.md`, `docs/contracts.md`, `docs/schemas.md`. Use `.\.venv\Scripts\python.exe -m pytest|ruff|mypy`. Do not install deps, do not git commit, do not edit files outside your listed paths. Keep heavy deps optional with in-memory fakes. Use `voiceloop_common` (`BaseSettings`, `settings_config`, `configure_logging`, `errors.*`, `utcnow`, `new_id`, `validate(instance, schema_name)`). Write `docs/handoffs/<task-id>.md` and a service `README.md` with run/test/contract notes. Tests green + ruff + mypy clean before reporting done.

## Wave 1

### P1-01 Ingest API — backend — paths `services/ingest/`
Deps P0-01, P0-02. Subtasks: POST /v1/turns validating `turn_event.schema.json` via `voiceloop_common.validate(event, "turn_event")` and idempotent on event_id; audio upload path to object store (presigned or multipart) with an in-memory fake selected by settings; OTel-compatible endpoint mapping spans/attributes into turn events; PATCH /v1/calls/{call_id}/outcome merging late outcomes by call_id; Postgres migrations for calls/turns/outcomes/model versions (SQL files + a fake repository implementation, real driver optional). Do NOT create `services/ingest/consent/` — P1-09 owns it. Done when fixture events ingest end to end, invalid events return structured `{code,message,details}` errors, and a load test shows storage failure does not block callers. Use FastAPI TestClient (httpx installed) for tests, including fixtures from `tests/fixtures/`.

### P1-03 Python SDK — sdk — paths `sdk/` root only
Deps P0-02. Subtasks: `log_turn(...)` async non-blocking send with local retry queue; timeouts, circuit breaker, drop-on-failure that never raises into the call path; audio capture helper preserving original sample rate. Done when tests prove an unreachable server adds no more than a few ms to the caller and raises no exceptions. Do NOT create `sdk/pipecat/` or `sdk/connectors/` — P1-04/P1-05 own them. Provide in-memory transport fakes and deterministic timing assertions.

### P1-07 Evaluation harness v0 — ml-eval — paths `services/evalgate/` root only
Deps P0-04. Subtasks: WER and entity-level error (names/dates/numbers); parser slot F1 and schema-valid rate; end-to-end booking-field accuracy from audio (primary); paired bootstrap CIs and slice metrics; `report.json` writer conforming to `schemas/eval_report.schema.json` 1.0.0. Done when the harness runs on synthetic fixtures and produces a valid report, with unit tests for each metric against hand-computed cases. Do NOT create `services/evalgate/gate/`, `artifact/`, or `benchmarks/` — later tasks own those subdirs. Keep entity extraction deterministic and configurable; never claim WER alone promotes.

## Wave 2

### P1-02 Fail-open behavior — backend — `services/ingest/` root only
Deps P1-01. Bounded in-memory/disk buffer when storage is down; health/backpressure metrics; tests kill Postgres/S3 fakes mid-stream, ingest accepts and later flushes or sheds with a defined signal, never blocking beyond a configured timeout.

### P1-04 Pipecat integration — sdk — `sdk/pipecat/`
Deps P1-03. Observer capturing per-turn user audio + STT/parser results; conversation ID propagation to OTel traces; example app. Pipecat is optional — provide a protocol-based shim and a fake conversation driver. Done when the example bot produces correct turn events into a local ingest fake.

### P1-05 Outcome connector — sdk — `sdk/connectors/`
Deps P1-01. Generic webhook/poller interface; mapping booking status → `outcome.status` and `final_record`. Done when a mock booking system updates outcomes for fixture calls.

### P1-06 Golden set — data — `services/datasets/golden/`
Deps P1-01. Random sampler 300–500 utterances stratified by call, not failure; export/import + second-reviewer flow; seal/hash + access audit log. Golden set is sealed (§2 invariant 5): no training, no tuning, changes only via audited second review. Use synthetic fixtures as the source pool. Done when versioned/hashed/immutable with second-review coverage recorded.

### P1-09 Consent and PII basics — governance — `services/ingest/consent/` + `services/datasets/filters/`
Deps P0-02, P1-01. Enforce consent flags on dataset eligibility; transcript PII detection incl. card numbers (always dropped); retention policy config for raw audio; tests proving non-consented and card-containing rows are excluded. Do not edit other ingest or datasets paths.

## Wave 3

### P1-08 Baseline measurement — ml-eval — `docs/baseline.md`
Deps P1-06, P1-07. Run harness on deployed STT + parser artifacts (fakes standing in for real artifacts), record baseline with CIs and slice breakdown; commit baseline report as champion reference. No promotion-threshold changes (escalation item).

## Known post-worker integration follow-ups (lead)

- **P1-02 → P1-01:** the fail-open worker is isolated in `services/ingest/failopen/` to avoid colliding with the concurrent ingest worker. After both pass review, the lead must wire its `BoundedBuffer`/metrics into the ingest POST handler and health endpoint, then add an integration test that storage failure does not block callers.
- **P1-04 → P1-03:** the Pipecat observer uses a narrow `TurnSink` protocol with a lazy `sdk.log_turn` adapter. After both pass review, the lead must confirm the real SDK API matches the adapter and switch the default sink from in-memory to the SDK.
- **P1-05 → P1-01:** the outcome connector emits `PATCH /v1/calls/{call_id}/outcome` payloads; verify the ingest endpoint accepts them after both land.

## Vocabulary and rules notes (from P2-08 re-review, CP-2)

- P2-04 metrics were generated from synthetic fixture replay only. Real CTranslate2/ONNX hotword, initial-prompt and `condition_on_previous_text` behaviour, including empty/truncated outputs on converted models, must be validated in P3-04 against the deployed artifact.
- Room aliases remain limited to phrases observed in fixtures. Any real review-data alias additions require audit; no automatic learning from the sealed golden set is implemented.
- WER improvement from rules is fixture evidence only and never promotion justification (§2 invariant 4).

## Review adapter notes (from P2-03 review, CP-2)

- Real Label Studio/Argilla compose service and Postgres `ReviewStore` are documentation-only seams. The lead must wire both before CP-2 or any production claim; the fake-tool passing result is not human review.
- `Stage2Answer.warnings` are copied verbatim from the tool today. Sanitise/allowlist warning codes when a real adapter is added so tool-supplied text cannot carry PII into the result store.
- OpenAPI 1.0.0 has `additionalProperties: false`; warnings and original item ID remain internal. Exposing them over HTTP requires a lead-owned contract bump plus a `docs/CHANGELOG.md` entry.

## Benchmark and release-sync notes (from P3-05/P3-06 reviews, CP-3)

- P3-05 reports are SYNTHETIC/FIXTURE STAND-IN only. Both candidates' embedded reports validate against `eval_report.schema.json`; comparison CIs are candidate A−B deltas, not absolute intervals, and no promotion/champion decision is made. Real Whisper/Parakeet/Qwen3-ASR adapters are seams only. The sealed golden stand-in and fixture hashes were verified unchanged.
- P3-06 consumes gate aliases read-only and never flips aliases or changes thresholds. Failed releases are not auto-retried; retries belong in the caller's scheduler with a bounded budget, not inside `tick()`. For real serving adapters, harden rollback by comparing post-rollback state against `SwapPlan.before` and downgrading `reverted` to `rollback_failed` on mismatch.
- Gate-side write-ahead audit from P2-09 remains a lead follow-up; P3-06 only logs its own structured `release_sync.tick` events.

## Dataset builder notes (from P2-05 re-review, CP-2)

- P2-13 failure injection must include a malformed slot object with a card number as a mapping key; `_iter_json_strings` screens string values, and `additionalProperties: false` prevents conforming review results from expressing that case.
- Exporters are labelled plain JSON/JSONL stand-ins. With optional dependencies, P2-13 must verify real `NeMo`, `datasets.load_dataset("json", ...)` and TRL loading.
- The sealed synthetic golden stand-in excludes all six fixture calls; CP-2 must wire real review-adapter output and a human-reviewed golden set before any production claim.
- Task wording says `generated_at`; schema 1.0.0 requires `created_at`. The builder follows the schema; no code change needed.

## Pipeline integration notes (from P2-10 review, CP-2)

- Production adapters are unwired: the lead must inject P2-05 dataset builder, P2-06 trainer and P1-07 evaluator as `PipelineSteps`; non-`--dry-run` CLI exits 2 by design.
- "Verified since last run" currently comes from synthetic fixture IDs in `--state`; a real verified-example source must be supplied by the caller.
- Drift is a boolean flag/env input; the real detector is P4-02. Chain stops at the eval report path: no promotion, alias flip or threshold change.
- Full-repo ruff/mypy findings in other agents' paths must be green before CP-2 merge.

## Triage queue notes (from P2-02 review, lead follow-ups)

- On small pools the realised random-slice share can sit outside 3–5% (36 → 2.78%); the behaviour is deterministic and documented, but `QueueStats.to_dict()` should expose requested/actual share and band deviation so orchestration can escalate a production drift.
- `random_slice_target=0` is accepted via env and disables the slice; leaving it unset is safe. Make the public bound `ge=1` or warn on zero to prevent an operator typo suppressing §2 #6.
- `QueueItem` records `cluster_id`/`cluster_size` but not collapsed member IDs; add `cluster_members` for auditable dedup/lineage.

## Baseline and governance escalation (from CP-1 review, §14)

- P1-08 baseline is a SYNTHETIC STAND-IN only. Re-run the harness on real converted/quantised STT + parser artifacts with production decoding settings before making any CP-1 production claim or replacing the champion reference.
- P1-09 retention is an engineering placeholder, not a legal or compliance decision. `docs/compliance.md` stays `OPEN — counsel review required`; no numeric retention decision has been made.
- Transcript screening remains `stt.text`-only. Reviewed/corrected transcripts and audio redaction are P3-10 scope; P2-05 must screen slots/context as well (review finding under rework).

## Failure scoring notes (from P2-01 review, lead follow-ups)

- Clean turns (26/36) truthfully carry `signals: []`; do not fabricate signals to satisfy a literal reading of "each score shows which signals fired" — that would destroy the failure/clean separation required by §2 #6.
- Env overrides can currently make a non-mismatch weight exceed `final_record_mismatch` (`services/triage/scoring/settings.py:30-37`); defaults satisfy the highest-weight rule. Add a validator if the invariant must hold under all configs.

## Promotion gate escalation (from P2-09 review, §14)

- New engineering defaults: `min_e2e_margin=0.01`, `min_ci_level=0.95`, CI tolerance `0.0` (`services/evalgate/gate/settings.py:4-6`). These are NOT approved production thresholds; §14 requires human sign-off before any real promotion uses them.
- `other_languages` has zero turns in the synthetic fixtures, so the gate correctly fails closed on real fixture reports. A real multilingual slice with evidence is required before STT promotion work (P3-04/P3-08).
- Reviewer trust boundary: CI deltas are treated as authoritative by design; P3-04 must emit internally consistent absolute metrics and deltas. Hard-crash alias/audit atomicity is deferred to P3-06's MLflow backend (write-ahead audit recommended).

## Fail-open integration notes (from P1-02 review, lead follow-ups)

- `BoundedBuffer.enqueue()` is not yet wired into the validated ingest POST path; `flush()` is not called from a background/shutdown hook; `health().to_dict()` is not merged into `/healthz`. App wiring is required at CP-1.
- Decide whether to replace or compose with P1-01's `services/ingest/queue.py` — both are additive today.
- Flush is at-least-once: a timed-out sink write can complete later while the event is requeued. The `StorageSink` must stay idempotent on `event_id` (P1-01 already promises this).
- Outage coverage uses fakes/gates, not real Postgres/S3 kills. CP-1 must include a real-service failure injection rehearsal before claiming the done-when literally.

## Trainer core notes (from P2-06 review, lead follow-ups)

- Default job lock is per-`TrainerRunner` instance, not per-interpreter (`services/trainers/core/lock.py:52-57`, `runner.py:52`). Two runners can train concurrently today. Fix to a shared module-level lock, correct the "per-interpreter" wording, and add a two-runner contention test. Must be closed at the P2-10 integration checkpoint before process-wide locking is claimed.
- Missing `lineage` in a plugin result yields a bare `KeyError` (`runner.py:90`) instead of the documented `ValidationError`; schema makes lineage optional. Fix with `result.get("lineage")` and an explicit validation error plus test.

## Outcome connector notes (from P1-05 review, lead follow-ups)

- Poller source outage is logged but not counted (`sdk/connectors/poller.py:43-47`); `SyncReport.failed` stays 0 for a total source outage. Fail-open holds, but observability is misleading. Fix by propagating a failure signal so `failed` increments.
- Mapping failures increment `failed` (`sdk/connectors/connector.py:145-146`) while README/handoff call them "skipped". Align wording or counters.
- `sdk/connectors/mapping.py:172-178` validates `final_record` twice: once via `validate_booking()` (honours `VOICELOOP_BOOKING_SCHEMA`) and once against the bundled schema unconditionally. A project-specific valid record can be rejected. Fix by relying on `validate_booking()` alone and adding an env-override test.

## Evaluation harness notes (from P1-07 review, lead follow-ups)

- `confidence_intervals.*` bound **challenger − champion deltas**, not absolute metric values. The schema is satisfied but the example looks absolute. Lead must state this in `docs/contracts.md` / schema description before P2-09 builds the gate; gate code must not read CIs as absolute intervals. If absolute intervals are required, add a separate field with a schema version bump.
- `--eval-set golden|regression` currently labels the report without filtering rows (all runs score the non-sealed synthetic fixtures). Restrict accepted sets to `general_capability` until real set manifests exist (P1-06/P1-08), or filter by set membership. Never let a labelled "golden" report be mistaken for the sealed set (§2 invariant 5).
- `artifact_uri` defaults to `fake://evalgate/identity` even for oracle-vs-identity runs. Derive from challenger kind or require explicit URI for non-identity runs.
- Slice metrics can report `turns: 0` with perfect scores for neutral slices (e.g. `other_languages`). P2-09 must gate on `turns > 0` before treating a slice as "no regression".
- E2E booking accuracy from audio is the ONLY promotion metric (§2 invariant 4); better WER alone must never promote.

## CP-1 gate (lead)

**PASS (provisional, synthetic-fixture scope only; NOT production readiness).** Independent reviewer verified SDK `log_turn` → HTTP ingest → in-memory fakes, golden seal, baseline labelling and all nine Phase 1 handoffs. At review time repo-wide gates were red only in untracked Phase 2/3 WIP; Phase 1-scoped gates were green. Restore full-repo green before CP-2 and complete the production-readiness follow-ups listed in `docs/PROGRESS.md`. No promotion on the synthetic baseline.

## Phase 2 wave plan (pre-drafted)

- **Wave A (after CP-1, parallel):** P2-01 `services/triage/scoring/` · P2-05 `services/datasets/` root · P2-06 `services/trainers/core/` · P2-08 `services/trainers/vocab/`.
- **Wave B:** P2-02 `services/triage/queue/` (after P2-01) · P2-09 `services/evalgate/gate/` (after P2-06 + P1-07) · P2-12 `services/datasets/lineage/` + `services/*/auth/` (after P2-05) · P2-10 `infra/pipelines/` (after P2-05 + P2-06).
- **Wave C:** P2-03 `services/review_adapter/` (after P2-02) · P2-07 `services/trainers/parser/` (after P2-05 + P2-06) · P2-11 `infra/dashboards/` (after P2-02).
- **Wave D:** P2-04 `services/review_adapter/metrics/` + `docs/review_metrics.md` (after P2-03) · P2-13 `tests/integration/` (after P2-03, P2-07, P2-09).
- **Boundary rules:** P2-05 owns `services/datasets/` root only; it may import but not edit `golden/`, `filters/`, `lineage/`. P2-12 may add `auth/` subpackages inside services but must not edit a service's core files. P2-11 and P2-10 own disjoint `infra/` subdirectories. P2-09 owns `services/evalgate/gate/` only; P1-07 owns the evalgate root. P2-13 is the only task writing `tests/integration/`.
- **CP-2 invariant coverage:** P2-13 must produce a matrix mapping each of the 11 invariants in `context.md` §2 to at least one test (e.g. fail-open SDK test, golden-set sealed test, converted-artifact gate test, e2e-accuracy promotion test, call-level split property test, consent/PII filter test, lineage test, alias-flip rollback test, cheapest-tier routing test). Gate refuses promotion on deliberately worse models; rollback restores the previous champion.

## Phase 3 wave plan (pre-drafted)

- **Wave A (after CP-2):** P3-01 `services/trainers/whisper/` · P3-02 `services/trainers/nemo/` · P3-05 `services/evalgate/benchmarks/` · P3-06 `services/release/sync/` · P3-09 `services/datasets/golden/` (audited extension only; sealed-set rules still apply) · P3-10 `services/datasets/redaction/`.
- **Wave B:** P3-03 `services/datasets/augment/` (after P3-01) · P3-04 `services/evalgate/artifact/` (after P3-01) · P3-07 `services/release/shadow/` (after P3-06).
- **Wave C:** P3-08 `services/evalgate/gate/` (after P3-04 + P3-07; only the lead may change promotion thresholds, and context.md §14 says escalate for that) · P3-12 `tests/integration/` (after P3-01, P3-04, P3-07).
- **CP-3 gate:** STT path gated end to end with shadow evidence; broken/empty/hallucinating artifacts caught before promotion; alias rollback rehearsed.
- **P3-11 note:** compliance is a questionnaire only; no legal conclusion may be treated as final by any later task.

## Phase 4 wave plan (pre-drafted)

- After CP-3, all lanes are independent: P4-01 `services/release/canary/`, P4-02 `infra/pipelines/drift/`, P4-03 `ui/`, P4-04 `services/triage/teacher/`, P4-05 cross-cutting tenant scoping (lead — needs its own contract bump and an explicit file list before workers touch it), P4-06 `docs/runbooks/`.
- P4-03 go/no-go must use P2-04 review metrics (which clicks hurt), not a subjective preference. P4-06 runbooks must be followed by someone who did not write them before completion.
