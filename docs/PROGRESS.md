# VoiceLoop build progress (lead-owned)

Status legend: `[ ]` todo · `[~]` worker active / in review · `[x]` reviewed and merged · `[!]` blocked/escalated.

Update rule: a task becomes `[x]` only after an independent reviewer subagent reports PASS and the lead commits it. Checkpoints become `[x]` only after their gate review.

## Phase 0 — Foundations and contracts

- [x] P0-01 Repo skeleton and tooling
- [x] P0-02 Turn event schema v1
- [x] P0-03 Booking JSON Schema (example)
- [x] P0-04 Interface contracts
- [x] P0-05 Synthetic fixtures
- [x] CP-0 Schemas/contracts reviewed; compose validated; 36 synthetic fixtures available (P0-05 reviewer PASS).

## Phase 1 — Capture and baseline

- [x] P1-01 Ingest API
- [x] P1-02 Fail-open behaviour (fake-storage probes; app wiring pending)
- [x] P1-03 Python SDK
- [x] P1-04 Pipecat integration
- [x] P1-05 Outcome connector
- [x] P1-06 Golden set sampling and labelling workflow (synthetic stand-in)
- [x] P1-07 Evaluation harness v0
- [x] P1-08 Baseline measurement (provisional synthetic stand-in only)
- [x] P1-09 Consent and PII basics
- [x] CP-1 — PROVISIONAL, SYNTHETIC-FIXTURE SCOPE ONLY — NOT PRODUCTION READY.
  Independent reviewer PASS: public SDK `log_turn` → HTTP ingest → in-memory
  Postgres/S3 fakes verified with `tests/fixtures/events/call-0001/turn-001.json`
  (GET /v1/turns/{id} 200 `persisted`; GET /v1/calls/{id} 200; duplicate idempotent
  `X-VoiceLoop-Idempotent: hit`; invalid event 422 `{code,message,details}`).
  Outcome connector PATCH merged `edited_by_staff` + final_record at call level;
  ingest stores outcomes per call, connector fake merges all 6 fixture turns.
  Golden stand-in sealed: independent SHA-256 662cdaf43ee2d5fde03faf5e69fde196114d4f312406d94c00bf53714d03da8e
  matches manifest/seal/docs; read-only seal verify ok, 36 synthetic labels / 6 calls.
  Baseline is SYNTHETIC STAND-IN / NOT A PRODUCTION BASELINE; PRIMARY metric is
  e2e booking-field accuracy from audio; no promotion or champion claim.
  Limitations: no Docker/real Postgres/S3/GPU/deployed STT-parser artifacts;
  no human-reviewed golden set; P1-02 buffer not wired into ingest app and outage
  coverage is fake-only; promotion thresholds pending §14 sign-off; retention is an
  engineering placeholder and compliance OPEN. At CP-1 review time, repo-wide
  pytest/ruff/format/mypy were red ONLY in untracked Phase 2/3 WIP
  (P2-07/P2-10/P2-11/P3-05); Phase 1-scoped gates green (245 pytest passed,
  ruff clean, mypy 109 files clean). No promotion on this baseline; real-artifact
  re-run required before production readiness.

## Phase 2 — Triage, review, parser loop, gate

- [x] P2-01 Failure scoring
- [x] P2-02 Sampling, clustering, queue
- [~] P2-03 Review adapter
- [ ] P2-04 Review UX measurement
- [~] P2-05 Dataset builder
- [x] P2-06 Trainer plugin interface
- [~] P2-07 Parser trainer
- [~] P2-08 Vocabulary and rules tier
- [x] P2-09 Gate and promotion logic (thresholds pending sign-off)
- [x] P2-10 Pipelines and triggers
- [x] P2-11 Dashboards
- [ ] P2-12 Lineage and access control
- [ ] P2-13 Loop rehearsal
- [ ] CP-2 First full parser-path loop; all §2 invariants tested

## Phase 3 — STT path, release safety

- [ ] P3-01 Whisper trainer
- [ ] P3-02 Parakeet/NeMo trainer
- [ ] P3-03 Augmentation
- [ ] P3-04 Conversion and artifact tests
- [~] P3-05 Benchmark harness for base-model swaps
- [~] P3-06 Alias sync agent
- [ ] P3-07 Shadow mode
- [ ] P3-08 Promotion workflow update
- [ ] P3-09 Golden-set maintenance
- [ ] P3-10 Audio redaction and pseudonymisation
- [x] P3-11 Compliance checklist
- [ ] P3-12 STT loop rehearsal
- [ ] CP-3 STT path gated end to end; rollback rehearsed

## Phase 4 — Hardening and scale

- [ ] P4-01 Canary with auto-rollback
- [ ] P4-02 Scheduled and drift-triggered retraining
- [ ] P4-03 Custom review UI
- [ ] P4-04 Teacher-model triage signal at scale
- [ ] P4-05 Multi-project support
- [ ] P4-06 Operations docs

## Wave plan and path boundaries (lead)

- **CP-0 gate:** P0-01 re-review → P0-05 review → lead checkpoint review (schemas, contracts, compose YAML, fixtures). Resolve the `schemas` vs `voiceloop_common.schemas` split (documented in `docs/conventions.md`).
- **Phase 1 wave 1 (parallel):** P1-01 `services/ingest/` root · P1-03 `sdk/` root · P1-07 `services/evalgate/` root.
- **Phase 1 wave 2 (parallel, after wave 1):** P1-02 `services/ingest/` fail-open root files · P1-04 `sdk/pipecat/` · P1-05 `sdk/connectors/` · P1-06 `services/datasets/golden/` · P1-09 `services/ingest/consent/` + `services/datasets/filters/`.
- **Phase 1 wave 3:** P1-08 `docs/baseline.md` → CP-1 gate.
- **Boundary rules:** P1-01 must not create `services/ingest/consent/` (owned by P1-09). P1-07 owns `services/evalgate/` root only; later gate/artifact/benchmarks tasks own their own subdirectories. P2-05 owns `services/datasets/` root and must only import, not edit, `golden/` and `filters/`.

**Overall:** 21/40 tasks reviewed · 2/4 checkpoints passed.
