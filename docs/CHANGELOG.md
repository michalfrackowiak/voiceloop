# Changelog

All notable changes to VoiceLoop are recorded here. Cross-service contract
changes require a schema version bump and an entry under the matching version.

## [Unreleased]

### Added

- `P0-01`: repository skeleton (`voiceloop_common/`, placeholders for
  `services/`, `sdk/`, and `tests/`, `infra/`).
- `P0-01`: shared helpers in `voiceloop_common` — typed settings, JSON
  structured logging, error hierarchy, injectable fail-open UTC clock, UUID
  IDs, and repository-root JSON Schema loader/validator.
- `P0-01`: ruff, mypy, pytest, and pre-commit configuration; Python 3.11 CI
  running lint, type check, and tests.
- `P0-01`: local Compose stack (Postgres 16, MinIO, MLflow with Postgres
  backend and S3 artifacts) plus `.env.example` with non-secret defaults.
- `P0-02`: `turn_event.schema.json` **1.0.0** — authoritative turn event
  contract (§5 of `context.md`), with incremental-adoption required fields and
  valid/invalid examples.
- `P0-03`: `booking.schema.json` **1.0.0** — example booking slots with
  cross-field validators and a swappable schema path.
- `P0-04`: `trainer_contract.schema.json`, `eval_report.schema.json`,
  `dataset_manifest.schema.json`, and `review_adapter.openapi.yaml` all at
  **1.0.0** — the four cross-service interface contracts.
- `P1-03`: fail-open Python SDK with bounded retry queue, circuit breaker and
  native-rate audio helpers.
- `P1-07`: evaluation harness v0 with WER, entity error, slot F1, schema-valid
  rate, e2e booking accuracy from audio, paired bootstrap CIs and slice metrics.
- `P1-04`: fail-open Pipecat observer with native-rate audio capture,
  conversation-ID trace propagation and a runnable example bot.
- `P1-05`: fail-open booking outcome connector (webhook + poller) with
  `call_id`-only join and status-to-outcome mapping.
- `P1-08`: provisional synthetic-stand-in baseline report with metrics, CIs and
  slice breakdown; explicitly NOT a production champion reference.
- `P1-01`: ingest API with idempotent turn POST, audio upload, OTel endpoint,
  late-outcome PATCH and Postgres migrations; fail-open bounded queue.
- `P2-06`: trainer plugin base/registry, pinned-image runner with single-job
  lock and MLflow lineage logging (fakes by default).
- `P1-06`: synthetic golden workflow stand-in with strict content-addressed
  seal, second-review gate, audited verification and amendment versioning.
- `P1-02`: bounded in-memory/disk fail-open buffer with health/backpressure
  metrics, deadline-bounded enqueue and at-least-once flush via fakes.
- `P2-09`: fail-closed promotion gate using end-to-end accuracy deltas only,
  zero-turn slice refusal, converted-artifact enforcement, alias flip/rollback
  with hash-chained audit (fakes; thresholds pending §14 sign-off).
- `P1-09`: consent eligibility filter and strict transcript PII screening with
  card-number hard-drop across separators and long digit runs; retention config
  is an engineering placeholder pending counsel.
- `P2-01`: deterministic failure scoring signals with strict final-record
  mismatch weight, per-turn explanations and injectable teacher seam.
- `P2-02`: seeded random slice outside the failure filter, near-duplicate
  clustering, daily cap/prioritisation and blind-item flags.
- `P2-10`: dry-run dataset→train→evaluate chain with validated CLI overrides,
  policy-gated cron ticks and fail-fast step reporting.
- `P2-11`: read-only static dashboards from synthetic/local providers with
  explicit provenance and no fabricated version or promotion.
- `P2-07`: deterministic parser LoRA training stand-in with content-addressed
  adapter export, fake runtime load/unload and schema-valid constrained decoding.
- `P2-05`: split-by-call dataset builder with non-vacuous property tests,
  content-addressed manifest, three export shapes, and card PII hard-drop
  across transcripts, hypotheses, nested slots and dialogue context.
- `P3-05`: fixture-only two-candidate benchmark harness with schema-valid
  reports, candidate A−B delta CIs and read-only sealed-golden handling.
- `P3-06`: read-only alias sync agent with hash-before-serve transactional
  STT/parser swaps and exact-version health-check auto-revert.
- `P2-03`: two-stage review adapter with schema-generated slot forms, blind
  prefill skip, routing labels and idempotent fake-tool result storage.
- `P2-08`: fixture-derived aliases/normalisation rules with schema-validated
  room canonicals, fail-closed date handling and an internal boosting config seam.
- `P3-07`: fake-only shadow mode with bounded fail-open mirroring, deferred
  candidate/champion runs and call_id-joined outcome comparison reports.
- `P2-04`: fixture-only review UX metrics with injected-clock timing, edit
  distances, label distribution, paired anchoring comparison and scripted
  inter-reviewer agreement; P4-03 go/no-go input only.
- `P3-10`: fake audio/text redaction with merged padded spans, consent-gated
  CLI, stable pseudonyms, cued-name expansion and raw-only retention rehearsal.
- `P2-12`: bidirectional lineage, tombstone-based deletion with affected-model
  flags, and immutable default-deny roles with fixed-code audit events.
- `P3-02`: fake Parakeet/NeMo fine-tune with content-addressed artifacts and
  schema-shaped per-word confidences wired through the public P2-01 scoring seam.
- `P3-03`: deterministic synthetic augmentation manifests with transform lineage,
  explicit golden/regression exclusion, and leak-rejection guards.
- `P4-04` (blocked work product only): read-only exploratory teacher-disagreement
  harness with per-depth counts and Wilson CIs; no precision-improvement claim
  and no roadmap completion.
- `P3-12`: fake STT loop rehearsal through Whisper conversion, artifact probe,
  evaluation, synthetic shadow evidence, gated promotion, serving sync and
  rollback; artifact/shadow failures refuse before promotion with hash-chained,
  PII-free audit.
- `P3-09`: audited golden-set extension harness with per-candidate second-review
  coverage, trusted reviewer pins, allowlisted actors and an explicit human-review
  drift gate; sealed golden artifacts remain read-only.
- `P3-04`: content-addressed fake artifact probes with eight decoding-setting
  combinations; decoding-hash tampering fails closed at all public seams and
  champion reports carry mandatory hallucination/forgetting diagnostics.
- `P3-08`: STT promotions require fresh, complete, synthetic-labelled shadow
  evidence; missing/partial/stale/mismatched evidence fails closed with
  structured, hash-chained, PII-free refusals. Parser and P2-09 rules unchanged.
- `P2-13`: synthetic parser-path loop rehearsal with exact P2-12 lineage queries,
  pre-hash tombstone exclusions, failure injections and a strict tracked XFAIL.
- `P3-01`: fake Whisper LoRA/conversion pipeline with call-disjoint replay,
  8 kHz-aware handling, content-addressed artifacts and logged scaling evidence.
- `P4-01`: deterministic fake canary split with paired outcome metrics and
  idempotent, hash-chained auto-rollback; no live traffic or registry evidence.
- `P4-02`: fake drift detectors and trigger policy; volume/drift/manual inputs
  decide retraining and calendar ticks alone never trigger.
- `P4-06` (blocked work product only): structured runbooks and a text-only
  walkthrough checker; no human follow-through is claimed and acceptance stays OPEN.
- `P4-03` (blocked work product only): deterministic synthetic review UI with
  waveform, schema form, blind/second-opinion flows and measurement-driven
  `no_go_fail_closed` go/no-go; no human usability evidence is claimed.
- `P4-05` (blocked work product only): synthetic project ID, registry and
  namespace-seam fakes; no contract bump, cross-service wiring or multi-project
  guarantee, and roadmap acceptance remains OPEN.
- `CP-3` (checkpoint, PASS at provisional synthetic-fixture scope only): clean
  `ecf0f62` archive/worktree reproduced all protected hashes and green scoped,
  full and static gates; STT path gated end to end with synthetic shadow
  evidence and rollback rehearsed. Not a production-readiness milestone.
- `CP-2` (checkpoint, PASS at provisional synthetic-fixture scope only): clean
  committed snapshot verified with all §2 invariants mapped to exercised fake
  behaviour; not a production-readiness milestone.

### Changed

- `P0-01`: extended seeded `pyproject.toml` with `pre-commit` in the `dev`
  extra.
- Lead fix (CP-3): path-scoped `.gitattributes` rules preserve the distinct LF
  and CRLF bytes expected by protected hash pins in fresh checkouts and
  archives; no schema content or pin changed.
- Lead decision (CP-0): two schema helper APIs are intentional —
  `voiceloop_common.schemas` for runtime code (raises
  `voiceloop_common.errors.ValidationError`) and the `schemas` package for
  contract authoring/fixture tests (relative paths, `jsonschema.ValidationError`).
  No third implementation should be added.

No schemas or contract versions were changed by `P0-01`.
