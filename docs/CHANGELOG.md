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

### Changed

- `P0-01`: extended seeded `pyproject.toml` with `pre-commit` in the `dev`
  extra.
- Lead decision (CP-0): two schema helper APIs are intentional —
  `voiceloop_common.schemas` for runtime code (raises
  `voiceloop_common.errors.ValidationError`) and the `schemas` package for
  contract authoring/fixture tests (relative paths, `jsonschema.ValidationError`).
  No third implementation should be added.

No schemas or contract versions were changed by `P0-01`.
