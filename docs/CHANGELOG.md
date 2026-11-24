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

### Changed

- `P0-01`: extended seeded `pyproject.toml` with `pre-commit` in the `dev`
  extra.
- Lead decision (CP-0): two schema helper APIs are intentional —
  `voiceloop_common.schemas` for runtime code (raises
  `voiceloop_common.errors.ValidationError`) and the `schemas` package for
  contract authoring/fixture tests (relative paths, `jsonschema.ValidationError`).
  No third implementation should be added.

No schemas or contract versions were changed by `P0-01`.
