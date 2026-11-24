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

### Changed

- `P0-01`: extended seeded `pyproject.toml` with `pre-commit` in the `dev`
  extra.

No schemas or contract versions were changed by `P0-01`.
