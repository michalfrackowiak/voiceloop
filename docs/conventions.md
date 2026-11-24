# Shared engineering conventions (lead-owned)

Applies to every agent and every task. Read `context.md` (especially §2 invariants and §14 working agreement) and your task in `roadmap.md` before writing code. If a task conflicts with an invariant, stop and escalate.

## Language and layout

- Python 3.11+ syntax (CI targets 3.11; local toolchain may be newer). Type hints on all public functions; docstrings with a one-line summary and, where useful, Args/Returns/Raises.
- Flat namespace packages in the repo root (no `src/`):
  - `voiceloop_common/` — shared utilities, imported by every component.
  - `schemas/` — JSON Schema files plus small Python loaders/validators.
  - `services/<name>/` — one package per service in `context.md` §4.
  - `sdk/` — Python SDK, `sdk/pipecat/`, `sdk/connectors/`.
  - `tests/unit/`, `tests/integration/`, `tests/fixtures/`.
  - `infra/`, `docs/`, `ui/`.
- Each service package has `__init__.py`, `README.md` (run/test/contract), and its own tests under `tests/unit/<service>/`.

## Shared utilities (`voiceloop_common`)

Use these instead of reimplementing:

- `settings.BaseSettings` — pydantic-settings base class; each component subclasses with its own `env_prefix`.
- `logging.configure_logging(service_name: str)` — JSON-formatted structured logging.
- `errors` — `VoiceLoopError` base plus `ConfigError`, `ValidationError`, `StorageError`, `TransientError`.
- `voiceloop_common.schemas` — `schema_path(name)`, `load_schema(name)`, `validate(instance, schema_name)`. Wraps `jsonschema`; raises `voiceloop_common.errors.ValidationError` and accepts a bare schema name. **Use this in services, SDK and trainers.**
- The `schemas` Python package exposes a parallel, lower-level helper set (`schema_path`/`load_schema`/`validate`) that accepts relative paths under `schemas/` (including `examples/...`) and raises `jsonschema.ValidationError`. This split is deliberate: contract-authoring and fixture tests use the low-level API; runtime code uses `voiceloop_common`. Do not add a third implementation.
- `clock.utcnow()` — injectable UTC clock for deterministic tests.
- `ids.new_id()` — UUID helpers.

## Contracts first

- Cross-service interfaces live in `schemas/` as JSON Schema or OpenAPI YAML, with examples. Never inline an interface definition in a service and call it the contract.
- Files owned by contract tasks (do not overwrite from another task):
  - `schemas/turn_event.schema.json` (P0-02)
  - `schemas/booking.schema.json` (P0-03)
  - `schemas/trainer_contract.schema.json`, `schemas/eval_report.schema.json`, `schemas/dataset_manifest.schema.json`, `schemas/review_adapter.openapi.yaml` (P0-04)
- Schema changes require a version bump and a `docs/CHANGELOG.md` note.

## Configuration

- Typed settings objects built from environment variables. No secrets in the repo. No hard-coded base model names, thresholds, tool choices, or file paths where a setting is appropriate.
- Defaults for unanswered questions come from `roadmap.md` "Decisions pending" (e.g. Label Studio Community behind the review adapter, faster-whisper/CTranslate2 STT, small LLM + LoRA parser, English-only, mock booking connector, nightly non-serving GPU, 5 h/week reviewer capacity).

## Testing and quality

- `uv run pytest` must pass locally from the repo root; `pyproject.toml` sets `pythonpath = ["."]`.
- `uv run ruff check .` and `uv run mypy .` must be clean (config in `pyproject.toml`; ignore third-party stubs where needed).
- Unit tests per task; integration tests per phase in `tests/integration/`.
- Synthetic/anonymised fixtures only. Never commit real guest audio or transcripts.
- Property tests are preferred for invariants (e.g. "no call appears in two splits", "failure score orders known failures above clean turns").
- Heavy ML/runtime deps (torch, transformers, NeMo, vLLM, CTranslate2, MLflow, Label Studio, Postgres drivers) are **optional**. Put real integrations behind interfaces and provide in-memory/fake backends selected by settings so the default test suite runs without GPU, network, or services. Real backends are exercised via marked integration tests that skip when unavailable.

## Fail-open rule (non-negotiable)

Anything on or near the call path (SDK, ingest) must return within a bounded time and must never raise into the caller. Use bounded queues, timeouts, circuit breakers, and drop-on-failure with metrics. Storage failures are buffered or shed with a defined signal.

## Governance defaults

- Consent flags gate dataset eligibility. Transcript PII detection incl. card numbers; card numbers always dropped. Audio never in Postgres. Full lineage example → call → model. Roles: reviewer, ml_engineer, admin.
- Escalate to the lead (stop work) for legal/compliance conclusions, golden-set changes, promotion-threshold changes, or any §2 conflict.

## Definition of done

- Acceptance criteria met; tests pass; service README and `docs/CHANGELOG.md` updated.
- A handoff note at `docs/handoffs/<task-id>.md` using the template at the end of `roadmap.md`.
- No invariant violated. Workers do **not** commit (concurrent agents share the index); the lead commits per task with a Conventional Commit message including the task ID, e.g. `feat(P0-01): repo skeleton and tooling`.
