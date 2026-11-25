# P4-05 Multi-project support — isolated fake seam

> **SYNTHETIC/FIXTURE STAND-IN - NOT A MULTI-PROJECT PRODUCTION GUARANTEE**

**STATUS: NOT DONE / BLOCKED** — this is a self-contained, testable namespace
and registry seam only. The roadmap task `P4-05 Multi-project support` remains
`[ ]` (not done). `roadmap.md` was not edited; its existing status already
records that fact. Full completion requires a lead-owned cross-service contract
bump and wiring through active lanes, neither of which is performed here.

The only artefact label used here is the internal one:
`voiceloop.project_scope` / `internal-1.0.0`.
**No cross-service contract version bump is implied or performed.** No schema,
fixture, sealed artifact, active service, SDK, UI, pipeline or integration test
is read for behaviour or written by this package. There is no network, real
database or GPU access and no new dependency.

## Public API

| Module | Entry points |
|---|---|
| `ids` | `ProjectId`, `InvalidProjectIdError`, `validate_project_id` |
| `registry` | `ProjectRegistry` (protocol), `InMemoryProjectRegistry`, `ProjectConfig`, `UnknownProjectError`, `DisabledProjectError`, `DuplicateProjectError`, `InvalidProjectConfigError` |
| `scope` | `ProjectScope`, `ProjectTag`, `LabelledValue`, `Tagged`, `assert_same_project`, `audio_object_key`, `dataset_id`, `artifact_uri`, `report_path`, `mlflow_alias`, `mlflow_model_name`, `all_namespaces`, `ProjectScopeError`, `MissingProjectTagError`, `ProjectScopeMismatchError` |
| `settings` | `load_project_settings()` (public factory), `INVALID_PROJECT_SETTINGS_MESSAGE`; `ProjectSettings` is a typed internal class (`VOICELOOP_PROJECTS_` prefix) |

All symbols above are re-exported from `services.projects`, together with the
constants `STANDIN_LABEL`, `STANDIN_TOKEN`, `INTERNAL_SCOPE_LABEL`,
`INTERNAL_SCOPE_VERSION`, `CONTRACT_NOTE` and `INVALID_PROJECT_SETTINGS_MESSAGE`.
Construct settings only through `load_project_settings()`: it replaces a raw
pydantic `ValidationError` with a fixed-code `ConfigError` whose stable message
never contains the rejected value or validation details.

## What this seam guarantees

- `ProjectId` is an immutable lowercase pseudonymous slug (2–64 characters,
  letter-led, single-hyphenated ASCII alphanumerics). Missing, blank, malformed
  and email/phone/card-looking values raise `InvalidProjectIdError`. Errors use
  fixed codes and never echo the rejected value or a caller-supplied `field`;
  unknown field names fall back to the fixed `project_id` label.
- `InMemoryProjectRegistry` maps a validated ID to a booking schema path, a
  non-empty tuple of eval-set names and an enabled flag. Unknown, disabled,
  invalid or missing IDs fail closed (`get()` returns `None`; `require()` raises
  a fixed-code error). Duplicate registrations are rejected.
- All namespace builders are pure string functions: identical inputs yield
  identical outputs and no state is read or mutated. Two synthetic projects
  (`synthetic-alpha`, `synthetic-beta`) are provably disjoint across audio
  object keys, dataset IDs, artifact URIs, report paths, MLflow aliases and
  registered model names.
- **Every output is labelled.** Each builder returns a `LabelledValue` whose
  `label` is exactly `SYNTHETIC/FIXTURE STAND-IN - NOT A MULTI-PROJECT PRODUCTION GUARANTEE`,
  whose string embeds a deterministic slug of that label, and whose `to_dict()`
  adds the internal label/version. `assert_same_project(...)` returns the shared
  `ProjectId` only when every supplied tag is present, correctly labelled and
  identical; mismatch, missing tags, foreign labels and empty calls fail closed.

## Run

There is no server, CLI, database or object store. From the repository root:

```powershell
.\.venv\Scripts\python.exe -c "from services.projects import InMemoryProjectRegistry, ProjectConfig, ProjectId, ProjectScope, all_namespaces, assert_same_project; a = ProjectId.parse('synthetic-alpha'); r = InMemoryProjectRegistry([ProjectConfig(a, 'schemas/booking.schema.json', ('golden', 'regression'))]); c = r.require(a); s = ProjectScope.for_project(c.project_id); v = all_namespaces(s); print(assert_same_project(*v.values())); print(v['dataset_id'].label); print(v['dataset_id'].value)"
```

The printed label is the mandatory stand-in label above, and the printed value
is a fake namespace string. Nothing is written to disk. The registry does not
open or validate the referenced schema file; it only stores the safe relative
path, leaving real contract wiring to a lead-approved lane.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/projects -q
.\.venv\Scripts\python.exe -m ruff check services/projects tests/unit/projects
.\.venv\Scripts\python.exe -m ruff format --check services/projects tests/unit/projects docs/handoffs/P4-05.md
.\.venv\Scripts\python.exe -m mypy services/projects tests/unit/projects
```

## Contract notes

- Internal artefact label only: `voiceloop.project_scope` / `internal-1.0.0`.
  **No cross-service contract version bump is implied or performed**, and no
  `docs/CHANGELOG.md` entry is claimed by this task.
- Five contract files require a version bump and §14 lead approval before any
  real multi-project wiring: `schemas/turn_event.schema.json`,
  `schemas/dataset_manifest.schema.json`, `schemas/trainer_contract.schema.json`,
  `schemas/eval_report.schema.json` and `schemas/review_adapter.openapi.yaml`.
  `schemas/booking.schema.json` already supports a swappable per-project path
  via `schemas.booking.booking_schema_path`; selecting that path is not a schema
  change, which is why it is not one of the five.
- No service contract is consumed: this package imports only stdlib,
  `voiceloop_common` settings and its own modules; no active lane imports it.

## Roadmap done-when (P4-05, `roadmap.md`, verbatim)

> two projects run side by side with no data crossing

The extended cross-service requirements for this lane are: across schemas,
storage, registry, review, gate, datasets, SDK and pipelines; five contract
files require version bumps and §14 lead approval.

This seam does **not** meet that criterion. It neither runs two projects side by
side nor touches schemas, storage, registry integrations, review, gate,
datasets, SDK or pipelines. See `docs/handoffs/P4-05.md` for the blocking list
and the next lead-owned steps.
