# VoiceLoop interface contracts (v1)

Authoritative cross-service interfaces live in `schemas/` as JSON Schema (draft 2020-12) or OpenAPI 3.1, with examples in `schemas/examples/`. Services must implement these contracts rather than defining parallel shapes in their own code. This document is the human-readable index; the files are the source of truth.

## Summary

| Contract | File | Version | Shape | Implemented by (roadmap) | Consumed by (roadmap) |
|---|---|---|---|---|---|
| Trainer `train(dataset_manifest, config) -> {artifact_uri, metrics, run_id}` | `schemas/trainer_contract.schema.json` | 1.0.0 | JSON Schema, `oneOf` input/output payloads | P2-06 (plugin base + registry), P2-07 (parser), P3-01 (Whisper), P3-02 (Parakeet) | P2-10 (dataset → train → evaluate pipeline), P2-09 (evaluates the returned `artifact_uri`) |
| Evaluation `evaluate(artifact_uri, eval_sets, config) -> report.json` | `schemas/eval_report.schema.json` | 1.0.0 | JSON Schema for `report.json` | P1-07 (harness + report writer), P2-09 (gate), P3-04 (artifact tests), P3-05 (benchmarks), P3-08 (shadow requirement) | P1-08 (baseline), P2-09 (promotion rule), P3-06/P3-07 (release and shadow evidence) |
| Dataset manifest | `schemas/dataset_manifest.schema.json` | 1.0.0 | JSON Schema | P2-05 (dataset builder; P1-09 filters feed it) | P2-06/P2-07/P3-01/P3-02 (trainers), P2-12 (lineage and deletion), P2-10 (orchestration) |
| Review adapter API | `schemas/review_adapter.openapi.yaml` | 1.0.0 | OpenAPI 3.1, four endpoints | P2-03 (adapter wrapping Label Studio Community or Argilla) | P2-02 (triage pushes items), P2-05 (dataset builder pulls results), P2-04 (review UX metrics), P4-03 (custom UI) |

## Trainer contract

`schemas/trainer_contract.schema.json` validates a single call payload. The `contract_part` discriminator is `input` for the arguments to `train` and `output` for its return value; both branches require `schema_version`.

- Input requires `dataset_manifest` (an object with at least a `content_hash` string, normally the full dataset manifest) and `config` (object). Model selection must come from settings, not a hard-coded name.
- Output requires `artifact_uri`, numeric `metrics` and `run_id`. `artifact_uri` must address the deployable artifact (merged adapter or converted model), not a raw checkpoint, because the gate evaluates what will serve (context.md §2 invariant 3).
- Optional `lineage` carries `dataset_hash`, `git_sha` and `image_digest` together; production jobs must emit it (context.md §2 invariant 9).
- Examples: `schemas/examples/trainer_contract.input.json`, `schemas/examples/trainer_contract.output.json`; a deliberately invalid output is in `schemas/examples/trainer_contract.invalid.json`.

## Evaluation report

`schemas/eval_report.schema.json` defines `report.json` from `evaluate(artifact_uri, eval_sets, config)`.

- `eval_sets` values: `golden`, `regression`, `general_capability` (the last covers silence, line noise and other served languages).
- Core metrics: `wer` (>= 0, may exceed 1 when insertions dominate), `entity_error`, `slot_f1`, `schema_valid_rate`, `e2e_booking_accuracy` (the remaining four are all 0..1).
- **`e2e_booking_accuracy` is the primary promotion metric.** WER alone never justifies promotion (context.md §2 invariant 4). Each core metric requires a paired-bootstrap `confidence_intervals` entry with `lower`, `upper` and `level`; an interval missing a bound is invalid.
- `slice_metrics` is keyed by slice (recommended: `names`, `dates`, `numbers`, `silence`, `other_languages`) and maps metric names to numbers. Gate regression checks use these slices and the CI bounds.
- Optional: `decoding_settings` (record the exact production decoding settings), `shadow_evidence` (needed before STT promotion, P3-08), `champion_report_uri`, `gate_decision`.
- Examples: `schemas/examples/eval_report.valid.json` (full report) and `schemas/examples/eval_report.invalid.json` (missing a CI upper bound).

## Dataset manifest

`schemas/dataset_manifest.schema.json` describes a versioned, hashed dataset.

- Required: `schema_version`, `dataset_id`, `content_hash`, `source_call_ids`, `splits`, `filter_counts`, `created_at`, `examples`.
- `splits.train`, `splits.val` and `splits.test` are **arrays of call IDs**, not counts and not utterance IDs. The arrays must be pairwise disjoint and together cover `source_call_ids`; the builder enforces this with property tests because plain JSON Schema cannot express it (context.md §2 invariant 7). Golden and regression calls are excluded from all three.
- `filter_counts` audits `consent_dropped`, `pii_dropped`, `replay_added`, `golden_excluded`, `regression_excluded` and `synthetic_tagged` as non-negative integers. Optional `synthetic: true` marks datasets containing augmented/synthesised rows.
- Examples: `schemas/examples/dataset_manifest.valid.json` and `schemas/examples/dataset_manifest.invalid.json` (missing `pii_dropped`).

## Review adapter API

`schemas/review_adapter.openapi.yaml` is the stable HTTP boundary for human review.

| Endpoint | Purpose |
|---|---|
| `POST /v1/items` | Push an item (`call_id`, `turn_index`, `audio_uri`, `hypothesis`, `low_confidence_spans`, `blind`, `schema_version`); returns the item with `item_id` and `status`. |
| `GET /v1/items/{item_id}` | Poll item status. |
| `POST /v1/items/{item_id}/result` | Submit `reviewer_id`, `corrected_transcript`, `slots`, `routing`, `started_at`, `completed_at`. |
| `GET /v1/items/{item_id}/result` | Pull the completed result for the dataset builder. |

`RoutingLabel` is exactly `stt_wrong`, `parser_wrong`, `both_fine` or `unusable_audio` and drives the adaptation ladder (context.md §7). `slots` is validated against the project's `schemas/booking.schema.json`. Audio is referenced by URI only and is never stored in Postgres.

**The adapter hides the Label Studio/Argilla choice.** Callers depend only on this API; swapping tools must require changes only inside `services/review_adapter`. Example payloads are embedded in the OpenAPI document and mirrored in `schemas/examples/review_adapter.*.json` (push item, item, result, error).

## Structured errors

Every VoiceLoop HTTP error response, including all review adapter endpoints, uses one shape:

```json
{
  "code": "validation_error",
  "message": "low_confidence_spans[0].start_ms must be a non-negative integer.",
  "details": { "field": "low_confidence_spans[0].start_ms", "received": -1 }
}
```

- `code`: stable machine-readable identifier (e.g. `validation_error`, `not_found`).
- `message`: human-readable summary safe to log.
- `details`: object with structured context (field names, rejected values); never raw guest audio or PII.
- Trainers and evaluation jobs expose the same shape when their failures are surfaced over HTTP or written to logs; CLI failures should map their exception class to a `code`.

## Versioning and validation

All four contracts are at version **1.0.0**. JSON Schemas declare `"$schema": "https://json-schema.org/draft/2020-12/schema"` and an `$id`; payloads carry a `schema_version` string pinned by `const`. The OpenAPI document declares `openapi: 3.1.0` and `info.version: 1.0.0`.

A contract change requires a version bump here and in the affected payloads, a note in `docs/CHANGELOG.md` (lead/docs-owned), and updates to the examples and tests in `tests/unit/contracts/`. Validate with `jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())` so `date-time` formats are checked, and parse the OpenAPI file with `yaml.safe_load`.
