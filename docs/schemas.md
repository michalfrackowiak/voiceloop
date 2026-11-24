# JSON Schema contracts

This directory documents the versioned contracts owned by tasks P0-02 and P0-03. Schemas are JSON Schema **draft 2020-12** and live in [`schemas/`](../schemas/); examples live in [`schemas/examples/`](../schemas/examples/).

## Versioning rules

- Each schema declares `$schema`, a versioned `$id` (`https://voiceloop.example/schemas/<name>-<version>.schema.json`) and `schema_version` in semver form (`MAJOR.MINOR.PATCH`). Both schemas start at **1.0.0**.
- **Additive changes → minor bump.** Adding a new optional property or loosening a constraint is backwards-compatible for existing payloads. Update the examples and `$id` in the same change.
- **Breaking changes → major bump.** Removing or renaming a property, adding a required property, narrowing a type/enum/range, or otherwise rejecting previously valid payloads requires a new major version. Consumers must opt in explicitly.
- **Documentation-only changes → patch bump.** Clarifying descriptions without changing validation behaviour.
- Every schema change needs a note in `docs/CHANGELOG.md` as required by `context.md` §13 and `docs/conventions.md`. Contract tasks may not edit the changelog directly; the lead records the note at review/commit time.
- `additionalProperties: false` keeps payloads explicit so that typos are caught at ingest. Opaque maps that must accept project-specific fields (`parser.output`, `outcome.final_record`) are the deliberate exception and are documented in the schema.

## Turn event (P0-02, version 1.0.0)

[`schemas/turn_event.schema.json`](../schemas/turn_event.schema.json) describes one guest utterance plus the agent's processing of it.

### Incremental adoption

Only five fields are required at ingest:

| Required field | Rule |
|---|---|
| `event_id` | UUID string; idempotency key |
| `call_id` | Non-empty string; late outcomes are merged by this value |
| `turn_index` | Integer >= 0 |
| `audio` | Object with required `uri`; `sample_rate_hz`, `channels`, `duration_ms` optional |
| `stt` | Object with required `text` (may be `""`); model ids, `words`, `decoder_stats` optional |

Everything else is optional at the top level: `ts` (RFC 3339), `consent` (`recording` and `training` together), `parser`, `dialogue_context`, and `outcome`. When a nested block is present, its semantically essential fields are required: `words` items need all four fields (`w`, `start_ms`, `end_ms`, `conf`), `decoder_stats` needs all three values, and `outcome` needs `status`. Optional blocks whose parts can arrive independently (`parser`, `dialogue_context`) have no required nested fields.

This lets an integration start with `log_turn(event_id, call_id, turn_index, audio, stt)` and add consent, parser, context and outcome data later. `outcome` may be merged by `call_id` after the call ends.

`consent` is optional in the *transport* schema for incremental adoption; it is **not** optional for dataset eligibility. Datasets must still check consent flags and PII (`context.md` §2 and §12).

### Examples

| File | Purpose |
|---|---|
| `turn_event.valid.minimal.json` | Five required fields only |
| `turn_event.valid.full.json` | Every optional block populated |
| `turn_event.invalid.missing_required.json` | Missing `stt` |
| `turn_event.invalid.bad_status.json` | `outcome.status` outside the enum |

## Booking slots (P0-03, version 1.0.0)

[`schemas/booking.schema.json`](../schemas/booking.schema.json) is the reference hotel-booking slot contract used by the parser, review form and evaluation gate.

| Slot | Type | Rule |
|---|---|---|
| `check_in` | string | ISO date (`YYYY-MM-DD`) |
| `check_out` | string | ISO date; must be after `check_in` |
| `guests` | integer | >= 1 |
| `room_type` | string | `standard`, `deluxe` or `suite` |
| `guest_name` | string | Non-empty |
| `contact` | object | Optional; `email` (format email) and `phone` each optional |

The first five slots are required; `contact` is optional. Unknown top-level or `contact` fields are rejected by the example schema.

### Cross-field validators

JSON Schema draft 2020-12 cannot compare two properties, so [`schemas/booking.py`](../schemas/booking.py) enforces these rules in Python and raises `BookingValidationError` with a specific message:

- `check_out` must be strictly after `check_in` (equal dates are invalid).
- `room_type` must exist in the active schema's enum; unknown values are rejected.
- `guests` must be >= 1 (the minimum is read from the active schema).
- Missing fields listed in the active schema's `required` are reported together.
- `contact.email` must look like an email address; both the Python check and the schema's `format: email` check apply.

`validate_booking(record, schema_path=None)` accepts a per-project schema override. Precedence: explicit argument, then `VOICELOOP_BOOKING_SCHEMA`, then the bundled example schema. Relative paths resolve against the repository root; absolute paths are used as-is. Enum values and required fields are read from the active schema, so a project can change its room types or slots without changing the validator. Cross-field checks are keyed by the example field names and are skipped when those fields are absent from the substituted schema's records.

Examples: `booking.valid.json` plus `booking.invalid.missing_required.json`, `booking.invalid.dates.json`, `booking.invalid.room_type.json`, `booking.invalid.guests.json`, and `booking.invalid.email.json`.

## Using the helpers

`schemas/__init__.py` provides small self-contained helpers that use `jsonschema` directly and resolve names relative to the repository root:

```python
from schemas import load_schema, schema_path, validate
from schemas.booking import validate_booking
from schemas.turn_event import validate_turn_event

validate_turn_event(event)  # raises jsonschema.ValidationError with path and schema name
validate(record, "booking.schema.json")
schema = load_schema("turn_event.schema.json")
path = schema_path("examples/turn_event.valid.full.json")
```

Format checks (`uuid`, `date`, `date-time`, `email`) are enabled in both validators. The helpers are intentionally local to `schemas/` for these contract tasks; `voiceloop_common` may re-export them later without changing callers.

Unit tests: `tests/unit/schemas/`. Run `.\\.venv\\Scripts\\python.exe -m pytest tests/unit/schemas -q` from the repo root.
