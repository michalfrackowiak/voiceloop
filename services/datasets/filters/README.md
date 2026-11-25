# Dataset eligibility filters (P1-09)

Consent and transcript-PII screening for dataset rows. Consumers are dataset
builders (P2-05 onward); the input is the read-only turn event contract
`schemas/turn_event.schema.json` 1.0.0. No schema is changed here.

## API

- `filter_by_consent(events) -> (kept, dropped_with_reasons)` excludes a row
  unless both `consent.recording` and `consent.training` are boolean `True`.
- `filter_eligible_events(events) -> (kept, dropped_with_reasons)` adds
  transcript PII screening. A card-number match is a **hard drop**, never a
  redaction into training data. Other findings (email, phone, person-name-like
  tokens) are detected but left for P3-10 redaction/pseudonymisation.
- `detect_pii(text) -> (PiiFinding, ...)` returns category plus half-open
  character offsets and no matched values. `PiiFinding.as_dict()` gives a
  JSON-safe audit shape.
- `DroppedEvent` carries `event_id`, `call_id`, `turn_index`, `audio_uri`, all
  `reasons`, and PII offsets. It never carries transcript text or matched PII,
  so drop records are safe to emit in manifests/filter counts and audit logs.

Card detection covers any run with at least 13 digits and repeated unicode
whitespace, hyphen/dash, dot, slash or underscore separators, including long
contiguous runs, regardless of grouping, adjacent letters or an earlier phone
number. A dedicated position-by-position scan makes the hard-drop decision
independent of other detectors' overlap resolution; false-positive drops are
deliberately preferred to misses (a date-like 13+ digit run may be dropped).
Email, phone and title/“name is” name patterns are deterministic regexes;
spelled-out phone numbers and names without a cue are a known detection gap.

## Run and test

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/governance -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy .
```

Tests read `tests/fixtures/events/` read-only and construct in-memory variants
with synthetic test card numbers. No real guest data is used; no fixture,
schema, or ingest/dataset root file is modified by this package.

## Contract notes

- Consent flags are optional in the transport schema but required for dataset
  eligibility (`docs/schemas.md`). This filter is the enforcement point.
- `filter_counts.consent_dropped` and `filter_counts.pii_dropped` in
  `schemas/dataset_manifest.schema.json` 1.0.0 can be derived from the returned
  drop records by reason. No manifest contract change is required.
- Screening currently applies to `stt.text` only. A dataset builder must call
  `detect_pii()` directly on any reviewed/corrected transcript it introduces
  (P2-05/P3-10).
