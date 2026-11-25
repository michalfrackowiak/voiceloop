# Consent and retention controls (P1-09)

Read-only helpers for consent eligibility and raw-audio retention policy. This
package does not change ingest APIs or storage (`services/ingest/` root belongs
to P1-01/P1-02), and it never stores audio in Postgres (`context.md` §12).

## Run and test

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/governance -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy .
```

## Contract notes

- `consent.py:is_eligible(event) -> bool` reads `turn_event.schema.json` 1.0.0
  structures read-only and returns `True` only when both `consent.recording`
  and `consent.training` are boolean `True`. The flags are optional at ingest
  for incremental adoption but mandatory at dataset eligibility
  (`docs/schemas.md` §"Incremental adoption"). Missing or malformed flags fail
  closed. Auditable reasons are produced by `services/datasets/filters`.
- `retention.py` exposes typed `RetentionSettings` with prefix
  `VOICELOOP_RETENTION_`, a `RetentionPolicy` interface, a metadata-only
  `RawAudioRecord`, and `is_deletion_eligible()` using
  `voiceloop_common.utcnow()`. Deletion eligibility, not deletion execution,
  is provided here; P3-10 owns the retention job.
- **Policy status: OPEN — counsel review required.** The numeric
  `ENGINEERING_DEFAULT_MAX_RAW_AUDIO_AGE_DAYS` is a clearly marked development
  placeholder, not a legal decision. The human data protection policy owner
  (`docs/compliance.md` §5, section 6) must set it. No production retention
  conclusion is implied.

## Known gaps

- No audio redaction or pseudonymisation here (P3-10).
- No retention deletion job here (P3-10); this package only answers whether a
  metadata-only audio reference is eligible under the configured control.
