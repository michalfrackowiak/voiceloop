# Audio redaction and pseudonymisation (P3-10) — SYNTHETIC / FAKE STAND-IN

> **LABEL: SYNTHETIC/FAKE STAND-IN — no real audio, store, network, speech recognition, or deletion.**
> Everything here is a deterministic rehearsal using a fake waveform and fake
> in-memory stores. Retention windows, padding and the pseudonym salt are
> **engineering placeholders, NOT legal or compliance decisions**.
> `docs/compliance.md` remains **OPEN — counsel review required**.

## Scope

This package implements three governance subtasks:

- **P3-10.1** Timestamp-based audio redaction for detected PII spans.
- **P3-10.2** Name pseudonymisation in training transcripts.
- **P3-10.3** Retention job deleting expired **raw** audio (fake store only).

It consumes the committed P1-09 `detect_pii` / consent APIs **read-only**. It
does not reimplement card detection: card findings are a **hard drop**, never a
pseudonym. No other lane's paths are modified.

## Run

From the repository root, with the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m services.datasets.redaction.cli redact `
  --output <temp-report-dir>
.\.venv\Scripts\python.exe -m services.datasets.redaction.cli retention `
  --assets <metadata-only-assets.json> --now 2026-03-01T00:00:00+00:00 `
  --output <temp-report-dir>
```

`redact` reads the read-only labels manifest `tests/fixtures/turns.jsonl` and
the read-only event JSON files, **gates every event on consent using the
read-only P1-09 `filter_by_consent()` API before any redaction**, then generates
a deterministic fake waveform per eligible turn, zeroes every supplied
timestamp span with padding, writes PII-free transcripts and audit events, and
prints a JSON summary. Consent-denied rows produce a PII-free record in
`consent_drops` inside the JSON report (with `training_consent_denied` and the
source audio URI) and never appear in `redacted_transcripts.jsonl`,
`hard_drops.jsonl`, or `audit_events.jsonl`, and never receive a
`fake-redacted://` URI. On the committed fixtures the split is 36 processed /
35 redacted / 1 consent-dropped. `retention` reads a
metadata-only asset JSON array and prints/writes an auditable report; it never
deletes redacted or training-derived assets and makes **no real deletion
claim**. Both commands are offline and accept injected settings/clock/audit/
store seams in tests.

Example retention asset record:

```json
[
  {
    "uri": "fake://raw/call-0001",
    "kind": "raw_audio",
    "captured_at": "2026-01-01T00:00:00+00:00",
    "source_call_id": "call-0001"
  }
]
```

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/datasets_redaction -q
.\.venv\Scripts\python.exe -m ruff check services/datasets/redaction tests/unit/datasets_redaction
.\.venv\Scripts\python.exe -m ruff format --check services/datasets/redaction tests/unit/datasets_redaction
.\.venv\Scripts\python.exe -m mypy services/datasets/redaction tests/unit/datasets_redaction
```

## Public API

Import from `services.datasets.redaction` (all names are synthetic/fake):

- `__init__.py` re-exports the public types and functions; module-level
  constants `POLICY_STATUS`, `COMPLIANCE_STATUS`, `SYNTHETIC_LABEL`,
  `ENGINEERING_DEFAULT_PADDING_MS`, `ENGINEERING_DEFAULT_RETENTION_DAYS`
  carry the placeholder status labels.
- `settings.py`: `RedactionSettings` (typed, `VOICELOOP_REDACTION_` env prefix),
  `Clock`, `SystemClock`, `FixedClock`, `AuditEvent`, `AuditSink`,
  `InMemoryAuditSink`, and the engineering defaults.
- `spans.py`: `CharSpan`, `WordTiming`, `TimestampSpan`, `SpanCategory`,
  `RedactionError`/`InvalidSpanError`/`UnmappedSpanError`/
  `UnverifiedRedactionError`/`CardNumberRejected`, `validate_span`,
  `merge_spans`, `find_known_name_spans`, `expand_cued_name_spans`,
  `resolve_char_spans`, `word_timings_from_text`, `PiiSpanDetector`,
  `P109WordSpanDetector`, `TranscriptPiiDetector`, `P109TranscriptDetector`.
- `audio.py`: `FakeAudio`, `AudioStore`, `InMemoryAudioStore`,
  `AudioRedactionResult`, `assert_no_unredacted_span`, `redact_audio`,
  `redact_audio_uri`.
- `text.py`: `TranscriptRecord`, `TranscriptSource`,
  `InMemoryTranscriptSource`, `TextReplacement`, `PseudonymisedText`,
  `make_pseudonym`, `pseudonymise_text`, `pseudonymise_transcript`.
- `retention.py`: `AssetKind`, `RetainedAsset`, `RetentionStore`,
  `InMemoryRetentionStore`, `DeletionDecision`, `RetentionReport`,
  `run_retention`, `asset_from_dict`, and the stable `REASON_*` labels.
- `cli.py`: `main` plus `python -m services.datasets.redaction.cli`.

### Contracts and invariants

- **Audio**: `FakeAudio.synthetic(uri, sample_rate_hz, duration_ms, seed=...)`
  generates a signed 16-bit fake waveform with an explicit sample rate (the
  native 8 kHz fixture rate is never resampled). `redact_audio` validates every
  supplied span, applies non-negative padding on both sides, merges
  overlapping/adjacent spans deterministically, and zeroes the corresponding
  samples. `assert_no_unredacted_span` verifies the fake envelope of both the
  supplied and merged spans; any non-zero peak raises
  `UnverifiedRedactionError`. Negative, reversed, beyond-duration spans and
  malformed durations fail closed with structured codes.
- **Cued-name safety net**: `expand_cued_name_spans` is shared by
  `resolve_char_spans()` (text) and `P109WordSpanDetector.detect_spans()`
  (audio). P1-09's non-overlapping `name is <token>` branch reports only
  `"Mr"` for `"The name is Mr. Smith."`; P3-10 extends a bare-title finding to
  the following capitalized token(s) so the default API (no `known_names`)
  pseudonymises and zeroes the full name. A bare title with no capitalized
  continuation raises `UnverifiedRedactionError` (`unverified_cued_name`)
  rather than leaving a possible surname unscreened.
- **Consent gate**: the CLI calls read-only P1-09 `filter_by_consent()` on each
  event before redaction. Non-consented events are recorded in
  `redaction_report.json` under `consent_drops` and never become training text
  or a `fake-redacted://` audio reference (§2 #8).
- **Text**: `pseudonymise_text` screens with read-only P1-09 `detect_pii`.
  Names become stable per-call salted pseudonyms via SHA-256 of
  `salt + call_id + casefolded matched name` (12 hex chars, e.g.
  `[PERSON_1a2b3c4d5e6f]`); the same name in one call is stable across turns,
  different calls never collide under normal hashing, and replay is byte
  identical. Email, phone and other non-card findings become category masks.
  Non-PII text is preserved exactly. Card findings raise `CardNumberRejected`.
  Output is re-screened and caller-supplied known names must not remain as
  whole words, otherwise `UnverifiedRedactionError` is raised. No matched PII
  value is stored in results, audit events or logs — only offsets, categories,
  counts, URIs and policy labels.
- **Retention**: `run_retention(store, settings=..., clock=..., audit=...)`
  deletes only `AssetKind.RAW_AUDIO` at or beyond the configured engineering
  window; redacted and training-derived assets are always kept with
  `not_raw_audio`. Future capture timestamps are kept. The run is idempotent
  and emits `raw_audio_fake_deleted` / `retention_run` audit events plus a
  report with deleted/kept counts, reason counts and an explicit
  `NO_REAL_DELETION_CLAIM` string.

## Real seams (documented, not imported)

A production integration would inject implementations of these protocols
without changing this package:

- `PiiSpanDetector` / `TranscriptPiiDetector`: replace the read-only P1-09
  adapter with a reviewed detector or real word-level timings; no heavy
  imports here.
- `AudioStore`: map `load_raw`/`save_redacted` to an object store with real
  audio decoding — this package stays codec-free.
- `TranscriptSource`: supply reviewed transcripts and known names.
- `RetentionStore`: implement `list_assets`/`delete_raw` against a real
  metadata catalog; legal approval must precede any real deletion.
- `Clock`, `AuditSink`: replace the deterministic in-memory fakes with shared
  infrastructure.

The fake `InMemoryAudioStore` and `InMemoryRetentionStore` never touch a real
object store, network, or filesystem outside the optional report output
directory.

## Known gaps

- P1-09 name detection is cue-based (`Mr|Mrs|Ms|Miss|Dr X`, `name is X`).
  The specific **cued title + dot case** (`"The name is Mr. Smith."`) is now
  covered by P3-10's shared expansion even when `known_names` is omitted; a
  bare title with no capitalized continuation fails closed with
  `unverified_cued_name`. Names with **no cue at all** and spelled-out phone
  numbers are still **not detected**; pass `known_names` from reviewed metadata
  to cover those in this package.
- Audio is a fake deterministic waveform, not speech: zeroing proves the
  envelope contract on synthetic data, not perceptual or codec-level
  redaction.
- Retention numbers are engineering defaults. Counsel review in
  `docs/compliance.md` is still **OPEN — counsel review required**; no legal
  retention period has been decided here.
- No conflict with `context.md` was found. Invariants §2 #8/#9 (consent/PII
  screening and no raw PII in outputs) are preserved read-only; §5, §8, §13,
  §14 fake-only and escalation rules are followed.
- No commits, installs, edits to other lanes, or real deletion were performed.
