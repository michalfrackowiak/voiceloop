# Synthetic fixtures (P0-05)

This directory contains **entirely synthetic** VoiceLoop fixtures for integration
tests: 36 labelled turns across 6 synthetic calls, with 36 short mono WAV files.
The WAVs contain deterministic tone carriers, band-limited line hiss/crackle, or
exact digital silence. **They contain no intelligible speech, no real guest
audio, and no real guest data.** Transcripts and booking slots are invented
labels used to exercise capture → triage → review → dataset → eval plumbing;
they are not recordings of those words.

These fixtures are **not** the sealed golden set (`context.md` §2.5) and are not
a substitute for consented, human-reviewed production traffic.

## Layout

| Path | Contents |
|---|---|
| `generate_fixtures.py` | Deterministic stdlib-only generator and CLI. |
| `turns.jsonl` | One label record per turn (the labels manifest). |
| `summary.json` | Counts, call IDs, sample rates, failure types, named case IDs. |
| `audio/<call_id>/turn-<nnn>.wav` | 16-bit mono PCM, 240–400 ms, 16 kHz or native 8 kHz. |
| `events/<call_id>/turn-<nnn>.json` | Turn events conforming to `schemas/turn_event.schema.json` 1.0.0. |
| `contexts/<call_id>.jsonl` | Synthetic prior-turn hypothesis history; for turn *k*, lines with `turn_index < k` are prior context. |
| `test_synthetic_fixtures.py` | Pytest checks for schema validity, labels, audio, named failures, PII and determinism. |

## Regenerate

From the repository root:

```powershell
.\.venv\Scripts\python.exe tests\fixtures\generate_fixtures.py
```

Optional: write to a different directory, or plug in a future real TTS backend
(the committed fixtures are always generated with the deterministic backend):

```powershell
.\.venv\Scripts\python.exe tests\fixtures\generate_fixtures.py --output-dir C:\temp\voiceloop-fixtures
.\.venv\Scripts\python.exe tests\fixtures\generate_fixtures.py --tts-backend my_package.tts:MyBackend
```

A `tts_backend` must implement `TtsBackend.synthesize(text, sample_rate_hz,
duration_ms, seed) -> list[int]`, returning exactly
`sample_rate_hz * duration_ms // 1000` signed 16-bit samples at the **native**
sample rate (8 kHz is never resampled; `context.md` §16). Only speech-like
tone turns use the backend; silence and noise are generated locally. A custom
backend must still produce synthetic data only. Regeneration is deterministic:
same generator version → byte-identical WAVs, events, contexts and manifests.

## URI convention

All URIs in events and manifests are **repo-relative POSIX paths**, resolved as
`REPO_ROOT / uri`. They are canonical: regenerating into `--output-dir` does not
rewrite URIs, so payload bytes stay identical. Example:

```
tests/fixtures/audio/call-0004/turn-003.wav
tests/fixtures/events/call-0004/turn-003.json
tests/fixtures/contexts/call-0004.jsonl
```

There are no `file://` URIs because no absolute host path should be committed;
consumers that need one can prefix `file://` after resolving against the repo
root.

## Labels manifest

`turns.jsonl` is UTF-8 JSON Lines, one object per turn, sorted keys. Required
fields: `call_id`, `turn_index`, `event_path`, `audio_path`,
`expected_transcript`, `expected_slots`, `failure_type`, `consent`
(`recording`/`training`), `sample_rate_hz`, `synthetic: true`, `notes`.
Additional fields: `expected_outcome` (event `outcome.status` when present),
`audio_kind` (`tone`/`silence`/`noise`), `case_id` (named failure or `null`),
and `duration_ms` is carried by the event's `audio` block.

`expected_slots` is `{}` when the turn establishes no bookable slot (greetings,
silence, noise); otherwise it is a complete booking object in the shape of
`schemas/examples/booking.valid.json` and passes `schemas.booking.validate_booking`.
`failure_type` follows `context.md` §7 plus `clean`: `stt_wrong`, `parser_wrong`,
`both_fine`, `unusable_audio`, `clean` (`both_fine` means an adversarial-looking
turn whose hypothesis and parse were both verified correct; `clean` is an
ordinary turn).

### Named failure cases

| `case_id` | Expected outcome |
|---|---|
| `misheard_surname` | STT hypothesis "Mr. Smythe" for expected "Mr. Smith" (`stt_wrong`, low word confidence). |
| `date_confusion` | Correct STT "three nights and two guests" but parser attaches "three" to guests (`parser_wrong`). |
| `schema_invalid_parse` | Parser emits `check_out` before `check_in`, `schema_valid: false` (`parser_wrong`). |
| `silence_exact`, `silence_hallucination` | Exactly-zero PCM; one empty hypothesis, one hallucinated one. |
| `line_noise`, `line_noise_burst` | Deterministic hiss/crackle with no usable transcript. |
| `misheard_guests` | 8 kHz telephony: "four" misheard as "two". |
| `nights_misparsed`, `schema_invalid_missing_field` | Additional parser errors. |
| `no_training_consent` | `recording: true`, `training: false`: must be excluded from training datasets (`context.md` §2.8). |

`call-0004` is the native 8 kHz telephony call. All fixtures use invented names
(`Ada Fixture`, `Sam Sample`, …), `example.com` emails and reserved-range
`+1 555 010 xxxx` phone numbers. Card numbers do not appear anywhere.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pytest tests\fixtures -q
.\.venv\Scripts\python.exe -m ruff check tests\fixtures
.\.venv\Scripts\python.exe -m mypy tests\fixtures
```
