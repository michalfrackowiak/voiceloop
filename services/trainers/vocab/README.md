# Vocabulary and rules tier (P2-08)

Offline, deterministic tier-1 fixes for names, room types, numbers, dates and
STT phrase boosting, with no heavy ML/runtime dependencies. It reads only
committed synthetic fixtures and the sealed synthetic golden stand-in; it never
loads audio, an STT runtime, a parser model, a GPU, a network client or a live
decoder.

> [!WARNING]
> **SYNTHETIC/FIXTURE STAND-IN.** No real STT runtime decoded audio and no real
> decoder applied phrase boosting or LM fusion. `FakeSttRuntime` is an
> in-memory seam used to exercise the emitted config; all measured gains are
> fixture replay results. The sealed v1 golden artifact is a simulated-label
> workflow stand-in, not a human-reviewed production golden set. Nothing here
> authorises a promotion (`context.md` §2 invariants 2, 3, 4 and 5).

## Layout

| File | Responsibility |
|---|---|
| `settings.py` | Typed `VOICELOOP_VOCAB_*` settings; expected golden hash guard. |
| `aliases.py` | Name/room-type alias tables derived with fixture provenance. |
| `normalise.py` | Spoken number, stay-duration and itinerary-date rules. |
| `boosting.py` | Fixture-derived JSON boost config + injectable STT seam/fake. |
| `rules.py` | Pipeline orchestration, harness-based regression check, CLI. |
| `__init__.py` | Public API re-exports. |

## Run and test

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m services.trainers.vocab.rules --report <tmp>\p2-08-golden.json --config-output <tmp>\p2-08-boost.json
.\.venv\Scripts\python.exe -m pytest tests/unit/trainers_vocab -q
.\.venv\Scripts\python.exe -m ruff check services/trainers/vocab tests/unit/trainers_vocab
.\.venv\Scripts\python.exe -m ruff format --check services/trainers/vocab tests/unit/trainers_vocab
.\.venv\Scripts\python.exe -m mypy services/trainers/vocab tests/unit/trainers_vocab
```

The CLI prints a JSON regression report and writes optional JSON artifacts to
the paths supplied on the command line. It never writes to fixtures, golden
artifacts or docs. `--report`/`--config-output` are optional; no repo file is
written by default. `-m` may emit a harmless `RuntimeWarning` about the package
already importing `rules`; output is unaffected.

## Public API

```python
from services.trainers.vocab import (
    VocabSettings,
    VocabRuleSet,
    derive_alias_table,
    generate_boosting_config,
    write_boosting_config,
    check_golden_regression,
)

settings = VocabSettings()
rules = VocabRuleSet(settings)
outcome = rules.apply(stt_text, parsed_slots, words=word_confidences)
config = generate_boosting_config(settings)
write_boosting_config(config, "boosting-config.json")
report = check_golden_regression(settings, rules)
```

`RuleOutcome` carries the transformed `transcript`, `slots`, the fired `rules`
and a `changed` flag. `SttRuntime.transcribe(transcript, words, config)` is the
injectable seam; pass a real adapter to `VocabRuleSet(runtime=...)`, or call
`apply_boosting(..., runtime=...)`. `AliasTable`, `BoostConfig`, `Hotword`,
`Substitution`, `FakeSttRuntime`, `normalise_slots` and `normalise_transcript`
are also exported. All settings use prefix `VOICELOOP_VOCAB_`, for example
`VOICELOOP_VOCAB_MAX_HOTWORDS`, `VOICELOOP_VOCAB_ENABLE_BOOSTING` and
`VOICELOOP_VOCAB_NAME_ALIAS_OVERRIDES` (JSON object).

## Fixture derivation (no invented data)

- `misheard_surname` (`call-0002:2`): read-only event STT says "Mr. Smythe";
  the fixture label and its simulated second review expect "Mr. Smith". The
  token diff yields `smythe → Smith`, used both as a name alias and as the
  canonical hotword in the boosting config.
- Room types: every canonical value (`standard`, `deluxe`, `suite`) comes from
  `expected_slots.room_type`. Surfaces `standard room` and `deluxe room` come
  from those turns' `expected_transcript` strings. No other room vocabulary is
  fabricated.
- `date_confusion` (`call-0002:3`) and `nights_misparsed` (`call-0005:3`):
  spoken nights in read-only STT transcripts disagree with `check_out`; rules
  recompute `check_out = check_in + N` and set `guests` from "two guests".
- `schema_invalid_parse` (`call-0002:4`): transcript says "Arrive on the tenth
  and leave on the fourteenth" while the parse reverses the two ISO dates;
  the date rule restores their order.
- `misheard_guests` (`call-0004:3`): 8 kHz event marks "two" at confidence
  0.41; the label expects "four guests". That pair becomes a narrow
  low-confidence-only substitution in the synthetic runtime config.

The sealed golden stand-in is **not** used to derive or tune any rule. It is
used only as read-only gold in the regression check, via the committed P1-07
harness primitives (`evaluate_rows`, `predict`, `evaluate_turn`,
`aggregate_metrics`). `verify_seal` is called without actor/purpose/task so
the audit log is not appended to. All five artifact hashes are compared before
and after.

## Contract

- Input: a dict conforming to `schemas/turn_event.schema.json` 1.0.0 supplies
  `stt.text`, word confidences and the parser output; fixture labels supply
  read-only gold. No schema file was changed.
- Boost config: internal `voiceloop.vocab-boost.v1` JSON with `runtime`,
  `language`, `hotwords`, `substitutions`, decoding settings, `provenance`,
  `synthetic: true` and an explicit warning. This is a task-local runtime-seam
  contract, not a `schemas/` cross-service contract; no version bump was due.
- The P1-07 `eval_report` 1.0.0 contract is consumed read-only by the
  regression comparison; no report is written as a baseline or gate decision.
- Metrics are direction-aware: WER and entity error must not rise; slot F1,
  schema-valid rate and **E2E booking accuracy from audio** must not fall.
  WER alone never justifies promotion (§2 invariant 4).
- Room aliases (derived and `VOICELOOP_VOCAB_ROOM_ALIAS_OVERRIDES`) are checked
  case-insensitively against the committed `booking.schema.json` 1.0.0
  `room_type` enum; an invalid canonical raises `ValidationError`. Invalid dates
  or date arithmetic outside `datetime`'s range leave slots unchanged.

## Known gaps

- The WAV fixtures are tone/noise/silence carriers. The fakes replay recorded
  hypotheses; they do not decode audio. Real CTranslate2/ONNX hotword and
  initial-prompt behaviour (including empty/truncated output on converted
  models) must be validated in P3-04 against the deployed artifact.
- Room-type aliases currently cover only phrases observed in fixtures. Real
  review data may need audited alias additions; no automatic learning from
  the golden set is implemented.
- `normalise_transcript()` is available but disabled by default: gold fixture
  transcripts keep word forms, so rewriting them would worsen WER without
  improving booking fields. Slot-level number/date rules are enabled.
- The synthetic boosting substitution is deliberately gated on low word
  confidence; a production runtime must implement its own phrase boosting and
  re-run the harness before any gate decision.
- One-step rollback, promotion thresholds and alias flips remain outside this
  task; no gate or release service was touched.
