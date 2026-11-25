# Conversion and artifact safety tests (P3-04)

> [!WARNING]
> **SYNTHETIC/FIXTURE STAND-IN — NOT A DEPLOYED ARTIFACT TEST.** The built-in
> artifact is a content-addressed descriptor for a deterministic fake decode
> behaviour. No CTranslate2/ONNX runtime, model download, GPU, network or real
> audio is imported or used. The sealed v1 golden stand-in and
> `tests/fixtures/turns.jsonl` are read-only. No promotion or threshold change is
> performed: the P2-09 gate is called through its **read-only** `Gate.evaluate`
> seam (`context.md` §2 invariants #2/#3/#5, §9 and §14).

This package proves that the gate evaluates the **exact deployable artifact** and
records the **exact decoding settings**, then catches injected empty/truncated
output and reports silence hallucination and multilingual-forgetting evidence in
every STT report.

## Layout

| File | Responsibility |
|---|---|
| `settings.py` | Typed `VOICELOOP_ARTIFACT_` settings and committed hash pins |
| `artifact.py` | Immutable, content-addressed fake artifact and write-once store |
| `decoding.py` | Exact decoding hash and hotword/prompt/previous-text matrix |
| `probes.py` | Read-only fixture loading, failure injection and safety diagnostics |
| `report.py` | Contract-valid STT report, read-only gate seam and validation |
| `cli.py` | One-command CLI with structured, fail-closed errors |
| `__init__.py` | Public re-exports |

## Run

From the repository root (`.venv\Scripts\python.exe` is this environment's
`python`):

```powershell
.\.venv\Scripts\python.exe -m services.evalgate.artifact.cli --behaviour healthy --output reports/evalgate/artifact/stt_report.json
.\.venv\Scripts\python.exe -m services.evalgate.artifact.cli --behaviour empty --iterations 100 --output reports/evalgate/artifact/empty_report.json
.\.venv\Scripts\python.exe -m services.evalgate.artifact.cli --behaviour forgetting --served-languages en,es --output reports/evalgate/artifact/multilingual_report.json
```

Useful flags: `--fixtures`, `--artifact-store`, `--checkpoint-uri`,
`--runtime-label`, `--hotwords`, `--initial-prompt`,
`--condition-on-previous-text` / `--no-condition-on-previous-text`, `--language`,
`--level`, `--iterations`, `--seed`. Behaviours are `healthy`, `empty`,
`truncated`, `hallucinating` and `forgetting`.

The CLI prints the artifact and decoding hashes, both mandatory diagnostic
metrics, the primary metric rule and the synthetic limitation. Exit code is `0`
only when every probe passes; a known failure returns `2` with a single JSON
error on stderr. The empty-output case still writes its evidence report, then
returns `2` with `code: "artifact_probe_failed"`.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/evalgate_artifact -q
.\.venv\Scripts\python.exe -m ruff check services/evalgate/artifact tests/unit/evalgate_artifact
.\.venv\Scripts\python.exe -m ruff format --check services/evalgate/artifact tests/unit/evalgate_artifact
.\.venv\Scripts\python.exe -m mypy services/evalgate/artifact tests/unit/evalgate_artifact
```

Tests cover hash immutability and store tamper detection, all eight decoding
combinations, pinned fixture/golden hashes, every injected behaviour,
empty/truncated detection, silence hallucination, multilingual surrogates,
contract validation, exact-URI enforcement, no transcript/audio/PII leakage and
the CLI round-trip.

## Public API

```python
from services.evalgate.artifact import (
    ArtifactSettings,
    ExactArtifactConverter,
    HashedReportEvaluator,
    create_fake_artifact,
    decode_combinations,
    evaluate_artifact,
    evaluate_checkpoint_uris_do_not_change_result,
    run_exact_artifact_gate,
    validate_stt_report,
    write_stt_report,
)
```

- `create_fake_artifact(behaviour, runtime_label)` — frozen, content-addressed
  descriptor; the URI embeds the SHA-256 of its canonical bytes.
- `ReadOnlyArtifactStore.put/load` — write-once descriptor persistence; reload
  fails if bytes no longer match the address.
- `DecodingSettings.to_report_fields()` / `decoding_hash_from_fields()` — exact
  hotwords, prompt, previous-text flag and language, plus their hash.
- `decode_combinations(settings)` — the eight deterministic switch combinations;
  `language_combinations(...)` clones them for configured non-English languages.
- `load_probe_inputs(settings)` / `run_probe_suite(...)` / `fake_transcribe(...)`
  — read-only synthetic probe seams.
- `hallucination_summary(...)` / `multilingual_summary(...)` — mandatory
  diagnostics, never promotion metrics or gate evidence.
- `evaluate_artifact(settings, artifact)` — scores fixtures, pairs them with a
  healthy fake champion, runs the exact-artifact gate read-only and returns a
  validated `ArtifactEvaluation`.
- `ExactArtifactConverter` / `HashedReportEvaluator` / `run_exact_artifact_gate`
  — P2-09 seam adapters that refuse checkpoints and mismatched URIs.
- `validate_stt_report(report)` / `write_stt_report(report, path)` — enforced
  extension and schema validation before any file write.
- `services.evalgate.artifact.cli.main(argv)` — CLI entry point.

### Fake seams

The P2-09 protocols are consumed, not modified:

- `ArtifactConverter.convert(source_uri)` is implemented by
  `ExactArtifactConverter`: a trainer/registry checkpoint URI is replaced by the
  content-addressed deployable artifact URI. Changing the checkpoint URI cannot
  change what the gate evaluates.
- `ArtifactEvaluator.evaluate(artifact_uri, eval_sets)` is implemented by
  `HashedReportEvaluator`: it accepts only the exact hashed artifact URI and
  revalidates the report.
- `fake_transcribe(...)` is the deterministic decode seam for the five injected
  behaviours. It returns only transcripts/slots in memory; probe artefacts keep
  counts and error codes, never text.

### Real-runtime seam (documentation only)

A future CTranslate2/ONNX adapter can implement the same converter/evaluator
protocols against the **converted/quantised artifact**, pass the exact
hotwords, initial prompt, `condition_on_previous_text` and language recorded in
`decoding_settings`, and read `row["audio_path"]` at its native sample rate.
That adapter belongs in its own runtime task with optional heavy dependencies;
this package neither imports nor installs it. Real multilingual evidence needs a
committed, human-reviewed multilingual set before it can count as gate evidence.

## Artifact contract

The generated JSON is a full `eval_report` **1.0.0** payload built by the P1-07
writer and validated with `voiceloop_common.validate(report, "eval_report")`.
`schemas/eval_report.schema.json` closes top-level properties, so the internal
P3-04 extension (`extension_version: "internal-1.0.0"`) is nested under the
existing optional `decoding_settings` object. It never removes, renames or
weakens a contract field and contains no new top-level keys. It records:

- `artifact` — evaluated SHA-256/URI, runtime label, behaviour, conversion
  recipe, source checkpoint URI and explicit `checkpoint_uri_evaluated: false`.
- `decoding` — exact settings hash and the deterministic combination matrix.
- `probes` — counts, error codes and bounded structured details; no transcripts.
- `hallucination` — `rate`, firing case IDs, paired delta CI and
  `promotion_metric: false`, measured on committed empty-reference
  silence/noise fixture turns.
- `multilingual_forgetting` — served languages, surrogate rate, CI and explicit
  `human_labels: false` / `gate_evidence: false`. With the roadmap default of
  English only (`context.md` §14 open question), this is
  `not_evaluated_no_committed_multilingual_fixtures` with zero gate-visible
  `other_languages.turns`.
- `provenance` — committed fixture SHA-256, verified golden seal/hash and
  explicit false flags for training/tuning and real audio/human labels.
- `gate` — the read-only P2-09 decision for the exact artifact URI.

`metrics.silence_hallucination_rate` and
`metrics.multilingual_forgetting_rate` are present in **every** generated STT
report, and the matching `slice_metrics.silence.hallucination_rate` and
`slice_metrics.other_languages.forgetting_rate` are present in every report's
slice section. Both `report` and `champion_report` returned by
`evaluate_artifact` pass `validate_stt_report` and carry the validated
extension; the champion is the healthy fake baseline and is never a promotion
candidate. `confidence_intervals` keep P1-07 semantics: paired
`challenger − champion` deltas over shared fixture turns. The only promotion
metric remains `e2e_booking_accuracy` from audio; WER alone never promotes.

## Settings (`VOICELOOP_ARTIFACT_`)

Every field is typed and environment-backed (`.env` supported). Key variables:
`FIXTURES_PATH`, `GOLDEN_ARTIFACT_DIR`, `GOLDEN_VERSION`,
`EXPECTED_FIXTURE_SHA256`, `EXPECTED_GOLDEN_CONTENT_HASH`, `OUTPUT_PATH`,
`ARTIFACT_STORE_DIR`, `CHECKPOINT_URI`, `RUNTIME_LABEL`, `ARTIFACT_BEHAVIOUR`,
`HOTWORDS`, `INITIAL_PROMPT`, `CONDITION_ON_PREVIOUS_TEXT`, `LANGUAGE`,
`HALLUCINATION_TEXT`, `TRUNCATION_RATIO`, `SERVED_LANGUAGES_CSV`,
`OTHER_LANGUAGE_PATTERN`, `MAX_ERROR_DETAILS`, `BOOTSTRAP_LEVEL`,
`BOOTSTRAP_ITERATIONS` and `BOOTSTRAP_SEED`. Defaults pin the committed fixture
hash `aca23a303954d035245a97f78319df3694d63611b717a2ff2131bc2e0a4453fd` and
golden content hash
`662cdaf43ee2d5fde03faf5e69fde196114d4f312406d94c00bf53714d03da8e`; a changed
input is refused before evaluation.

## Known gaps

- All evidence is synthetic/fixture-based. No human-reviewed golden set, real
  audio, real converted artifact or production decoding runtime is available, so
  no promotion claim is made.
- The committed fixture pool is English-only. `served_languages_csv` can exercise
  a deterministic surrogate diagnostic, but it is explicitly not gate evidence
  and must not be presented as a multilingual evaluation.
- The hallucination metric uses fixture `audio_kind` and empty
  `expected_transcript` labels; no human listened to the carrier WAVs, so
  `context.md` §2 invariant #2 remains unverified here.
- P3-01 (trainer/conversion handoff) is pending; its dependency was relaxed for
  parallel implementation. The real-runtime seam above is documentation only.
- CP-2/preamble integration remains pending. No threshold, alias, schema,
  fixture or other lane's file was changed.

## Conflicts with `context.md`

None found. §2 invariants #2/#3/#4/#5/#7/#8/#9, §5, §9, §13 and §14 were treated
as constraints: the gate sees only the hashed deployable artifact, WER is never
a promotion metric, committed labels/seals are read-only and hash-checked, CI
values are paired deltas, and the English-only open question is preserved rather
than silently answered.
