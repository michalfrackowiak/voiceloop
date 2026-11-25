# Synthetic augmentation (P3-03)

`services/datasets/augment/` generates **deterministic synthetic variants** for
repeated STT failure classes without any audio library, TTS engine, codec,
network or GPU. Everything produced here is labelled:

> `SYNTHETIC/FIXTURE STAND-IN - NOT REAL AUDIO`

The built-in signals are integer carriers with seeded perturbations. They are
**rehearsal stand-ins**, not intelligible speech and not training-ready audio.
Only a JSON manifest is written; stand-in samples stay in memory and are never
serialised.

## Public API

| Module | Entry points |
|---|---|
| `settings` | `AugmentSettings` (`VOICELOOP_AUGMENT_` prefix), `parse_csv`, `STAND_IN_LABEL`, `EXPECTED_GOLDEN_CONTENT_HASH`, `EXPECTED_FIXTURE_SHA256`, `ARTIFACT_TYPE`, `AUGMENTATION_SCHEMA_VERSION`, `ALLOWED_TRANSFORMS` |
| `variants` | `VariantSpec`, `AudioStandIn`, `TtsBackend`, `DeterministicFakeTts`, `build_variant_specs` |
| `noise` | `NoiseProfile`, `TelephonyBand`, `add_line_noise`, `apply_telephony_band` |
| `manifest` | `AugmentedVariant`, `TransformRecord`, `SourceHashes`, `snapshot_source_hashes`, `assert_source_hashes_unchanged`, `load_dataset_manifest`, `build_manifest`, `validate_manifest`, `write_manifest` |
| `cli` | `AugmentationRun`, `build_augmentation`, `main` |

All symbols above are re-exported from `services.datasets.augment`.

`build_augmentation()` accepts injectable `tts`, `dataset_manifest`,
`golden_call_ids`, `regression_call_ids` and `created_at` seams. The CLI uses the
same orchestration function.

## Run

Create a read-only P2-05 manifest first using the existing
`services.datasets` API, then run the augmentation CLI from the repository root:

```powershell
.\.venv\Scripts\python.exe -m services.datasets.augment.cli `
  --manifest artifacts/datasets/p2-05-augment-source.json `
  --output reports/datasets/augment/manifest.json `
  --golden-ids call-0001,call-0002 `
  --regression-ids call-0003 `
  --seed 17 --duration-ms 240 --noise-level 8
```

Omitting `--golden-ids` loads the sealed P1-06 golden call IDs read-only via
`load_golden_call_ids()`. Because the synthetic fixture pool is entirely sealed,
that default produces an empty batch by design; pass explicit injected IDs for
fixture rehearsals only. `--golden-ids ""` means an empty injected golden list
and is intended only for rehearsal.

On success the CLI prints a PII-free summary, writes exactly one manifest to
`--output`, and re-hashes the golden manifest and fixture pool. On a known
config/validation error it exits 2 and prints a structured
`{"code": ..., "message": ...}` object to stderr; no file is written.

Settings are typed and read `VOICELOOP_AUGMENT_` variables, e.g.
`VOICELOOP_AUGMENT_SEED`, `VOICELOOP_AUGMENT_DURATION_MS`,
`VOICELOOP_AUGMENT_NOISE_LEVEL`, `VOICELOOP_AUGMENT_TELEPHONY_HIGH_HZ`.
No secrets or real guest data are stored here.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/datasets_augment -q
.\.venv\Scripts\python.exe -m ruff check services/datasets/augment tests/unit/datasets_augment
.\.venv\Scripts\python.exe -m ruff format --check services/datasets/augment tests/unit/datasets_augment
.\.venv\Scripts\python.exe -m mypy services/datasets/augment tests/unit/datasets_augment
```

## Contract notes

- **Input contract:** `schemas/dataset_manifest.schema.json` **1.0.0**, consumed
  read-only and validated with `voiceloop_common.validate(..., "dataset_manifest")`.
  The input P2-05 manifest is never edited.
- **Output contract:** internal artifact `voiceloop.augmentation.manifest`,
  version `internal-1.0.0`. This is deliberately **not** a 1.0.0 dataset manifest:
  that schema has `additionalProperties: false` and cannot carry transform names,
  params or exclusion proof. No schema change was made.
- Every manifest and every variant record carries `synthetic: true` plus the full
  stand-in label. `evaluation_set_exclusion` records the injected golden and
  regression call IDs and asserts zero synthetic variants in either set.
- Each variant records `source_call_id` (pseudonymous), a token digest, ordered
  transform records (`fake_tts`, then optional `line_noise` and
  `telephony_band`) with scalar params, and audio metadata (digest, sample count,
  rate, duration, label). Raw samples/audio bytes are rejected by
  `validate_manifest()`.
- `content_hash` is SHA-256 over `canonical_json()` of the manifest without the
  hash. Sorted variant IDs and calls make the digest independent of batch order.
- Before and after generation the CLI verifies the sealed golden
  `content_hash` (`662cdaf4…4d03da8e`) and fixture `turns.jsonl` SHA-256
  (`aca23a30…a2ff2131bc2e0a4453fd`). A mismatch fails closed.
- Manifest strings are screened with `services.datasets.filters.detect_pii`;
  names, emails, phone numbers and card-like digit runs are rejected. Logs carry
  only counts, transform names and truncated hashes.
- Splits remain by call. Eligible calls are `source_call_ids - golden - regression`;
  `build_manifest()` and `validate_manifest()` independently reject any leaked
  evaluation call.
- `promotion_decision` is always `not_evaluated`. No threshold, gate, MLflow
  alias, dataset root module, sealed golden artifact or fixture file is changed.

## Fake seams and real seams

- **TTS seam:** implement `TtsBackend.synthesize(token, sample_rate_hz,
  duration_ms, seed)` and `TtsBackend.params(spec)`, then pass it as `tts=` to
  `build_augmentation()`. The default `DeterministicFakeTts` uses a seeded tone
  carrier and local `random.Random`; it emits only a token digest in params.
  A real TTS adapter is intentionally **not** implemented here: it would need a
  new dependency/network policy and human approval before any generated audio
  could be used as a training label (context.md §2 invariant 2).
- **Noise/telephony seam:** `NoiseProfile` and `TelephonyBand` are injectable
  parameter objects; `add_line_noise()` and `apply_telephony_band()` are integer
  stand-ins, not DSP filters or codecs. A real audio adapter can replace those
  functions behind the same transform names, but must preserve the native
  8 kHz sample rate and sample count (context.md §16) and must never serialise
  bytes into this manifest.
- **Storage seam:** `write_manifest()` refuses destinations under
  `services/datasets/golden/artifacts`, `tests/fixtures` or the supplied P2-05
  input path. The only write is the selected JSON output path.

## Known gaps

- P3-01 (Whisper trainer) and CP-2 are still pending, so this lane is implemented
  in parallel against the P2-05 manifest contract only. No trainer is invoked.
- Real TTS, real telephony simulation and audio artifacts are out of scope for
  this fake-only task.
