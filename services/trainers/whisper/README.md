# Whisper trainer (P3-01) — SYNTHETIC/FIXTURE STAND-IN

`services/trainers/whisper/` implements the P2-06 `TrainerPlugin` contract for a
deterministic Whisper LoRA **stand-in**. It performs no real training: no
`torch`, `transformers`, PEFT, CTranslate2, faster-whisper, GPU, network or
model download is imported or started. Every artifact is labelled:

```text
SYNTHETIC/FIXTURE STAND-IN - NOT A REAL FINE-TUNE
```

Integration status: **CP-2-pending**. P2-05 dataset manifests and P2-06 trainer
settings/contracts are consumed read-only, but CP-2 has not merged in this
workspace, so no CP-2 wiring, alias promotion, threshold change or deploy gate
is performed here.

## Run / test

From `C:\r\model-training-api` (Python 3.11+, no new dependencies):

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/trainers_whisper -q
.\.venv\Scripts\python.exe -m ruff check services/trainers/whisper tests/unit/trainers_whisper
.\.venv\Scripts\python.exe -m ruff format --check services/trainers/whisper tests/unit/trainers_whisper
.\.venv\Scripts\python.exe -m mypy services/trainers/whisper tests/unit/trainers_whisper
```

Minimal in-process run with injected synthetic examples:

```python
from services.trainers.core import TrainerSettings
from services.trainers.whisper import WhisperLoraTrainer, WhisperSettings
from tests.unit.trainers_whisper.conftest import make_example

manifest = {"content_hash": "a" * 64}  # from a P2-05 manifest, read-only
job = TrainerSettings(
    model="fake-whisper-base",
    seed=7,
    artifact_uri_prefix="fake://whisper-artifacts",
)
trainer = WhisperLoraTrainer(
    settings=WhisperSettings(replay_fraction=0.0),
    examples=(make_example("ex-1"),),
)
result = trainer.train(manifest, job)
# result["artifact_uri"] addresses the converted fake artifact, not a checkpoint.
```

For a real P2-05 HF export, inject
`HfJsonExampleLoader(path=export_result.hf_path, exclude_call_ids=...)`; the
dataset schema does not carry an export path, so the caller supplies it from
`services.datasets.export_all`. The loader reads only JSON metadata, never
audio bytes. Golden/regression call IDs must be supplied by the caller from
P1-06/P2-05; this package never opens the sealed set for training data.

## Contract

- Input: P2-05-style `dataset_manifest` with a non-empty `content_hash`, plus
  P2-06 `TrainerSettings` supplying `model`, `seed` and `artifact_uri_prefix`.
- Output: P2-06 `TrainerResult` (`trainer_contract.schema.json` 1.0.0) built by
  `build_trainer_result`: `schema_version`, `contract_part`, `artifact_uri`,
  `metrics`, `run_id` and `lineage`.
- `lineage` contains the manifest hash and the injected `git_sha` / `image_digest`
  from `TrainerSettings`; no real SHA or digest is fabricated.
- `artifact_uri` is a content-addressed fake CTranslate2 descriptor URI. It is
  not a promotion alias, champion pointer, threshold or deploy decision.

## Public API

Settings and validation:

- `WhisperSettings`, `WhisperBackend`
- `DEFAULT_LORA_R`, `DEFAULT_LORA_ALPHA`, `MAX_LORA_SCALING`
- `validate_conservative_lora`, `LoraConfig`

Dataset seam:

- `WhisperTrainingExample`, `DataManifest`
- `example_from_hf_row`, `build_data_manifest`, `mix_replay`
- `SampleRateResolver`, `StaticSampleRateResolver`, `EventSampleRateResolver`
- `ReplayProvider`, `FakeReplayProvider`
- `ReadOnlyHashGuard`, `HashCheck`
- `EXPECTED_GOLDEN_HASH`, `EXPECTED_FIXTURE_HASH`, `STANDIN_WARNING`
- `canonical_json`, `sha256_text`

Training / artifact seam:

- `WhisperLoraTrainer`
- `ExampleLoader`, `InMemoryExampleLoader`, `HfJsonExampleLoader`,
  `validate_training_examples`
- `LoraTrainingBackend`, `FakeTransformersPeftBackend`, `FakeLoraAdapter`,
  `LoraDelta`
- `MergeBackend`, `FakeMergeBackend`, `MergedModel`
- `Ctranslate2ConvertBackend`, `FakeCtranslate2Backend`, `ConvertedModel`,
  `DecodingSettings`
- `ConvertedModelStore`, `FakeConvertedModelStore`
- `Ctranslate2Runtime`, `FakeCtranslate2Runtime`, `TranscriptionResult`

## Fake seams and real-adapter injection points

Every stage behind the plugin is a Protocol; the shipped default is fake.

| Stage | Protocol | Fake default | Real seam |
| --- | --- | --- | --- |
| Examples | `ExampleLoader` | `InMemoryExampleLoader` / `HfJsonExampleLoader` | Caller-owned P2-05 export reader |
| Sample rate | `SampleRateResolver` | `StaticSampleRateResolver` / `EventSampleRateResolver` | Audio metadata service (never this package) |
| Replay | `ReplayProvider` | `FakeReplayProvider` | Audited replay sampler with call-level disjointness |
| LoRA training | `LoraTrainingBackend` | `FakeTransformersPeftBackend` | Inject a backend that owns transformers/PEFT imports; select `backend="transformers_peft"` only with injection |
| Merge | `MergeBackend` | `FakeMergeBackend` | Real adapter merge implementation |
| Convert | `Ctranslate2ConvertBackend` | `FakeCtranslate2Backend` | Real CTranslate2/faster-whisper exporter |
| Store | `ConvertedModelStore` | `FakeConvertedModelStore` | Content-addressed artifact storage |
| Runtime | `Ctranslate2Runtime` | `FakeCtranslate2Runtime` | Real converted-artifact decoder (P3-04) |

The trainer fails closed if `WhisperSettings(backend="transformers_peft")` is
selected without an injected `LoraTrainingBackend`. No heavy dependency is
available by default and none is installed by this task.

## Conservative LoRA policy (engineering guidance, not tuned truth)

`context.md` §10 records that over-scaled LoRA can hallucinate text on silence
and cites one reported fix from effective scaling `2.0 → 0.2`; the same section
says this is a single data point to verify on our own data. P2-08's fixture set
includes a `silence_hallucination` case, so the risk class is represented in
rehearsal data. Defaults and guards here are deliberately conservative:

- `r=8`, `alpha=1.0`, effective scaling `alpha/r=0.125` (below `2.0` and the
  reported `0.2` fix).
- Hard bound `max_lora_scaling=2.0` (`VOICELOOP_WHISPER_MAX_LORA_SCALING`);
  values above it fail closed.
- Decoding defaults avoid guest text: empty `initial_prompt`, empty `hotwords`,
  `condition_on_previous_text=False` (CTranslate2 empty/truncated-output risk
  in `context.md` §10). P2-08 boost config remains a separate runtime seam.

These values are **not a tuned production setting**. A real fine-tune must
re-check silence, other-language and e2e booking slices on the exact converted
artifact via P3-04 before any promotion claim (§2 invariants 3 and 4).

## Replay and 8 kHz handling

- Replay is selected only from an injected pool, using `random.Random(seed)`,
  `floor(fraction * primary_count)` examples, sorted IDs and call-disjoint
  selection. Duplicate IDs and excluded calls fail closed.
- Every mixed example records `replay`, `source`, `sample_rate_hz`,
  `telephony_band`, `effective_bandwidth_hz`, `sample_rate_policy` and
  `resampled=False`.
- 8 kHz rows are **tagged, not resynthesised**: the fake preprocessor preserves
  the native rate and records 4 kHz effective bandwidth. Real resampling belongs
  in an injected preprocessor and must be validated on real audio.
- Text and audio URIs are stored only as SHA-256 digests. Artifacts contain no
  transcript text, names, emails, phone numbers or card digits.

## Protected data and hash checks

`ReadOnlyHashGuard` reads the sealed golden seal/manifest `content_hash` and
hashes `tests/fixtures/turns.jsonl` before and after each run. Expected values:

```text
golden content hash 662cdaf43ee2d5fde03faf5e69fde196114d4f312406d94c00bf53714d03da8e
fixture sha256      aca23a303954d035245a97f78319df3694d63611b717a2ff2131bc2e0a4453fd
```

It never calls the P1-06 seal-verification API with audit arguments, never
writes to protected paths and never uses the golden set as training data.

## Settings

Whisper hyperparameters use `VOICELOOP_WHISPER_*`, e.g.
`VOICELOOP_WHISPER_LORA_R`, `VOICELOOP_WHISPER_LORA_ALPHA`,
`VOICELOOP_WHISPER_REPLAY_FRACTION`, `VOICELOOP_WHISPER_REPLAY_SEED`,
`VOICELOOP_WHISPER_PRESERVE_NATIVE_SAMPLE_RATE`. Job identity and lineage stay
in P2-06 `TrainerSettings` (`VOICELOOP_TRAINERS_*`).

## Known gaps

- No real LoRA weights, CTranslate2 binary, audio decoding or GPU run exists.
- CP-2 integration is pending; no parser/CP-2 alias or threshold is touched.
- The sealed golden artifact has simulated labels and is a workflow stand-in.
- P3-04 must evaluate the exact deployed converted artifact; e2e booking
  accuracy, not WER, decides promotion (§2 invariants 3 and 4).
