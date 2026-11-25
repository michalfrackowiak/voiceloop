# Parser trainer (P2-07) — **SYNTHETIC / FIXTURE TRAINING STAND-IN**

> **This package does not train a real model.** No GPU, network, `torch`,
> `transformers`, `datasets`, Unsloth, TRL, LLaMA-Factory, GLiNER2, vLLM,
> XGrammar or Outlines code is imported or executed here. The trainer derives
> deterministic *pseudo*-weight deltas from synthetic/fixture examples, exports
> them as a canonical JSON artifact and validates outputs through the booking
> schema gate. Every result and artifact is labelled
> `synthetic_training_standin: true`. Do not treat the metrics as training loss
> from a real fine-tune or promote on them — promotion uses the eval gate's
> end-to-end booking-field accuracy (`context.md` §2 invariant 4).

## What is implemented

- **P2-07.1** `ParserLoraTrainer` (`parser-lora`): LoRA SFT plugin for gold
  (corrected transcript) and noisy (original STT hypothesis) examples. Its real
  training seam is the injected `SlotTrainingBackend` Protocol; the shipped
  `FakeLoraTrainingBackend` is a deterministic stand-in.
- **P2-07.2** `Gliner2SlotTrainer` (`parser-gliner2`): optional GLiNER2
  slot-filler plugin behind the same Protocol. It **fails closed** with a
  `ConfigError` unless a backend is injected; no GLiNER2 dependency is installed
  or assumed.
- **P2-07.3** `FakeAdapterStore` exports a content-hash-addressed artifact URI,
  `FakeVllmRuntime` demonstrates idempotent load/unload and fail-closed loading
  of tampered artifacts, and `ConstrainedParserRuntime` rejects schema-invalid
  output instead of repairing it.

## Public API

Import from `services.trainers.parser`:

| Symbol | Purpose |
|---|---|
| `ParserLoraTrainer`, `Gliner2SlotTrainer` | `TrainerPlugin` implementations (`parser-lora`, `parser-gliner2`). |
| `ParserSettings` | Typed `VOICELOOP_PARSER_*` settings (backend label, epochs, LR, LoRA rank/alpha, excluded call IDs). |
| `ParserExample` | Validated gold/noisy training example (source text, reviewed slots, call ID). |
| `ExampleLoader`, `InMemoryExampleLoader`, `SftJsonlExampleLoader` | Dataset seam: inject examples or read P2-05's SFT JSONL; exclude golden calls. |
| `validate_training_examples` | Fail-closed checks: both kinds, unique IDs, excluded calls, valid slot targets. |
| `SlotTrainingBackend`, `FakeLoraTrainingBackend` | Injectable real-trainer seam and deterministic stand-in. |
| `FakeLoraAdapter`, `AdapterDelta`, `AdapterStore`, `FakeAdapterStore` | Hash-addressed artifact and persistence seam. |
| `ParserRuntime`, `FakeVllmRuntime`, `RuntimeStats` | Injectable vLLM seam with load/unload counters. |
| `ConstrainedDecoder`, `BookingSchemaDecoder`, `decode_booking_output`, `ConstrainedParserRuntime` | Strict JSON parsing + `schemas.booking.validate_booking` gate. |
| `canonical_json`, `sha256_text`, `ADAPTER_VERSION` | Deterministic export helpers. |

## Run

From the repository root (PowerShell), using only in-memory fakes:

```powershell
.\.venv\Scripts\python.exe -c "from services.trainers.core import TrainerRegistry, TrainerRunner, TrainerSettings; from services.trainers.parser import ParserLoraTrainer, ParserExample; ex=(ParserExample.from_mapping({'example_id':'g1','kind':'gold','source_text':'Check in on the twelfth.','slots':{'check_in':'2026-12-12','check_out':'2026-12-14','guests':2,'room_type':'standard','guest_name':'Ada Fixture'}}), ParserExample.from_mapping({'example_id':'n1','kind':'noisy','source_text':'Check in on the twelf.','slots':{'check_in':'2026-12-12','check_out':'2026-12-14','guests':2,'room_type':'standard','guest_name':'Ada Fixture'}})); r=TrainerRegistry(); r.register(ParserLoraTrainer(examples=ex)); print(TrainerRunner(r, settings=TrainerSettings()).run('parser-lora', {'content_hash':'d'*64})['artifact_uri'])"
```

In application code, prefer explicit injection:

```python
from services.datasets import load_golden_call_ids  # P2-05/P1-06; optional at runtime
from services.trainers.core import TrainerRegistry, TrainerRunner
from services.trainers.parser import ParserLoraTrainer, SftJsonlExampleLoader

registry = TrainerRegistry()
registry.register(
    ParserLoraTrainer(
        example_loader=SftJsonlExampleLoader(exclude_call_ids=load_golden_call_ids()),
    )
)
result = TrainerRunner(registry).run("parser-lora", manifest)  # manifest from P2-05
```

The default loader is empty: training fails closed until examples are injected
or `manifest["sft_path"]` is supplied. `SftJsonlExampleLoader` understands the
P2-05 TRL-style export (`parser_clean` → `gold`, `parser_noisy` → `noisy`) and
drops excluded call IDs; it never reads the sealed set itself. Golden and
regression call IDs must be passed in by the caller (`context.md` §2 invariants
5 and 7). The project depends on P2-05 only through this seam, so an incomplete
dataset lane cannot block this package.

## Contract

- Implements the `TrainerPlugin` interface from `services.trainers.core` and
  returns `schemas/trainer_contract.schema.json` **1.0.0** via
  `build_trainer_result`. `artifact_uri` addresses the exported stand-in
  adapter, never a raw checkpoint (`context.md` §2 invariant 3).
- Lineage (`dataset_hash`, `git_sha`, `image_digest`) comes from the manifest
  and core `TrainerSettings`; `TrainerRunner` revalidates it (§2 invariant 9).
- Artifact URIs are `fake://parser-adapters/(fake-lora|fake-gliner2)/<sha256>`.
  Export is deterministic: the same examples (in any order), seed, model and
  parameters yield the same content hash; changing any of them changes it.
  Gold-only and gold+noisy payloads differ because each delta records its kind,
  source hash and reviewed slots.
- Training targets are validated with `schemas.booking.validate_booking`, which
  honours `VOICELOOP_BOOKING_SCHEMA` and enforces cross-field rules. No schema
  file was changed.
- Duplicate example IDs, missing either kind, invalid slots or an excluded call
  are rejected. The runtime rejects missing, malformed or hash-mismatched
  artifacts before registering a handle.

## Constrained decoding check

`BookingSchemaDecoder.decode` strictly parses JSON text (no coercion, no repair)
and calls the project's validated booking helper. `ConstrainedParserRuntime`
wraps any `ParserRuntime`; generated text that is not a schema-valid booking
object raises `voiceloop_common.ValidationError`. This is a **check on generated
text**, not grammar-constrained token generation. To get real JSON-schema
constrained decoding, inject a vLLM runtime configured with a guided-decoding
backend; the shipped fake cannot enforce tokens.

## Real training (future work — commands/config, NOT executed here)

These are the exact settings and integration steps a GPU-enabled follow-up must
wire up. The wrapper CLI and image below do **not** exist in this repository;
`P2-10` owns orchestration. Verify pinned versions and flags before use.

```powershell
# 1. Build a reviewed dataset with P2-05 and export artifacts/datasets/dataset.sft.jsonl.
#    Pass load_golden_call_ids() so sealed calls never enter training.
$env:VOICELOOP_TRAINERS_MODEL = "<org>/<parser-base-model>"   # required; never hard-coded
$env:VOICELOOP_TRAINERS_GIT_SHA = "<40-hex commit>"
$env:VOICELOOP_TRAINERS_IMAGE_DIGEST = "ghcr.io/voiceloop/parser-trainer@sha256:<64-hex>"
$env:VOICELOOP_TRAINERS_ARTIFACT_URI_PREFIX = "s3://voiceloop-artifacts/parser"
$env:VOICELOOP_PARSER_BACKEND = "trl"      # unsloth | trl | llama_factory
$env:VOICELOOP_PARSER_EPOCHS = "2"
$env:VOICELOOP_PARSER_LEARNING_RATE = "0.0005"
$env:VOICELOOP_PARSER_LORA_RANK = "8"
$env:VOICELOOP_PARSER_LORA_ALPHA = "16"
$env:VOICELOOP_PARSER_EXCLUDE_CALL_IDS = '["<golden-and-regression-call-ids>"]'

# 2. Illustrative pinned-image invocation (future CLI, not shipped):
docker run --rm --gpus all -v "$PWD:/work" ghcr.io/voiceloop/parser-trainer@sha256:<64-hex> `
  parser-train --backend trl --sft /work/artifacts/datasets/dataset.sft.jsonl `
  --output /work/artifacts/parser/merged-adapter

# 3. Illustrative vLLM serving of the merged adapter with constrained decoding
#    (pin the vLLM version and verify flag names; schema constraint must use the
#    active booking JSON Schema, e.g. guided_json / response_format):
vllm serve <org>/<parser-base-model> --enable-lora `
  --lora-modules parser=/work/artifacts/parser/merged-adapter `
  --guided-decoding-backend xgrammar
```

In code, replace the fake with a backend from the pinned image:

```python
ParserLoraTrainer(
    settings=ParserSettings(backend="trl"),
    example_loader=SftJsonlExampleLoader(exclude_call_ids=load_golden_call_ids()),
    backend=MyRealTrlBackend(),  # implements SlotTrainingBackend
)
# GLiNER2 deployments: Gliner2SlotTrainer(backend=MyRealGliner2Backend())
```

The Protocol and settings above are the seam; no real backend, adapter writer,
Docker image, vLLM client, MLflow adapter or GPU job is shipped by P2-07.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/trainers_parser -q
.\.venv\Scripts\python.exe -m ruff check services/trainers/parser tests/unit/trainers_parser
.\.venv\Scripts\python.exe -m ruff format --check services/trainers/parser tests/unit/trainers_parser
.\.venv\Scripts\python.exe -m mypy services/trainers/parser tests/unit/trainers_parser
```
