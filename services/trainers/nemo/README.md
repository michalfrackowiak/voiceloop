# Parakeet/NeMo trainer (P3-02) — synthetic fixture stand-in

> **SYNTHETIC/FIXTURE STAND-IN - NOT A REAL NEMO FINE-TUNE.** This package does
> not import NeMo, torch, CUDA or any GPU/network dependency. No model weights
> are learned, no audio bytes are read, and no promotion, alias or threshold is
> changed. All outputs are labelled with `PROVENANCE_LABEL`.

Status: **CP-2-pending**. CP-2 is relaxed for parallel implementation; this lane
is complete for local rehearsal but must not be treated as CP-2 approval or as
evidence of a real fine-tune.

## What it does

- **P3-02.1** `ParakeetNeMoTrainer.train()` is a deterministic fake fine-tune
  starting from a `fake://` checkpoint descriptor. It consumes the P2-05 NeMo
  ASR manifest export (`audio_filepath`, `duration`, `text`) **read-only** and
  exports a hash-addressed `FakeNeMoArtifact` containing only SHA-256 hashes of
  transcripts and audio references, injected lineage and recorded
  hyperparameters.
- **P3-02.2** `ParakeetNeMoTrainer.emit_confidences()` builds `turn_event`
  1.0.0-shaped `stt.words` entries (`w`, `start_ms`, `end_ms`, `conf`) from the
  same fake transcript and hands the schema-valid event to an injectable
  `TriageSignalSink`. The default sink calls P2-01's `ScoringEngine.score()`
  unchanged, proving confidences flow into triage events without editing
  P2-01.

## Public API

Re-exported from `services.trainers.nemo`:

| Symbol | Purpose |
|---|---|
| `ParakeetNeMoTrainer` | P2-06 `TrainerPlugin` named `parakeet-nemo`; `train()` and `emit_confidences()`. |
| `NemoSettings` | Typed `VOICELOOP_NEMO_*` settings (backend label, export path, checkpoint URI, recorded epochs/LR/batch size, confidence model id). |
| `NemoManifestRow`, `JsonlNemoManifestSource`, `InMemoryNemoManifestSource`, `NemoManifestSource` | Read-only P2-05 JSONL loader and injectable source seam. |
| `FakeCheckpoint`, `FakeCheckpointSource`, `CheckpointSource` | Deterministic fake starting checkpoint and real-checkpoint seam. |
| `FakeDelta`, `FakeNeMoFinetuneBackend`, `NeMoFinetuneBackend` | Reproducible pseudo-deltas and injectable real fine-tune seam. |
| `FakeNeMoArtifact`, `FakeArtifactStore`, `ArtifactStore` | Validated, content-addressed stand-in artifact and persistence seam. |
| `WordConfidence`, `WordConfidenceOutput`, `fake_word_confidences`, `build_turn_event` | Deterministic schema-shaped word confidences and event builder. |
| `TriageSignalSink`, `P2TriageSignalSink`, `InMemorySignalSink`, `ConfidenceEmitter`, `ConfidenceEmission`, `ScoreRecord`, `score_event_with_triage` | P2-01-backed or fake triage sink and labelled emission record. |
| `PROVENANCE_LABEL`, `DEFAULT_CHECKPOINT_URI`, `DEFAULT_CONFIDENCE_MODEL_ID`, `canonical_json`, `sha256_text` | Stand-in marker, fake defaults and hashing helpers. |

All confidence values are SHA-256-derived stand-ins, **not** NeMo posterior
probabilities. The label is carried on wrapper records (`WordConfidenceOutput`,
`FakeNeMoArtifact`, `ConfidenceEmission`, `ScoreRecord`); it is intentionally
not added to `turn_event`, whose schema sets `additionalProperties: false`.

## Run and test

From the repository root (Windows venv shown; use the equivalent interpreter on
other platforms):

```powershell
# Scoped unit tests
.\.venv\Scripts\python.exe -m pytest tests/unit/trainers_nemo -q

# Scoped lint/type gates
.\.venv\Scripts\python.exe -m ruff check services/trainers/nemo tests/unit/trainers_nemo
.\.venv\Scripts\python.exe -m ruff format --check services/trainers/nemo tests/unit/trainers_nemo
.\.venv\Scripts\python.exe -m mypy services/trainers/nemo tests/unit/trainers_nemo
```

There is no CLI, Docker image, GPU job or network call to run. A local rehearsal
looks like:

```python
from services.datasets import export_nemo
from services.trainers.core import TrainerRegistry, TrainerRunner, TrainerSettings
from services.trainers.nemo import NemoSettings, ParakeetNeMoTrainer

nemo_settings = NemoSettings(manifest_path="path/to/dataset.nemo.jsonl")  # P2-05 export
core_settings = TrainerSettings(
    model="fixture-parakeet-base",
    seed=7,
    artifact_uri_prefix="fake://nemo-artifacts",
    git_sha="<injected>",
    image_digest="<injected>",
)
registry = TrainerRegistry()
registry.register(ParakeetNeMoTrainer(settings=nemo_settings))
result = TrainerRunner(registry, settings=core_settings).run("parakeet-nemo", manifest)
```

Call `emit_confidences(rows, core_settings, identifiers=[(event_id, call_id, turn_index), ...])`
on the same trainer instance to produce labelled events and P2-01 scores.
`identifiers` are supplied by the caller so call identity is never fabricated
from fixture rows.

## Contracts and safety

- Consumes `schemas/trainer_contract.schema.json` 1.0.0 via P2-06's
  `build_trainer_result`; no schema change.
- Consumes `schemas/dataset_manifest.schema.json` 1.0.0 as an opaque mapping
  requiring `content_hash`; the P2-05 export path is an optional plain
  `nemo_path` sidecar and is never written back.
- Produces `schemas/turn_event.schema.json` 1.0.0 `stt.words` objects and calls
  P2-01's `ScoringEngine` through a Protocol; no scoring threshold is changed.
- `tests/fixtures/turns.jsonl` (SHA-256
  `aca23a303954d035245a97f78319df3694d63611b717a2ff2131bc2e0a4453fd`) and the
  sealed golden stand-in `content_hash`
  `662cdaf43ee2d5fde03faf5e69fde196114d4f312406d94c00bf53714d03da8e` are
  checked before and after training and are never read into the artifact.
- Artifact lineage (`dataset_hash`, `git_sha`, `image_digest`) comes only from
  injected values. No transcript, audio reference, audio byte or PII is logged
  or persisted in plaintext.
- No new dependencies; Python 3.11+.

## Seams for a real NeMo build

Nothing in this package makes a real fine-tune possible by configuration alone.
A production lane must inject:

1. `NemoManifestSource.load_rows()` — ideally backed by the P2-05 export with
   real audio paths resolved outside this package.
2. `CheckpointSource.load()` — resolves the real pretrained Parakeet/NeMo
   checkpoint and returns a descriptor matching `FakeCheckpoint`.
3. `NeMoFinetuneBackend.finetune()` — owns all `nemo`/`torch`/GPU imports,
   mixed precision, data loader and learning-rate schedule, and returns
   representable deltas for the deployed artifact format.
4. `ArtifactStore.save()/load()` — persists and verifies the exact deployed
   artifact with a real content hash and registry URI.
5. `TriageSignalSink.score()` — may inject a calibrated confidence adapter;
   P2-01 remains unchanged.

Set `VOICELOOP_NEMO_BACKEND=nemo` only after injecting a backend; the default
constructor fails closed with `ConfigError` otherwise. The exact pinned NeMo
image/CLI, GPU window, MLflow adapter and promotion path remain outside this
stand-in (see `docs/handoffs/P3-02.md`).
