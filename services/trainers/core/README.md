# Trainer core (P2-06)

Plugin interface, registry and pinned-image job runner shared by every trainer
(parser, Whisper, NeMo). This package depends only on `voiceloop_common` and the
standard library: **no MLflow, Docker, network or GPU dependency**. Heavy
integrations sit behind the interfaces below and are selected by settings.

## Contract

Trainer runs consume and produce `schemas/trainer_contract.schema.json`
**1.0.0**. Both directions are validated in `TrainerRunner.run` with
`voiceloop_common.validate(..., "trainer_contract.schema.json")`:

- **Input** (`contract_part: "input"`): `schema_version`, `dataset_manifest`
  (with a non-empty `content_hash`) and `config` (derived from typed settings).
- **Output** (`contract_part: "output"`): `artifact_uri`, numeric `metrics`,
  `run_id` and `lineage` with `dataset_hash`, `git_sha`, `image_digest`.

`artifact_uri` must address the **deployable artifact** (merged adapter or
converted model), never a raw checkpoint, so the gate evaluates what serves
(`context.md` §2 invariant 3). The runner rejects results whose lineage does
not match the job settings, keeping invariant 9 (lineage everywhere) true even
if a plugin forgets to populate it correctly. Use `build_trainer_result(...)`;
it fills lineage and validates the result for you.

The plugin method signature is `train(manifest, settings) -> result`. Here
`settings` *is* the typed config object required by `docs/conventions.md`, and
`TrainerSettings.as_contract_config(name)` projects it into the contract's
`config` object. This is a naming reconciliation with `context.md` §13's
`train(dataset_manifest, config)`, not a different interface.

## Public API

| Symbol | Purpose |
|---|---|
| `TrainerPlugin` | ABC with `name` and `train(manifest, settings)`. |
| `TrainerResult`, `TrainerLineage` | Typed dicts for the 1.0.0 TrainResponse shape. |
| `build_trainer_result` | Compose a validated result with complete lineage. |
| `TrainerRegistry` | `register(plugin, replace=False)`, `get(name)`, `list()`. |
| `TrainerRunner` | Validate → lock → train → validate → log; returns the result. |
| `TrainerSettings` | Typed env-backed settings (`VOICELOOP_TRAINERS_*`). |
| `MlflowLogger`, `FakeMlflowLogger`, `TrainingRun` | Logging seam and in-memory fake. |
| `JobLock`, `SingleJobLock`, `FileJobLock`, `JobBusyError` | Locking seam. |
| `TrainerRegistryError`, `PluginNotFoundError`, `DuplicatePluginError` | Registry errors (all `VoiceLoopError` subclasses). |

Import from `services.trainers.core`.

## Run

There is no CLI or heavy runtime in this package. A downstream plugin plug-in
looks like this:

```python
from services.trainers.core import (
    TrainerPlugin,
    TrainerRegistry,
    TrainerResult,
    TrainerSettings,
    TrainerRunner,
    build_trainer_result,
)


class MyTrainer(TrainerPlugin):
    @property
    def name(self) -> str:
        return "parser-lora"

    def train(self, manifest, settings):
        return build_trainer_result(
            manifest,
            settings,
            artifact_uri=f"{settings.artifact_uri_prefix}/merged-adapter",
            metrics={"train_loss": 0.1},
        )


registry = TrainerRegistry()
registry.register(MyTrainer())
result = TrainerRunner(registry).run("parser-lora", {"content_hash": "..."})
```

Set `VOICELOOP_TRAINERS_GIT_SHA` and `VOICELOOP_TRAINERS_IMAGE_DIGEST` in real
jobs. The all-zero defaults are fakes for local tests; production must inject
the real commit and pinned `repo@sha256:<64 hex>` reference.

| Env var | Default | Meaning |
|---|---|---|
| `VOICELOOP_TRAINERS_IMAGE_DIGEST` | `ghcr.io/voiceloop/trainer@sha256:000…` | Pinned image reference. |
| `VOICELOOP_TRAINERS_GIT_SHA` | 40 zeros | Training-code commit SHA. |
| `VOICELOOP_TRAINERS_ARTIFACT_URI_PREFIX` | `fake://trainer-artifacts` | Prefix for deployable artifact URIs. |
| `VOICELOOP_TRAINERS_MODEL` | `""` | Base model resolved from settings; never hard-coded. |
| `VOICELOOP_TRAINERS_SEED` | `0` | Seed recorded in contract config. |
| `VOICELOOP_TRAINERS_PARAMS` | `{}` | JSON object of trainer-specific hyperparameters. |

## Test

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/trainers_core -q
.\.venv\Scripts\python.exe -m ruff check services/trainers/core tests/unit/trainers_core
.\.venv\Scripts\python.exe -m ruff format --check services/trainers/core tests/unit/trainers_core
.\.venv\Scripts\python.exe -m mypy services/trainers/core tests/unit/trainers_core
```

The dummy trainer in `tests/unit/trainers_core/conftest.py` runs through the
runner end to end and asserts that the returned record is contract-valid with
all lineage, metrics, adapter URI and UTC timestamps. Concurrency tests prove a
second job is rejected while the lock is held and that the lock is released
after a training exception.

## Locking

`SingleJobLock` is a non-blocking in-process `threading.Lock`: a second job
raises `JobBusyError` rather than waiting. It is released in a `finally` block,
so exceptions cannot wedge the runner. For multiple processes, inject an
implementation of the documented `FileJobLock` protocol: hold an OS-level
exclusive lock (e.g. `fcntl.flock` on POSIX or `msvcrt.locking` on Windows) on a
well-known lock file for the whole `held()` context, reject contention with
`JobBusyError`, and release in `finally`. No file-lock implementation ships
here; orchestration (`P2-10`) owns that seam so only one job uses the training
GPU (`context.md` §11).

## MLflow adapter seam

`MlflowLogger.log_run(TrainingRun)` is the only method a real adapter needs to
implement. A production adapter should start a run with the supplied `run_id`,
log `dataset_hash`, `git_sha`, `image_digest`, `artifact_uri` (the adapter or
converted-artifact URI) and trainer name as params/tags, log `metrics`, and
record `started_at`/`finished_at` from `voiceloop_common.utcnow()` values. It
may optionally use MLflow aliasing for `@champion` / `@challenger`, but this
package never flips aliases or promotes a model.
