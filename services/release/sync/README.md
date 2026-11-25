# P3-06 alias sync agent

> [!IMPORTANT]
> **FAKE-ONLY REFERENCE IMPLEMENTATION — NO REAL INTEGRATIONS SHIP HERE.**
> The default `ChampionRegistry`, `ArtifactStore`, `ServingAdapter` and
> `HealthProbe` are deterministic in-memory fakes. There is **no MLflow server,
> object store, network, GPU, CTranslate2/ONNX runtime or vLLM endpoint** in this
> package, and none of those libraries is imported. The public protocols are
> the seams where production adapters plug in; this package only proves the
> poll → verify → atomic swap → health check → auto-revert transaction.

Background poller only: it is never called from the SDK, ingest or any live call
path (`context.md` §2 invariant 1). It **reads** `@champion` from the gate's
alias state and never writes an alias, changes a promotion threshold, promotes
or demotes a model (§14). Promotion and rollback stay gate-owned one-step alias
flips (§2 invariant 10).

## Layout

| File | Responsibility |
|---|---|
| `settings.py` | Typed `VOICELOOP_RELEASE_SYNC_` settings |
| `registry.py` | `ChampionRegistry` poll seam, alias snapshot/identity, read-only gate adapter and fake |
| `artifacts.py` | `ArtifactStore` download seam, deterministic SHA-256 tree hashing and verification, fake |
| `serving.py` | Shared `ServingAdapter` transaction protocol, STT/parser `ServingStack` and fakes |
| `health.py` | `HealthProbe` seam, scripted/state-checking fakes |
| `sync.py` | `AliasSyncAgent.tick()` orchestration, structured results/events and fake logger |

## Run and test

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/release_sync -q
.\.venv\Scripts\python.exe -m ruff check services/release/sync tests/unit/release_sync
.\.venv\Scripts\python.exe -m ruff format --check services/release/sync tests/unit/release_sync
.\.venv\Scripts\python.exe -m mypy services/release/sync tests/unit/release_sync
```

There is no CLI or scheduler in this package. Construct the agent with injected
seams and call `tick()` from your own cron/loop:
`VOICELOOP_RELEASE_SYNC_POLL_INTERVAL_SECONDS` is the recommended scheduler
interval, not an internal sleep. Call `configure_logging("release_sync")` once
at process startup to get JSON log events; the default `StructlogEventLogger`
otherwise still emits structured fields via structlog.

## Public API

```python
from services.release.sync import (
    AliasSyncAgent,
    ArtifactRef,
    ChampionSnapshot,
    CollectingLogger,
    FakeArtifactStore,
    FakeChampionRegistry,
    FakeServingStack,
    ReleaseSyncSettings,
    ServingStateProbe,
)

store = FakeArtifactStore()
digest = store.put("fake://mlflow/v2", b"converted-artifact-bytes")
registry = FakeChampionRegistry()
registry.bind("v2", ArtifactRef("fake://mlflow/v2", digest))
serving = FakeServingStack(stt_version="v1", parser_version="v1")

agent = AliasSyncAgent(
    settings=ReleaseSyncSettings(),
    registry=registry,
    artifacts=store,
    serving=serving,
    probe=ServingStateProbe(serving),
    logger=CollectingLogger(),  # omit in production to use structlog
)
result = agent.tick()  # outcome == "swapped"; both targets now serve v2
assert result.outcome.value == "swapped"
```

`ChampionSnapshot.bound(version, artifact)` / `.unbound()` build observations;
`identity()` returns `(alias, version, alias_state)`, the only change key.
`SyncResult` carries `outcome`, `alias_version`, `artifact_hash`, per-target
`previous_version` / `current_version`, `changed`, `reason`, `error_type` and
`rollback_errors`. `to_event()` is the structured log payload. Outcomes:
`unchanged`, `skipped`, `swapped`, `reverted`, `failed`, `rollback_failed`.

`agent.reset()` clears the observed identity so an operator can deliberately
retry the *same* alias state. No automatic retry happens inside `tick()`.

## Contract and transaction

1. `registry.poll()` reads alias, version, alias state, artifact URI and
   expected SHA-256. If the identity is unchanged, `tick()` reports `unchanged`
   and performs no download or serving call (idempotent by version + state).
2. A bound alias with no artifact metadata is a `failed` tick; an unbound alias
   is `skipped` and serving is left exactly as it is. The agent never binds,
   promotes or demotes an alias.
3. The artifact is downloaded into a fresh `tempfile.TemporaryDirectory` and
   hashed from the bytes on disk. A hash mismatch fails before any serving
   call; the temp directory is always removed.
4. `ServingStack.prepare()` stages on STT then parser. Any prepare error rolls
   back already-staged targets. `commit()` activates all targets; any commit
   error rolls back every target in reverse order.
5. `HealthProbe.check(expected_version=...)` runs only after a full commit. An
   unhealthy or raising probe rolls back serving to the exact recorded
   previous versions: `reverted` on success, `rollback_failed` if a rollback
   itself errors. Failures are returned, never raised into the caller.
6. Every tick emits one structured event with `task_id`, `outcome`, `alias`,
   `alias_state`, `alias_version`, `artifact_hash`, `previous_version`,
   `current_version`, `reason`, `error_type`, `rollback_errors` and
   `observed_at`. Only stable identifiers and hashes are logged: no transcripts,
   audio, credentials, URIs or exception bodies.

`GateAliasChampionRegistry` accepts any `get(alias) -> str | None` object —
including the P2-09 gate's `InMemoryAliasStore` — plus an out-of-band mapping
of version → `ArtifactRef`. Its protocol has no `set`, so read-only use is
enforced by the type.

## Settings (`VOICELOOP_RELEASE_SYNC_`)

| Variable | Default | Meaning |
|---|---|---|
| `VOICELOOP_RELEASE_SYNC_ALIAS` | `@champion` | Alias polled by the agent |
| `VOICELOOP_RELEASE_SYNC_POLL_INTERVAL_SECONDS` | `30.0` | Scheduler hint (> 0) |
| `VOICELOOP_RELEASE_SYNC_TEMP_ROOT` | unset (system temp) | Parent directory for artifact temp dirs |
| `VOICELOOP_RELEASE_SYNC_TASK_ID` | `P3-06` | Task ID stamped on log events |
| `VOICELOOP_RELEASE_SYNC_LOG_LEVEL` | `INFO` | Log level for startup configuration |

No secrets or model names belong in these settings; production adapters get
their own credentials from their deployment environment.

## Real adapter seams (not implemented here)

- `ChampionRegistry.poll()` → MLflow `get_model_version_by_alias`; map the
  model version, source artifact URI and SHA-256 tag into `ChampionSnapshot`.
- `ArtifactStore.download()` → `mlflow.artifacts.download_artifacts` (or an
  object-store client) writing exact bytes into the caller's temp directory.
  The agent re-hashes them; the adapter's reported hash is never trusted.
- `ServingAdapter` for STT → copy CTranslate2/ONNX artifact into a versioned
  staging dir, commit by one atomic symlink/rename, roll back to the recorded
  previous dir before removing staging.
- `ServingAdapter` for parser → vLLM load-adapter API for `prepare`, one
  control-plane alias switch for `commit`, switch back to the previous adapter
  before unloading the new one for `rollback`.
- `HealthProbe.check()` → STT readiness plus a parser constrained-decoding
  smoke check, comparing version tags and returning stable check IDs. Never
  return guest audio, transcripts or raw HTTP bodies in `HealthResult`.

Adapters must copy artifacts out of the temp dir during `prepare`, clean up
their own partial staging if `prepare` raises, and keep exception messages free
of secrets: the agent logs exception *types*, not exception bodies.

## Invariants and contracts touched

- §2 #1: background-only; no call-path work.
- §2 #3: only the deployable artifact URI is downloaded and hashed.
- §2 #4/#14: no thresholds, WER or gate decisions are read or changed; the
  gate's aliases are read-only input.
- §2 #7: no datasets or splits are touched.
- §2 #10: no alias flips in this package; serving auto-revert is not an MLflow
  alias rollback.
- No JSON Schema, OpenAPI, fixture, CHANGELOG or roadmap file was modified.

## Known gaps

- All four adapters are fakes. Real MLflow, CTranslate2/ONNX and vLLM adapters,
  scheduler/cron wiring and pinned serving-image rollout are downstream seams.
- The gate's audit hard-crash atomicity noted in the P2-09 review still needs a
  real MLflow backend (write-ahead audit is recommended there); this task only
  consumes alias state and does not own gate writes.
- A failed release keeps its alias binding. The serving stack stays at the
  previous version, and the identity is remembered so the agent does not
  retry forever; flipping the alias away and back (or calling `reset()`) is the
  explicit operator-driven retry path.
- Health is a probe of readiness/version, not a full end-to-end accuracy gate;
  P3-04/P3-08 own artifact decoding checks and shadow evidence.
