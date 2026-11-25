# P2-12 Lineage and access control

Pure-Python governance package (3.11+) over the committed P2-05 dataset
manifest contract and the P2-06 trainer result contract. **No Postgres, S3,
MLflow, identity provider, network or GPU access.** Every external dependency
is an injectable protocol with a deterministic in-memory fake.

> [!IMPORTANT]
> **Deletion is not physical deletion.** It is a fail-closed tombstone /
> denylist that future dataset builds must honour, plus non-destructive model
> flags. It never edits aliases, promotion state or sealed golden sets, and it
> does not assert any legal/compliance conclusion (`docs/compliance.md` remains
> OPEN — counsel review required).

## Layout and public API

| Module | Entry points |
|---|---|
| `graph.py` | `LineageGraph`, `DatasetRef`, `ModelRef`, `ExampleLineage`, `P205ExampleIdResolver`, `MappingExampleResolver`, `UnknownCallError`, `UnknownExampleError` |
| `deletion.py` | `DeletionWorkflow`, `DeletionReason`, `DeletionReport`, `guard_manifest`, `DeletedCallInBuildError` |
| `roles.py` | `Role`, `Capability`, `Principal`, `capabilities_for()` (the role matrix is a read-only module-local view and is not re-exported) |
| `store.py` | Protocols and fakes for manifests, trainer results, tombstones, exclusions, model flags, audit and clock; fixed `AuditAction`, `AuditReason`, `Resource` and `AuditPurpose` codes |
| `settings.py` | `LineageSettings` (`VOICELOOP_LINEAGE_` prefix) |

Access-control decisions live in `services/datasets/auth/policy.py`:
`Authorizer`, `AuthorizationError`, `AuthorizationDecision`,
`assert_pseudonymous_call_ref`.

```python
from services.datasets.auth import Authorizer
from services.datasets.lineage import (
    DeletionReason,
    DeletionWorkflow,
    FakeManifestSource,
    FakeTrainerLineageSource,
    FixedClock,
    InMemoryAuditSink,
    InMemoryModelFlagSink,
    InMemoryTombstoneSink,
    LineageGraph,
    P205ExampleIdResolver,
    Principal,
    Role,
)

graph = LineageGraph(
    FakeManifestSource(committed_manifests),
    FakeTrainerLineageSource(trainer_results),
    example_resolver=P205ExampleIdResolver(committed_example_ids),
)
lineage = graph.example_to_call("call-0001:1:parser_clean")
models = graph.models_for_call("call-0001")

clock = FixedClock(fixed_utc_datetime)
audit = InMemoryAuditSink()
tombstones = InMemoryTombstoneSink()
flags = InMemoryModelFlagSink()
workflow = DeletionWorkflow(
    graph=graph,
    authorizer=Authorizer(audit, clock),
    tombstones=tombstones,
    model_flags=flags,
    audit_sink=audit,
    clock=clock,
)
report = workflow.delete(
    "call-0001", Principal("admin-1", Role.ADMIN), DeletionReason.ERASURE_REQUEST
)
```

`lineage.call_id` and `report.models` use exact IDs; audit events and reports
use only `call_ref = call_sha256:<32 hex>` instead of raw call IDs.

## Run and test

There is no CLI or server. From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/lineage_access -q
.\.venv\Scripts\python.exe -m ruff check services/datasets/lineage services/datasets/auth tests/unit/lineage_access
.\.venv\Scripts\python.exe -m ruff format --check services/datasets/lineage services/datasets/auth tests/unit/lineage_access
.\.venv\Scripts\python.exe -m mypy services/datasets/lineage services/datasets/auth tests/unit/lineage_access
```

The tests build a real P2-05 manifest from the read-only synthetic fixture pool
and contract-valid P2-06 trainer results; no fixture, schema or service root
file is modified.

## Contracts and lineage rules

- **Manifest → call:** `source_call_ids` from `dataset_manifest` 1.0.0, validated
  with `voiceloop_common.validate`. `content_hash` is the exact join key to
  training. Conflicting duplicate manifests raise `ConflictingLineageError`;
  identical duplicates are deduplicated deterministically.
- **Trainer → model:** contract 1.0.0 `TrainResponse` results are validated
  read-only; `lineage.dataset_hash` must equal the manifest `content_hash`.
  Links use `run_id` (or an explicitly injected registry model ID via
  `model_id_for_run`) — never artifact text or partial IDs. Missing lineage
  fails closed because the trainer schema otherwise permits it.
- **Example → call:** manifest 1.0.0 does **not** store per-example IDs. An
  exact example index must therefore be injected (`P205ExampleIdResolver` over
  committed example IDs, or `MappingExampleResolver`); without one, example
  queries fail closed. This is an honest contract gap, not a conflict with
  `context.md` §2 invariant 9.
- **Call → every model:** all datasets containing the exact call are joined to
  every model on their exact content hash, deduplicated and sorted.

## Deletion and future builds

1. `Authorizer.require(DELETION_AUTHORIZE)` audits an allow or deny decision.
   Only `admin` is granted this capability; denial performs no side effects.
2. `LineageGraph` resolves impacted datasets and models. An unknown call or
   absent manifest raises and emits a `deletion_failed` audit event.
3. `TombstoneSink.deny()` records the exact call ID (the only place it is kept
   verbatim); `ExclusionSource.excluded_call_ids()` exposes it to builds.
4. Every affected model receives both `tainted` and `needs_retrain` flags via
   `ModelFlagSink`. No alias, promotion or rollback API is called.
5. A `deletion` audit event records actor, role, reason code, timestamp,
   pseudonymous `call_ref`, dataset IDs and affected model IDs. The audit sink
   is hash-chained; `verify_chain()` detects edits. Repeated requests are
   audited as `deletion_duplicate` and do not duplicate flags.
6. `guard_manifest()` validates a build's manifest and rejects any tombstoned
   call with `DeletedCallInBuildError`. To block a call before hashing, callers
   of the read-only P2-05 builder pass
   `regression_call_ids=tombstones.excluded_call_ids()` (or a future builder
   parameter dedicated to tombstone exclusions) and then call
   `guard_manifest()` on the result.

No numeric retention decision, erasure remedy or lawful basis is inferred
here; those remain human/counsel questions under `docs/compliance.md`.

## Roles (default deny)

| Role | Granted capabilities |
|---|---|
| `reviewer` | `review_read`, `review_write` only |
| `ml_engineer` | `dataset_build`, `train`, `lineage_read` |
| `admin` | `lineage_read`, `deletion_authorize`, `consent_change`, `audit_read` |

No review-write, dataset-build, train, deletion or consent capability is
granted outside its row. `Principal.role` is an immutable `Role` enum; strings
and spoofed claims are rejected, and no role-assignment API exists here. The
role matrix is a module-local `MappingProxyType` over copied `frozenset`
values: attempts to replace a role's capabilities raise `TypeError`, attempts
to add a capability raise `AttributeError`, and the package does not re-export
the matrix. `capabilities_for(role)` is the only supported accessor. A
production identity-provider adapter must verify credentials out of band and
construct the principal; none ships in this package. All check outcomes,
including denials, are appended to the audit sink.

`Authorizer.check()` requires `Capability`, `Resource` and `AuditPurpose` enum
members. Raw strings are not accepted and never enter audit records: an
unknown action/resource/purpose causes a fixed-code `DENY` event with reason
`unknown_action`, `unknown_resource` or `unknown_purpose`, followed by an
`AuthorizationError`. `InMemoryAuditSink.record()` rejects any event whose
action, resource, purpose or reason is not one of the fixed enums. Audit
records therefore contain fixed codes only — no free-text request data,
secrets or PII.

## Settings (`VOICELOOP_LINEAGE_`)

| Variable | Default | Meaning |
|---|---|---|
| `VOICELOOP_LINEAGE_PSEUDONYM_DOMAIN` | `voiceloop-lineage-v1` | Public, stable domain separator for one-way call references (not a secret) |
| `VOICELOOP_LINEAGE_AUDIT_LOG_PATH` | `reports/lineage/lineage_audit.jsonl` | Path for a future file-backed audit adapter; unused by the in-memory fake |

## Real integration seams (not implemented here)

- `ManifestSource.manifests()` — production adapter reads committed manifests
  from the dataset store; the fake validates them on ingest.
- `TrainerLineageSource.results()` / `model_id_for_run()` — production adapter
  reads contract results and registry model IDs from MLflow; the fake is
  in-memory.
- `TombstoneSink` / `ExclusionSource` — production adapter is an append-only
  deny-list consulted by the dataset builder; the fake is a dict.
- `ModelFlagSink` — production adapter tags model versions in the registry
  without alias changes; the fake is a list.
- `AuditSink` — production adapter appends immutable events to a restricted
  audit store; the fake is a hash-chained tuple.
- `Clock` — `SystemClock` uses `voiceloop_common.utcnow()`; `FixedClock` is the
  deterministic fake.

## Known gaps

- Fake storage/auth only: no transaction across tombstone + flags + audit. On
  a flag-sink failure the call remains tombstoned (fail closed) and a
  `deletion_failed` event is written, but a real adapter should add
  write-ahead/retry semantics.
- Tenant scoping (P4-05) is not modelled; single-project assumption from
  `context.md` §16 applies.
- Deletion reports identify models but do not themselves retrain or remove
  deployed artifacts; adapters cannot be surgically unlearned.
