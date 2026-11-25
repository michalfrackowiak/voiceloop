# Golden set workflow (P1-06)

This package builds **sealed evaluation sets**. A golden set is human-labelled
evaluation data: it is **never trained on**, **never used for tuning**, and
changes **only through an audited process with a distinct second reviewer**
(`context.md` §2 invariants 2 and 5). Every mutation (write, seal and amendment)
is written to an append-only audit log with actor, UTC timestamp, purpose and
task ID. Verification is read-only by default; supply actor, purpose and task
ID to append an explicit `verify` entry.

> **WARNING — synthetic workflow stand-in.** `services/datasets/golden/artifacts/golden.v1.*`
> is built from the read-only `tests/fixtures/turns.jsonl` pool: **36 tone/noise/silence
> turns across 6 calls**. It is **NOT** a real 300–500 utterance human-reviewed
> golden set from live traffic. No human listened to these WAVs; labels are
> simulated from fixture expectations. It must **not** be used for training,
> tuning, promotion decisions, or as a production golden set. It exists only to
> exercise sampling → labelling → sealing → audit end to end. No extra turns
> were fabricated to reach 300.

## Layout

| Path | Purpose |
|---|---|
| `settings.py` | `GoldenSettings`, env prefix `VOICELOOP_GOLDEN_`. |
| `sampler.py` | Deterministic call-level sampling; failure-blind `SourceTurn`. |
| `labelling.py` | Tool-agnostic export/import and distinct second-reviewer flow. |
| `audit.py` | Append-only, hash-chained access audit log. |
| `seal.py` | Content hash, versioned manifest, seal file, verification, amendment. |
| `artifacts/golden.v1.json` | Committed v1 manifest (hash, count, call IDs, second-review coverage, audit pointer). |
| `artifacts/golden.v1.labels.json` | Ordered canonical labels derived read-only from the fixture pool. |
| `artifacts/golden.v1.seal.json` | Seal over the canonical manifest bytes. |
| `artifacts/audit.jsonl` | Hash-chained access records for the sealed version. |

## Sampling policy

- **Stratified by call, never by failure or utterance.** Turns are grouped by
  `call_id`; a seeded RNG shuffles the sorted call IDs; every turn of a chosen
  call is included. A call that would exceed the upper target is skipped, not
  split, so a speaker can never appear in two sets.
- **Failure labels are invisible.** `SourceTurn` has no `failure_type` or
  `case_id` field, and sampling never reads triage signals. Unit tests rewrite
  every fixture's failure labels and assert the selection is unchanged.
- **Parameterised target band.** Production default is 300–500 utterances
  (`VOICELOOP_GOLDEN_TARGET_MIN_UTTERANCES` / `..._MAX_UTTERANCES`). Tests use
  smaller bands because the fixture pool has only 36 turns.
- **Unbiased random slice (§2 invariant 6).** In live traffic, 3–5% of turns
  bypass the failure filter so the real error rate remains measurable. Golden
  candidates likewise come from **all eligible calls**, not the failure queue;
  using only failures would bias the set. The v1 synthetic artifact selects the
  entire fixture pool and is explicitly a workflow stand-in, not an error-rate
  estimate.

```python
from services.datasets.golden import load_source_turns, sample_calls, select_entire_pool

turns = load_source_turns()  # tests/fixtures/turns.jsonl, read-only
sample = sample_calls(turns, seed=7, target_min=12, target_max=18)
stand_in = select_entire_pool(turns)  # all 36 synthetic turns, 6 calls
```

## Export, import and second review

The Label Studio/Argilla choice is hidden behind the `LabelTool` protocol. Only
an in-memory fake exists here; no network calls occur. Exports carry
`call_id`, `turn_index`, `audio_uri` and `hypothesis`. Imports carry a corrected
transcript, booking slots and a `reviewer_id`. A second review must come from a
**distinct** reviewer; agreement is recorded per item as transcript agreement
and slot agreement, with an overall `agreed` flag and coverage counts.

```python
from services.datasets.golden import (
    build_label_records,
    build_label_tool,
    export_to_tool,
    import_labels,
    import_second_reviews,
    select_entire_pool,
)

tool = build_label_tool()  # VOICELOOP_GOLDEN_LABEL_TOOL=in_memory
item_ids = export_to_tool(select_entire_pool(load_source_turns()).turns, tool)
# A real workflow has humans submit results here via tool.put_result(...).
primaries = import_labels(tool, item_ids)
seconds = import_second_reviews(tool, item_ids, primaries)
labels = build_label_records(turns, primaries, seconds)
```

`slots` are validated against `booking.schema.json` 1.0.0 when non-empty;
`{}` means no bookable slot was established.

## Seal, hash and verification

`seal_golden_set` writes labels, a versioned manifest and a seal file. The
manifest `content_hash` is SHA-256 over the ordered canonical JSON of **every**
label field: `source_id`, `call_id`, `turn_index`, `corrected_transcript`,
`slots`, `first_reviewer_id`, `second_reviewer_id` and `agreed`. Verification
also rejects labels whose `source_id` is not `f"{call_id}:{turn_index}"`, whose
reviewers are identical, or whose second reviewer differs from the seal. The
labels document carries the version and is checked against the manifest. The
seal stores a SHA-256 hash of the canonical manifest bytes, so editing any
manifest field fails verification. Sealed files are never overwritten.

Verify from the repository root. The CLI requires an identity, purpose and task
ID so an explicit `verify` entry is appended to the audit log:

```powershell
.\.venv\Scripts\python.exe -m services.datasets.golden.seal verify --dir services/datasets/golden/artifacts --actor reviewer-c --purpose "release gate" --task-id CP-1
```

A passing run prints compact JSON with `"ok":true`; a broken seal prints the
errors and exits 2. Python equivalent; omit the three context fields for a
read-only verification:

```python
from services.datasets.golden import assert_sealed, verify_seal

report = verify_seal(
    "services/datasets/golden/artifacts",
    actor_id="reviewer-c",
    purpose="release gate",
    task_id="CP-1",
)
report.assert_ok()
manifest = assert_sealed("services/datasets/golden/artifacts")  # read-only
```

Passing only some of `actor_id`, `purpose` and `task_id` is rejected.

Changes go through `amend_sealed_set(...)`, which verifies the predecessor,
requires a distinct second reviewer, writes a new `golden.vN` version and adds
an `amend` audit entry. The previous version stays reproducible.

## Inspect the audit log

```powershell
Get-Content services\datasets\golden\artifacts\audit.jsonl
```

Or in Python:

```python
from services.datasets.golden import AccessAuditLog

log = AccessAuditLog("services/datasets/golden/artifacts/audit.jsonl")
for entry in log.entries():
    print(entry.timestamp, entry.actor_id, entry.action, entry.task_id, entry.purpose)
assert log.verify_chain() == ()  # no unaudited edits to prior entries
```

The manifest's `audit_pointer` records the entry count and head hash at seal
time. Explicit `verify` entries may be appended by supplying the audit context;
altering the sealed prefix fails verification. Implicit `verify_seal()` and
`assert_sealed()` calls do not append anything.

## Regenerate the v1 synthetic stand-in

Do this **only for the initial fixture artifact**. A sealed version must never
be edited in place; use `amend_sealed_set` for real changes. The committed
artifact is deterministic because the clock is frozen and the source is the
read-only fixture manifest. Remove the old `artifacts` directory first (or
write to a temporary directory) when regenerating the stand-in:

```python
from datetime import UTC, datetime

from voiceloop_common import frozen_clock, use_clock
from services.datasets.golden import (
    load_source_turns,
    seal_golden_set,
    select_entire_pool,
    synthetic_labels_from_turns,
)

with use_clock(frozen_clock(datetime(2026, 11, 25, tzinfo=UTC))):
    turns = load_source_turns()  # tests/fixtures/turns.jsonl
    sample = select_entire_pool(turns)
    labels = synthetic_labels_from_turns(
        sample.turns,
        primary_reviewer_id="reviewer-a-synthetic",
        second_reviewer_id="reviewer-b-synthetic",
    )
    artifact = seal_golden_set(
        "services/datasets/golden/artifacts",
        labels,
        dataset_id="golden.v1.synthetic",
        actor_id="data-worker-p1-06",
        reviewer_id="reviewer-b-synthetic",
        task_id="P1-06",
        purpose="Synthetic workflow stand-in; not a production golden set",
        synthetic=True,
        workflow_stand_in=True,
        review_mode="simulated_synthetic_labels",
    )
    print(artifact.manifest.content_hash)
```

Expected: 36 labels, 6 call IDs, second-review coverage 1.0, simulated
agreement 36/36. The resulting hash must match the committed manifest.

## Run and test

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/golden -q
.\.venv\Scripts\python.exe -m ruff check services/datasets/golden tests/unit/golden
.\.venv\Scripts\python.exe -m ruff format --check services/datasets/golden tests/unit/golden
.\.venv\Scripts\python.exe -m mypy services/datasets/golden tests/unit/golden
```

## Contracts

No `schemas/` file is changed. `booking.schema.json` 1.0.0 validates non-empty
slots, `review_adapter.openapi.yaml` 1.0.0 supplies the item/result shape, and
`dataset_manifest.schema.json` 1.0.0 is read-only reference only: the sealed
golden manifest is an internal, versioned artifact owned by this directory.
