# Dataset builder (P2-05)

`services/datasets/` root builds versioned, hashed training datasets from human
review results. It owns only the root modules; `filters/` (P1-09) and `golden/`
(P1-06) are imported read-only through their public APIs.

## Public API

| Module | Entry points |
|---|---|
| `builder` | `load_reviewed_items_from_fixtures`, `load_golden_call_ids`, `build_dataset`, `build_dataset_from_fixtures` |
| `examples` | `ReviewedItem`, `DatasetExample`, `expand_reviewed_item`, `build_dialogue_context` |
| `splits` | `assign_calls`, `SplitPlan`, `assert_no_call_in_multiple_splits` |
| `regression` | `RegressionStore`, `RegressionRecord`, `is_reviewed_failure` |
| `replay` | `sample_replay`, `ReplaySelection`, `is_verified_replay_turn` |
| `manifest` | `build_manifest`, `content_hash`, `FilterCounts`, `DatasetBuild` |
| `exporters` | `export_nemo`, `export_hf`, `export_sft`, `export_all` |
| `settings` | `DatasetSettings` (`VOICELOOP_DATASETS_` environment prefix) |

All symbols above are re-exported from `services.datasets`.

## Run

From the repository root with the project virtualenv:

```powershell
.\.venv\Scripts\python.exe -c "from services.datasets import build_dataset_from_fixtures, build_manifest, export_all; build = build_dataset_from_fixtures(); manifest = build_manifest(build); print(manifest['content_hash']); print(export_all(build.examples, 'artifacts/datasets').counts)"
```

`build_dataset_from_fixtures()` reads `tests/fixtures/turns.jsonl` and its events
read-only. For a production build, pass `ReviewedItem` objects from the review
adapter (`schemas/review_adapter.openapi.yaml` 1.0.0) and explicit
`golden_call_ids` / `regression_call_ids` from `load_golden_call_ids()` and
`RegressionStore.call_ids()`.

Example with exclusion and replay:

```python
from services.datasets import (
    DatasetSettings,
    RegressionStore,
    build_dataset_from_fixtures,
    load_golden_call_ids,
    build_manifest,
)

settings = DatasetSettings(replay_size=2, replay_seed=7)
golden = load_golden_call_ids()  # sealed call IDs from P1-06
store = RegressionStore()  # load(path) for a cumulative set
build = build_dataset_from_fixtures(
    settings=settings,
    golden_call_ids=golden,
    regression_call_ids=store.call_ids(),
    replay_pool=(),  # earlier verified ReviewedItems
)
manifest = build_manifest(build)  # validates against schema 1.0.0
```

Settings are typed and read `VOICELOOP_DATASETS_` variables, e.g.
`VOICELOOP_DATASETS_VAL_FRACTION`, `VOICELOOP_DATASETS_SPLIT_SEED`,
`VOICELOOP_DATASETS_REPLAY_SIZE`. No secrets or real guest data are stored here.

## Contract notes

- `build_manifest()` validates against `schemas/dataset_manifest.schema.json`
  **1.0.0** using `voiceloop_common.validate()`. It emits `content_hash` (SHA-256
  over sorted example IDs and contents), `source_call_ids`, call-ID
  `splits.train/val/test`, all six `filter_counts`, schema version and a
  `created_at` from the injectable `voiceloop_common.utcnow()`.
- Splits are **by call**, never by utterance (context.md §2 invariant 7).
  Golden and regression calls are removed before assignment, and
  `assert_no_call_in_multiple_splits()` checks all five call sets.
- Consent and PII use `services.datasets.filters` (P1-09) only: card-containing
  and non-consented rows are dropped and their reasons counted in the manifest.
  Because the event filter only sees `stt.text`, the builder also screens every
  free-text field entering an example with that package's `detect_pii`:
  corrected transcript, original hypothesis, all strings nested in reviewed
  slots, and dialogue-context text including `prior_turns[*].hypothesis`. Any
  card finding hard-drops the whole reviewed item (including replay items) and
  is audited with `DropReason.CARD_NUMBER_PII`; no card text reaches examples or
  exports. PII logic is not reimplemented.
- One reviewed item yields at most three examples (STT, clean parser, noisy
  parser), each with dialogue context. Empty-target and no-slot turns yield
  fewer examples; no turns are fabricated.
- Regression accumulates every reviewed failure by `(call_id, turn_index)`;
  `both_fine`/`clean` turns remain eligible so the unbiased random slice
  (§2 invariant 6) is preserved.
- Exports are plain JSON/JSONL: NeMo ASR manifest, Hugging Face `datasets` JSON
  array, and TRL-style SFT chat JSONL. Loading with the real target libraries
  needs optional dependencies and is a follow-up; unit tests parse the files
  back and assert record counts and fields.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/datasets -q
.\.venv\Scripts\python.exe -m ruff check services/datasets/__init__.py services/datasets/*.py tests/unit/datasets
.\.venv\Scripts\python.exe -m ruff format --check services/datasets/__init__.py services/datasets/*.py tests/unit/datasets
.\.venv\Scripts\python.exe -m mypy services/datasets/__init__.py services/datasets/*.py tests/unit/datasets
```

Synthetic fixtures are read-only. The committed synthetic golden stand-in seals
all six fixture calls; building from those fixtures **with**
`golden_call_ids=load_golden_call_ids()` therefore leaves no training calls by
design (§2 invariant 5). Tests use explicit synthetic golden/regression IDs for
exclusion so they can still exercise the fixture pool.
