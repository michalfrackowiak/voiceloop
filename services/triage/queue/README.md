# Sampling, clustering and daily review queue (P2-02)

Offline, deterministic construction of the human-review queue. This package
takes turn events and P2-01 failure scores through **injectable protocols**,
clusters near-duplicate hypotheses, reserves an unbiased random slice, applies
a daily cap, and marks a configurable share of items blind (no parser prefill).
It never touches a live call, a database, a model, or a network (context.md
§2 invariant 1), and its scores are triage hints, never review labels (§2.2).

## Layout

| File | Responsibility |
|---|---|
| `settings.py` | Typed `VOICELOOP_TRIAGE_QUEUE_*` settings. |
| `sampling.py` | Seeded random slice and blind-item samplers; explicit `random`/`failure` tags. |
| `clustering.py` | Stdlib token/character similarity and complete-link clustering. |
| `queue.py` | Turn/score protocols and fakes, queue construction, report, CLI. |
| `__init__.py` | Public API re-exports. |

Ownership is `services/triage/queue/` only. P2-01's `services/triage/scoring/`
is consumed as a seam but is never imported or edited by this package.

## Run and test

From the repository root (PowerShell):

```powershell
# Generate a P2-01 score report first (separate lane, read-only here).
.\.venv\Scripts\python.exe -m services.triage.scoring.scoring --output reports/triage/scores.json

# Build the queue on the read-only synthetic fixtures.
.\.venv\Scripts\python.exe -m services.triage.queue.queue `
  --scores reports/triage/scores.json --day 2026-11-25 --output reports/triage/queue.json

.\.venv\Scripts\python.exe -m pytest tests/unit/triage_queue -q
.\.venv\Scripts\python.exe -m ruff check services/triage/queue tests/unit/triage_queue
.\.venv\Scripts\python.exe -m ruff format --check services/triage/queue tests/unit/triage_queue
.\.venv\Scripts\python.exe -m mypy services/triage/queue tests/unit/triage_queue
```

The CLI accepts P2-01's `{"explanations": [...]}` JSON report, a bare JSON
array, a JSON object mapping `event_id` to score, or JSON Lines records with
`event_id` and `score`. It only reads `tests/fixtures/turns.jsonl` to locate
events; fixture `failure_type` / expected labels never enter the queue. The
`-m` invocation can emit a harmless `RuntimeWarning` about the module already
being imported with its package, as in P2-01; output is unaffected.

## Public API

```python
from services.triage.queue import (
    ExplanationScoreSource,
    FixtureTurnSource,
    InMemoryScoreSource,
    TriageQueueSettings,
    build_queue,
)

settings = TriageQueueSettings(  # env prefix VOICELOOP_TRIAGE_QUEUE_
    random_slice_target=2,  # small-pool override; leave unset in production
    daily_cap=8,
    blind_share=0.5,
)
report = build_queue(FixtureTurnSource(), InMemoryScoreSource(scores), settings, day="2026-11-25")
for item in report.items:
    payload = item.to_review_item()  # ReviewItemCreate shape for P2-03
```

`TurnSource.iter_turns() -> Iterator[TurnRecord]` and
`ScoreSource.snapshot_for(event_id) -> ScoreSnapshot` are the two seams.
`InMemoryTurnSource` / `InMemoryScoreSource` are deterministic fakes.
`ExplanationScoreSource(store)` duck-types any P2-01-style store exposing
`all()` with explanations carrying `event_id`, `score` and `signal_names`;
P2-01's `InMemoryExplanationStore` works unchanged. This avoids importing an
in-flight lane while keeping the integration a one-line adapter in the caller.
`ScoreFileSource` reads a saved score report for the CLI.

`QueueItem.to_dict()` is the rich queue view: `event_id`, `call_id`,
`turn_index`, `audio_uri`, `hypothesis`, `score`, `slice`, `cluster_id`,
`cluster_size`, `no_prefill`, `training_consent`, `signals`,
`low_confidence_spans`. `QueueItem.to_review_item()` emits exactly the
`ReviewItemCreate` fields required by `schemas/review_adapter.openapi.yaml`
1.0.0, mapping `no_prefill` to `blind`. No schema file was changed.

## Behaviour and settings

| Setting (env var without prefix) | Default | Meaning |
|---|---:|---|
| `RANDOM_SLICE_FRACTION` | `0.04` | Fraction of eligible turns sampled before the failure filter; constrained to 0.03–0.05. |
| `RANDOM_SLICE_TARGET` | unset | Optional integer override for tiny pools/tests. |
| `RANDOM_SEED` | `20261125` | Base seed; named `slice`/`blind` streams are independent. |
| `FAILURE_THRESHOLD` | `0.0` | A turn is a failure candidate when `score > threshold`. |
| `CLUSTER_SIMILARITY_THRESHOLD` | `0.82` | Complete-link similarity `0.6*token_jaccard + 0.4*char_trigram_dice`. |
| `DAILY_CAP` | `30` | Maximum queued items per day (roadmap: ~5 h/week reviewer capacity). |
| `INTERLEAVE_EVERY` | `5` | Upper bound on failure items between random-slice items. |
| `BLIND_SHARE` | `0.20` | Share of final queue items with `no_prefill=True`. |
| `LOW_CONFIDENCE_CEILING` | `0.60` | Word confidence below which a highlight span is emitted. |

Pipeline order:

1. Eligibility: recording consent must be `true`; obvious 13–19 digit card
   numbers in hypothesis/parser input are screened. Training consent is not
   required for review but is carried as `training_consent` so the dataset
   builder can enforce §2.8 downstream.
2. Random slice: seeded sample of **all** eligible IDs, before scores are
   consulted, tagged `slice: random`. Random items are never clustered, so
   deduplication cannot bias the measured error rate.
3. Failure filter and clustering: score `> threshold`, excluding random IDs;
   complete-link groups by hypothesis; the highest-score member represents
   each cluster. Random items receive a singleton pseudo-cluster ID
   (`random-<event_id>`), so every cluster in the output has exactly one
   representative.
4. Cap and ordering: all random items are reserved first (truncated only when
   `daily_cap < slice size`, which is recorded in `stats.deferred_by_cap`);
   remaining slots take the highest-score representatives. Random items are
   interleaved through the priority list, not appended wholesale.
5. Blind items: seeded sample of the final queue at `ceil(len(items) *
   blind_share)`, tagged `no_prefill=True`.

### Percentage caveat on the 36 synthetic fixtures

Defaults stay in the mandated 3–5% band. With exactly 36 turns there is no
integer count inside 3–5% (1 = 2.8%, 2 = 5.6%), so fixture tests use the
explicit `random_slice_target` override and record the deviation. In
production, where pools are large, leave the override unset. This is the only
known tension with §2 invariant 6; intent and selection mechanism remain
compliant (unbiased, score-independent, explicitly tagged).

## Known gaps

- Review labels, routing, prefill and the Label Studio/Argilla choice belong to
  P2-03. `to_review_item()` matches the OpenAPI contract but no HTTP push is
  performed here.
- Real P2-01 scores are not imported; callers adapt the store or pass a saved
  report. Fixture tests use fakes that read event metadata only.
- Cluster similarity is a deterministic heuristic, not phonetic ASR or a
  semantic embedding; tune the threshold from reviewed duplicates later.
- The daily cap is per `build_queue` call. Persistent per-day idempotency and
  reviewer-capacity feedback belong to the orchestration/review lanes.
- Card-number screening is defence in depth, not the governance PII pipeline
  (P1-09/P3-10). Golden-set and dataset exclusions remain out of scope.
