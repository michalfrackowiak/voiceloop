# Review adapter (P2-03)

Tool-neutral bridge from the P2-02 review queue to human review and back to
review results. The audited boundary is `schemas/review_adapter.openapi.yaml`
1.0.0; callers never speak to Label Studio or Argilla directly.

> **STATUS: FIXTURE/FAKE STAND-IN ONLY.** No Docker daemon, Label Studio,
> Argilla, Postgres or network access was available in this environment. The
> checked-in backends are deterministic in-memory fakes with scripted reviewer
> answers. No item was reviewed by a human, no review tool was stood up, and no
> Postgres row was written. Every stored fake result carries its `tool_name`
> (`fake_in_memory` or `fake_alternate`) so the stand-in status is auditable.

## Tool choice and the real seam

The roadmap's unanswered question defaults to **Label Studio Community behind
the adapter**. That choice is *not implemented as a live service here*: it is
recorded so the seam is unambiguous. A real `LabelStudioReviewTool` (or
`ArgillaReviewTool`) belongs in `tool.py` alone. It must implement
`ReviewTool` (`tool_name`, `create_task`, `pull_stage1`, `prepare_stage2`,
`pull_stage2`) and be selected in `build_review_tool()`. Task construction,
results, settings and callers then need no changes. The second fake tool in
`tool.py` proves this: `tests/unit/review_adapter/test_tool.py` runs the same
coordinator against both fakes and gets the same contract result.

For reference only, the compose addition would look like this (**not applied**;
`infra/` is owned by another lane and was not edited):

```yaml
services:
  label-studio:
    image: heartexlabs/label-studio:latest
    restart: unless-stopped
    ports:
      - "8080:8080"
    environment:
      LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED: "true"
      LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT: /label-studio/data
      LABEL_STUDIO_USER_PASSWORD: ${LABEL_STUDIO_USER_PASSWORD:?set in .env}
    volumes:
      - label-studio-data:/label-studio/data
volumes:
  label-studio-data:
```

Reviewer accounts, task templates and object-store credentials are deployment
configuration, not source code. No credentials are committed here.

## Layout

| File | Responsibility |
|---|---|
| `interface.py` | Tool-neutral value types, routing enum and `ReviewTool`/`ReviewStore`/`SlotParser`/`QueueItemSource` protocols. |
| `settings.py` | Typed `VOICELOOP_REVIEW_*` settings. |
| `tasks.py` | Slot form from the booking schema, span sources, two-stage coordinator, warnings and idempotency. |
| `results.py` | `StoredReviewResult`, contract projection and in-memory Postgres stand-in. |
| `tool.py` | The **only** place tool implementations and tool selection live (two fakes today). |
| `README.md` | This document. |

## Run and test

From the repository root (PowerShell), no services or network required:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/review_adapter -q
.\.venv\Scripts\python.exe -m ruff check services/review_adapter tests/unit/review_adapter
.\.venv\Scripts\python.exe -m ruff format --check services/review_adapter tests/unit/review_adapter
.\.venv\Scripts\python.exe -m mypy services/review_adapter tests/unit/review_adapter
```

There is no CLI: this adapter is a library for the queue producer and the
dataset consumer. Both fakes stay pending until a caller supplies scripted
answers, mirroring a real tool waiting for a human. The fixture rehearsal in
`tests/unit/review_adapter/test_fixture_flow.py` supplies those answers from
read-only synthetic fixtures and asserts both stages plus one stored result.

## Public API

```python
from services.review_adapter import (
    InMemoryReviewStore,
    ReviewCoordinator,
    ReviewAdapterSettings,
    SequenceQueueSource,
    ScriptedReview,
    build_review_tool,
)

settings = ReviewAdapterSettings(tool="fake_in_memory")  # env prefix VOICELOOP_REVIEW_
tool = build_review_tool(
    settings,
    reviews={
        "event-id": ScriptedReview(
            corrected_transcript="The name is Mr. Smith.",
            slots={
                "check_in": "2026-12-05",
                "check_out": "2026-12-07",
                "guests": 1,
                "room_type": "standard",
                "guest_name": "Mr. Smith",
            },
            routing="stt_wrong",
        )
    },
)
store = InMemoryReviewStore()
coordinator = ReviewCoordinator(tool, store, settings=settings)

item_id = coordinator.push(queue_item)  # P2-02 QueueItem, read-only
transcript = coordinator.stage1(item_id)  # stage 1: corrected transcript
result = coordinator.stage2(item_id)  # stage 2: form + prefill -> stored
assert coordinator.pull_result(item_id) is result
results = coordinator.run_all(SequenceQueueSource(queue_items))  # batch form
```

`QueueItemSource.iter_items()` accepts P2-02 `QueueReport` through
`QueueReportSource`, or any sequence via `SequenceQueueSource`.
`stage1()`/`stage2()` can be called by a poller; `run_item()`/`run_all()` do
both stages in one call. `push()` is idempotent on the source event, and the
store inserts once per `event_id`, so a duplicate submission returns the same
result instead of creating a second row.

### Payload provenance

- `audio_uri`, `hypothesis`, `call_id`, `turn_index`, `blind` and highlights are
  read from the P2-02 `QueueItem`. The pushed `ReviewItemCreate` dictionary has
  exactly the contract fields (no extra keys).
- Highlights are the item's explicit `stt.words[*].conf` spans (below the
  P2-02 ceiling). P2-01 `FiredSignal` currently carries only name, weight,
  severity and reason — no audio timings — so `ExplanationSpanSource` uses
  them only if a future explanation actually supplies `low_confidence_spans`,
  otherwise it falls back to the queue item's spans. Nothing is fabricated.
- Stage 2 fields and enums come from the active booking JSON Schema via
  `schemas.booking.load_booking_schema()` (`VOICELOOP_BOOKING_SCHEMA` or
  `VOICELOOP_REVIEW_BOOKING_SCHEMA_PATH` can point at a project schema).
- Prefill calls the injected `SlotParser` on the **corrected** transcript.
  Blind items (`no_prefill=True` → `blind=True`) never call the parser and
  receive `prefill=None`. Parser failures become PII-free warnings.
- `routing` is restricted to `stt_wrong`, `parser_wrong`, `both_fine`,
  `unusable_audio`; other values are rejected before persistence.

## Results, warnings and storage

`StoredReviewResult.to_api_result()` emits only the nine fields allowed by
OpenAPI `ReviewResult`. `to_dict()` is the richer internal/Postgres record:
`item_id` plus the original `event_id`, `call_id`, `turn_index`, `audio_uri`,
`hypothesis`, `reviewer_id`, `started_at`, `completed_at`, `warnings`, `blind`,
`tool_name` and `training_consent`. `to_dataset_view()` is a flat seam for the
P2-05 reviewed-item adapter; it is not a turn event and does not invent labels.

The Postgres seam is a `review_results` table keyed uniquely on `event_id`
(indexed by `(call_id, turn_index)`), with `slots`/`warnings` in JSONB and
timestamps as `timestamptz`; `INSERT ... ON CONFLICT DO NOTHING` reproduces the
in-memory idempotency. Audio bytes are never stored here (context.md §2.8).

**Contract wording mismatch:** the task text asks to persist `warnings` and the
original item ID with the result, but OpenAPI 1.0.0 declares
`ReviewResult`/`ReviewResultCreate` with `additionalProperties: false` and no
such fields. Rather than silently violating the contract or editing a
lead-owned schema, those values live in the internal record only. A future
contract version bump would be needed to expose them over HTTP. No
`context.md` §2 invariant is affected.

## Settings

| Setting (env var without `VOICELOOP_REVIEW_`) | Default | Meaning |
|---|---|---|
| `TOOL` | `fake_in_memory` | `fake_in_memory` or `fake_alternate`; unknown values fail closed. |
| `BOOKING_SCHEMA_PATH` | unset | Optional project-specific booking schema path. |
| `DEFAULT_REVIEWER_ID` | `fixture-reviewer` | Fallback when a fake script omits a reviewer ID. |

## Known gaps

- Real Label Studio/Argilla stand-up, task templates, reviewer assignment and
  HTTP push/pull are not implemented; only the seam and a documentation-only
  compose snippet exist. The done-when is therefore satisfied only as a fake
  fixture rehearsal, **not** against real Postgres.
- No PII/consent screening is duplicated here: P2-02 screens queue eligibility,
  P1-09/P3-10 own text/audio redaction, and P2-05 filters reviewed output.
- `both_fine` for a random-slice item must be supplied by the reviewer/fake
  script; the adapter never infers a routing label from triage scores.
- Polling/backoff and FastAPI routes are out of scope for this task; the
  coordinator exposes explicit stage methods that a poll loop can call.
