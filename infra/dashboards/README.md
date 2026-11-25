# Dashboards (P2-11)

Offline, deterministic **static** HTML/Markdown dashboards rendered from local
fixture data or injected providers. There is no Prometheus/Grafana dependency,
no network call, no external dashboard service, and no promotion or rollback
action. Every metric is labelled with its **source**, **observation timestamp
(or explicit "unknown")**, **synthetic** flag and **stale** flag.

> [!WARNING]
> The default artifact is built from a **synthetic fixture labels manifest** and
> a **committed synthetic example snapshot**. It is a local plumbing demo, not a
> live operational dashboard. No real guest data, transcript, audio, credential
> or secret is rendered.

## Layout

| File | Responsibility |
|---|---|
| `settings.py` | Typed `VOICELOOP_DASHBOARDS_*` settings. |
| `data.py` | Immutable snapshots, provider protocols, fixture/snapshot providers, fakes and data assembly. |
| `render.py` | Pure HTML/Markdown renderers with escaping and provenance columns. |
| `cli.py` | `python -m infra.dashboards.cli --output <path>` entry point. |
| `examples/synthetic-snapshot.json` | Small, explicitly synthetic local snapshot; no model version or promotion is recorded. |
| `__init__.py` | Public API re-exports. |

Paths outside `infra/dashboards/` are read-only here: the default failure-rate
source is `tests/fixtures/turns.jsonl`.

## Run

From the repository root (PowerShell):

```powershell
.\.venv\Scripts\python.exe -m infra.dashboards.cli --output .artifacts/dashboards/local.html
.\.venv\Scripts\python.exe -m infra.dashboards.cli --output .artifacts/dashboards/local.md --format markdown
.\.venv\Scripts\python.exe -m infra.dashboards.cli --output report.html --fixtures tests/fixtures/turns.jsonl --snapshot infra/dashboards/examples/synthetic-snapshot.json --title "Fixture rehearsal"
```

`--output` is required; parent directories are created. Exit codes: 0 = written,
2 = config/input/render/write error (the error is printed to stdout). The
artifact is deterministic for given provider data and clock. The CLI uses the
system UTC clock for the `Generated at` line only; call
`infra.dashboards.data.collect_dashboard(..., now=...)` to freeze it in tests.

## Test and quality gates

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/dashboards -q
.\.venv\Scripts\python.exe -m ruff check infra/dashboards tests/unit/dashboards
.\.venv\Scripts\python.exe -m ruff format --check infra/dashboards tests/unit/dashboards
.\.venv\Scripts\python.exe -m mypy infra/dashboards tests/unit/dashboards
```

## Metrics and data contract

| Task | Metric | Default source | Meaning |
|---|---|---|---|
| P2-11.1 | Failure rate (fixture labels) | `tests/fixtures/turns.jsonl` | Count of `stt_wrong`, `parser_wrong`, `unusable_audio` labels divided by all 36 fixture turns (currently 10/36). No observation timestamp exists in that manifest, so it is shown as unknown and stale. |
| P2-11.1 | Turns counted | same fixture manifest | Denominator for the rate. |
| P2-11.1 | Queue depth | `examples/synthetic-snapshot.json` | Synthetic example value, or `unavailable — no queue provider configured` when an unavailable provider is injected. Never silently rendered as zero. |
| P2-11.1 | Review throughput (completed in window) | same snapshot | Completed count plus an explicit start/end window; no per-hour rate is invented when a window is missing. |
| P2-11.2 | Model version in service | injected `ModelStateProvider` | `unknown / no version recorded` when the source has no value. |
| P2-11.2 | Last promotion | injected `ModelStateProvider` | Injected version/timestamp/alias plus provenance, or exactly `unknown / no promotion recorded`. |

Failure **labels** come from the read-only synthetic fixture manifest. Triage
scores are hints, not labels (`services/triage/scoring/README.md`), so they are
not used to compute this rate and no score threshold is introduced. Queue,
review and release state are supplied through protocols, so P2-02/P2-03/P2-09
can be wired later without changing the renderer.

`examples/synthetic-snapshot.json` shape (version `1.0`):

```json
{
  "schema_version": "1.0",
  "synthetic": true,
  "notice": "SYNTHETIC EXAMPLE - LOCAL DEMONSTRATION DATA, NOT LIVE METRICS",
  "generated_at": "2026-11-25T00:00:00Z",
  "queue": {"depth": 3, "observed_at": "2026-11-25T00:00:00Z"},
  "review": {"completed": 5, "window_start": "...", "window_end": "...", "observed_at": "..."},
  "model": {"version_in_service": null, "promotion": null, "observed_at": "..."}
}
```

`model.version_in_service` and `model.promotion` are `null` in the committed
example on purpose: no version or timestamp is invented. A promotion object,
when injected by a real adapter, requires non-empty `version`, `promoted_at`
and `source`; optional `alias`, `actor` and `audit_ref` are displayed as
provenance. The dashboard only reads this state — it never flips an alias.

## Public API

```python
from datetime import UTC, datetime

from infra.dashboards import (
    DashboardSettings,
    FakeModelStateProvider,
    FakeQueueDepthProvider,
    FakeReviewThroughputProvider,
    PromotionRecord,
    collect_dashboard,
    render_dashboard,
)

settings = DashboardSettings()
data = collect_dashboard(
    settings=settings,
    now=datetime(2026, 11, 25, 12, 0, tzinfo=UTC),
    queue_provider=FakeQueueDepthProvider(depth=4, source="fake://queue"),
    review_provider=FakeReviewThroughputProvider(completed=2, source="fake://review"),
    model_provider=FakeModelStateProvider(
        version_in_service="test-v1",
        promotion=PromotionRecord(
            version="test-v1",
            promoted_at="2026-11-24T09:00:00Z",
            source="fake://release/aliases",
        ),
        source="fake://release/state",
    ),
)
artifact = render_dashboard(data, format="html")  # or "markdown"
```

Protocols to implement for real (future) adapters:

- `FailureProvider.failure_snapshot() -> FailureSnapshot`
- `QueueDepthProvider.queue_snapshot() -> QueueSnapshot`
- `ReviewThroughputProvider.review_snapshot() -> ReviewSnapshot`
- `ModelStateProvider.model_snapshot() -> ModelStateSnapshot`

`depth=None`, `completed=None`, `version_in_service=None` and `promotion=None`
mean **no data**; renderers show explicit unavailable/unknown strings. Fakes
(`Fake*Provider`) and `Unavailable*Provider` implementations make those states
testable without P2-02/P2-03/P2-09 services.

## Settings

| Environment variable | Default | Purpose |
|---|---|---|
| `VOICELOOP_DASHBOARDS_TITLE` | `VoiceLoop local dashboard` | Artifact title. |
| `VOICELOOP_DASHBOARDS_FIXTURES_PATH` | `tests/fixtures/turns.jsonl` | Read-only labels manifest. |
| `VOICELOOP_DASHBOARDS_SNAPSHOT_PATH` | `infra/dashboards/examples/synthetic-snapshot.json` | Local snapshot source. |
| `VOICELOOP_DASHBOARDS_FORMAT` | `html` | `html` or `markdown`. |
| `VOICELOOP_DASHBOARDS_STALE_AFTER_MINUTES` | `1440` | Display-only freshness window; not a promotion threshold. |

Repo-relative paths resolve against the repository root, so the CLI also works
outside that directory. No secret or tool credential belongs in these settings.

## Invariants and limitations

- No call-path work: this is an offline CLI (`context.md` §2.1).
- Fixture labels are synthetic and are **not** the sealed golden set
  (§2.2, §2.5); no human listened to the tone/noise WAVs.
- The dashboard neither evaluates nor promotes; WER/e2e accuracy are not shown
  here, and §2.4's e2e-accuracy promotion rule is untouched.
- Model state is shown only with injected provenance; §2.10's alias-flip
  rollback remains a release-service responsibility.
- Queue/review values in the default artifact come from a synthetic example,
  not a live queue or review adapter. Freshness and staleness are displayed
  per metric; a stale artifact must not be used for an operational decision.
- HTML is escaped; Markdown table separators in injected labels are escaped.
  No JavaScript, remote CSS, font, image or telemetry is included.
