# P4-03 Custom review UI — synthetic fixture stand-in

> **STATUS: NOT DONE**
> Every artifact here is `SYNTHETIC/FIXTURE STAND-IN - NOT A HUMAN USABILITY RESULT`.
> No real person used this UI, no audio was played, no network call was made, and no
> promotion, alias or threshold decision was changed.

This is a deterministic, offline prototype of the review surface described by
`roadmap.md` P4-03.1–P4-03.4. It is a build/evaluation seam only. The roadmap
done-when — “usability test shows time per item no worse than the adapter-based
tool” — remains **OPEN and release-blocking** because it requires an independent
human usability comparison that cannot be produced from committed fixtures.

## Run locally

From `C:\r\model-training-api` using the repository virtual environment:

```powershell
# 1. Generate the read-only P2-04 metrics input, if it is not already available.
.\.venv\Scripts\python.exe -m services.review_adapter.metrics.cli --output ui\data\p2_04_metrics_input.json

# 2. Verify the pinned read-only inputs.
.\.venv\Scripts\python.exe -m ui.cli --verify-hashes

# 3. Decide go/no-go from the P2-04 report. The committed fixture report fails closed.
.\.venv\Scripts\python.exe -m ui.cli --go-no-go --metrics ui\data\p2_04_metrics_input.json --output ui\data\go_no_go_report.json

# 4. Regenerate the static browser artifacts.
.\.venv\Scripts\python.exe -m ui.cli --build
```

Open `ui/index.html` as a local file. It loads only inline JSON, `style.css` and
`app.js`; there is no server, CDN, telemetry, audio element or external asset.
`--build` is deterministic: the same read-only inputs and decision produce the
same bytes.

## Tests and gates

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/review_ui -q
.\.venv\Scripts\python.exe -m ruff check ui tests/unit/review_ui
.\.venv\Scripts\python.exe -m ruff format --check ui tests/unit/review_ui
.\.venv\Scripts\python.exe -m mypy ui tests/unit/review_ui
.\.venv\Scripts\python.exe -m ruff format --check .
```

## Generated contract (internal only)

`ui/data/` and `ui/index.html` are generated from read-only sources:

| Artifact | Source | Meaning |
| --- | --- | --- |
| `booking.form.json` | `schemas/booking.schema.json` | Flattened field descriptors, required flags, enum, formats and `check_out_after_check_in` rule. |
| `review_state.json` | `ui/fixtures/synthetic_items.json` + form descriptor | One deterministic state per pseudonymous item: player, diff, form, slot issues, booking side panel, routing, blind mode, second-opinion state. |
| `go_no_go_report.json` | P2-04 metrics report read via `--metrics` | Build-evaluation decision only; never a promotion decision. |
| `index.html` | `index.template.html` + the three JSON payloads | Static renderer with no network or audio. |

Internal state version is `internal-1.0.0`. No JSON Schema, OpenAPI file,
fixture, golden artifact, promotion threshold or roadmap checkbox was changed.

### Go/no-go input contract

`decide_go_no_go(report, measured=None)` reads the P2-04 report object
(`artifact_type = voiceloop.review.metrics`,
`report_schema_version = internal-1.0.0`) and fails closed when:

- `provenance.real_reviewers_used` is not `True`;
- the report status or measured source is synthetic;
- `no_clickstream_or_ui_timing_measured` is present;
- the fixture or golden hashes do not match the pinned values;
- any subjective key (`preference`, `opinion`, `subjective_score`, …) appears;
- any string fails the pseudonym/privacy screen;
- measured time, clicks or edit cost are absent for either tool, or any of the
  nine deterministic click actions is missing.

An injected comparison with `real_reviewers_used = True`, both tools measured on
`time_per_item_seconds`, `clicks_per_item`, `edit_cost_per_item`,
`click_counts` and `click_seconds_by_action`, and all click actions enumerated,
may produce `go` only when both the aggregate time/click/edit deltas and every
per-action click/time delta are no worse than the adapter baseline. An aggregate
or per-action regression produces `no_go_measured_regression`; a per-action
regression carries the stable reason `per_action_click_regression` and lists the
specific actions that hurt, even if the aggregate deltas are neutral. Injected
measurements are test seams, not human evidence: `usability_status` stays
`NOT DONE`, `usability_acceptance` stays an open release blocker, `promotion`
stays `not_evaluated`, and `alias_threshold_change` stays `none`.

Malformed `sections.per_item.*` summary objects produce a structured
`no_go_fail_closed` report with reason `malformed_p2_04_per_item_sections`
instead of a bare `dict(...)` exception. From the CLI, that reason is translated
to the normal structured failure response (`code: validation_error`, exit `2`)
after the fail-closed report is written.

### Privacy grammar

All fixture values are pseudonymous tokens: `tok-*`,
`(call|event|item|reviewer|approval|audio|guest)-token-<digits>`, ISO dates,
RFC3339 timestamps, UUIDs and 64-hex digests. Fixed public labels are
allowlisted. A value under a `description` key is exempt only when it exactly
matches a description read from the hash-pinned `schemas/booking.schema.json`;
all other descriptions are screened as ordinary text. Emails, phone-like
strings, long digit runs, object-store URIs, free-form prose and audio bytes are
rejected by `ui.privacy.assert_no_sensitive_text()` before serialization. The
screen is structural allowlisting, not a general name detector.

## Public API (package-local)

- `ui.hashes.verify_read_only_inputs()` — pinned hash/provenance checks.
- `ui.form.build_form_descriptor(schema, schema_sha256)` — schema-generated descriptor.
- `ui.form.validate_slots(descriptor, values)` — deterministic slot validation.
- `ui.player.synth_waveform(item_id)` — amplitude sequence from a pseudonymous ID digest.
- `ui.player.new_player(...)`, `with_speed`, `with_loop`, `seek`, `tick` — synthetic timeline state.
- `ui.player.diff_tokens(hypothesis, corrected)` — deterministic token diff.
- `ui.review.load_review_items()` — screened synthetic items.
- `ui.review.build_review_state(item, descriptor)` — complete internal UI state.
- `ui.review.evaluate_second_opinion(request, submissions, approvals, descriptor=None)` — fail-closed flow.
- `ui.go_no_go.decide_go_no_go(report, measured=None)` / `decide_from_files(...)` — build-evaluation decision.
- `ui.build.build_artifacts(output_dir=None, decision=None, index_path=None)` — regenerate static artifacts.
- `ui.cli.main(argv=None)` — offline command runner.

## Fake seams and real seams

Fake seams:

- Waveform amplitudes are `sha256(item_id)` bytes scaled to `8..96`; they are not audio samples.
- Playback is an injected `tick(delta_ms)` state transition with fixed speed choices and loop counts.
- Low-confidence highlights come only from pseudonymous span metadata in the synthetic item fixture.
- The slot form is generated from the hash-pinned booking schema; validators mirror the descriptor.
- Routing, blind mode and second opinion operate on enum labels, pseudonymous IDs and an injected approval allowlist.
- Go/no-go measurements are injected objects; the committed P2-04 report contains scripted fake results only.

Real seams still required by a lead/reviewer (not implemented here):

- An independent usability study with real reviewers, an adapter-based baseline, measured clicks/time/edit cost, and a written comparison that the custom UI is no worse.
- Real audio playback behind a browser/Web-Audio adapter with consented audio; this prototype deliberately has no `audio` element.
- A production schema source and validator adoption through a lead-owned contract bump; this harness reads `schemas/booking.schema.json` read-only.
- A real second-reviewer identity and approval source, and durable pseudonymous audit storage.
- A real P2-04 report path supplied via `--metrics`; the UI never authors or edits metrics.

## Pinned read-only inputs

| Input | Expected value |
| --- | --- |
| `tests/fixtures/turns.jsonl` SHA-256 | `aca23a303954d035245a97f78319df3694d63611b717a2ff2131bc2e0a4453fd` |
| `services/datasets/golden/artifacts/golden.v1.json` `content_hash` | `662cdaf43ee2d5fde03faf5e69fde196114d4f312406d94c00bf53714d03da8e` |
| `schemas/booking.schema.json` SHA-256 | `27e9553ce1d70ba94230738a26fedd8bcf7703afe6e41d1cc8d87f55b9d382aa` |

CP-3 is **pending** by parallel-wave assignment and is not claimed complete here.
Production acceptance remains gated by CP-3 and lead approval; this package only
delivers the deterministic P4-03 prototype and its scoped verification.
