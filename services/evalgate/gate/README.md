# Gate and promotion logic (P2-09 + P3-08)

> [!IMPORTANT]
> **End-to-end booking-field accuracy from audio is the only promotion metric**
> (`context.md` §2 invariant 4). A better WER never justifies promotion. The
> gate compares the **lower bound of the paired challenger − champion e2e
> delta** with a configurable margin; eval-report CIs are deltas, not absolute
> metric intervals (`docs/LEAD_BRIEFS.md`, P1-07 review notes).

Pure-Python (3.11+) gate: no MLflow, HTTP, GPU or network dependency. Real
converted-artifact runners and MLflow backends are injected through protocols;
only in-memory fakes ship here. P3-08 adds a **read-only** shadow-evidence
precondition for STT promotions. It never runs a shadow candidate, flips an
alias, edits a gate or changes a threshold.

## Layout

| File | Responsibility |
|---|---|
| `rules.py` | Pure pass/fail rules: e2e delta margin, CI level, eval-set labels, critical slices |
| `settings.py` | Typed `VOICELOOP_GATE_` settings (unchanged P2-09 contract) |
| `audit.py` | Frozen, hash-chained promotion/rollback records |
| `aliases.py` | Injectable alias store, `@challenger` → `@champion` flip, one-step rollback |
| `gate.py` | Converted-artifact hook, evaluator hook and orchestration |
| `shadow.py` | P3-08 typed read-only shadow source, validator, structured refusals and refusal audit |
| `__init__.py` | Additive public re-exports |

## Run and test

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/evalgate_gate -q
.\.venv\Scripts\python.exe -m ruff check services/evalgate/gate tests/unit/evalgate_gate
.\.venv\Scripts\python.exe -m ruff format --check services/evalgate/gate tests/unit/evalgate_gate
.\.venv\Scripts\python.exe -m mypy services/evalgate/gate tests/unit/evalgate_gate
```

There is no CLI yet: P2-10 supplies orchestration that calls `Gate.run`.

## Public API

```python
from services.evalgate.gate import (
    Gate,
    GateSettings,
    InMemoryAliasStore,
    PrefixArtifactConverter,
    PromotionAuditLog,
    ShadowEvidenceSettings,
    StaticReportEvaluator,
    StaticShadowEvidenceSource,
)

gate = Gate(
    settings=GateSettings(),
    converter=PrefixArtifactConverter(),
    evaluator=StaticReportEvaluator({converted_uri: challenger_report}),
    alias_store=InMemoryAliasStore({"@champion": "v3", "@challenger": "v4"}),
    audit_log=PromotionAuditLog("reports/evalgate/promotion_audit.jsonl"),
    shadow_source=StaticShadowEvidenceSource({("v4", "v3"): p3_07_shadow_report}),
    shadow_settings=ShadowEvidenceSettings(),
    refusal_log=PromotionRefusalLog("reports/evalgate/shadow_refusals.jsonl"),
)
decision, report = gate.evaluate(checkpoint_uri, champion_report)
result = gate.run(
    checkpoint_uri,
    champion_report,
    challenger_version="v4",
    actor_id="ml-engineer",
    reason="gate pass on fixtures",
    change_kind="stt",
)
if result.refusal is not None:
    print(result.refusal.to_dict())  # structured code + PII-free reasons
elif result.promotion is not None:
    gate.rollback(result.promotion, actor_id="oncall", reason="canary regression")
```

`change_kind` defaults to `"parser"`, so existing P2-09 callers are unchanged.
Callers must explicitly label an STT change with `change_kind="stt"`. The
shadow constructor arguments are optional for the parser path; an STT run
without a configured `ShadowEvidenceSource` fails closed.

Pure rules can also be used directly via `decide_gate(champion, challenger, settings)`.
`Gate.rollback` restores the recorded `before_version` in one alias flip and
writes a `rollback` audit record.

### Protocols for real integrations

- `ArtifactConverter.convert(source_uri) -> ConvertedArtifact` — conversion
  from trainer output to the exact artifact that will serve. P3-04 supplies the
  real runner; `PrefixArtifactConverter` is a fake.
- `ArtifactEvaluator.evaluate(artifact_uri, eval_sets) -> report` — mirrors the
  `evaluate(artifact_uri, eval_sets, config)` contract in `docs/contracts.md`.
  `StaticReportEvaluator` returns precomputed reports keyed by exact artifact
  URI and validates them against `eval_report` 1.0.0.
- `AliasStore.get/set` — one `set` is one atomic alias flip. Production is
  MLflow; `InMemoryAliasStore` is the fake.
- `AuditLog.record(...)` — append-only sink; `PromotionAuditLog` writes a
  SHA-256 hash-chained JSONL file.
- `ShadowEvidenceSource.load(request) -> report | None` — read-only lookup of
  the P3-07 `voiceloop.shadow.comparison` artifact. `StaticShadowEvidenceSource`
  and `CollectingShadowEvidenceSource` are deterministic in-memory fakes. No
  model call, network request or registry write occurs here.
- `RefusalSink.record_refusal(...)` — append-only shadow-refusal audit seam.
  `PromotionRefusalLog` writes a hash-chained JSONL file;
  `CollectingRefusalLog` is the in-memory test fake.

## Decision contract (`rules.py`)

1. Both reports are validated against `schemas/eval_report.schema.json` 1.0.0.
2. **Primary metric:** `confidence_intervals.e2e_booking_accuracy.lower` must be
   **strictly greater** than `min_e2e_margin`, and its `level` must be at least
   `min_ci_level`. WER, entity error and slot F1 never affect the decision.
3. **Critical slices:** `names`, `dates`, `numbers`, `silence` and
   `other_languages` are compared on `e2e_booking_accuracy` between champion
   and challenger. Each side must report `turns > 0`; missing slices,
   zero-turn slices and regressions beyond `slice_regression_tolerance` fail
   closed. The report schema has no per-slice CIs, so these are point
   comparisons; no absolute CI is reused as slice evidence.
4. **Evidence sets:** `general_capability` and `regression` are accepted.
   A report labelled `golden` is rejected: the synthetic fixtures are not the
   sealed set and no set manifest exists yet (§2 invariant 5). A future sealed
   set needs a real manifest plus a lead-approved contract change before it can
   count as evidence.
5. Every failure carries a human-readable reason in `GateDecision.reasons`;
   `GateDecision.to_report_field()` emits the schema-compatible `gate_decision`
   object.
6. **P3-08 STT precondition:** all P2-09 rules above still apply. Shadow point
   deltas are evidence only, never promotion proof. An STT run additionally
   needs one complete, synthetic-labelled, fresh P3-07 report bound to the
   exact converted candidate and champion artifact URIs.

## Settings (`VOICELOOP_GATE_`)

| Variable | Default | Meaning |
|---|---|---|
| `VOICELOOP_GATE_MIN_E2E_MARGIN` | `0.01` | Required margin on the e2e delta lower bound (strict `>`) |
| `VOICELOOP_GATE_MIN_CI_LEVEL` | `0.95` | Minimum paired-bootstrap confidence level |
| `VOICELOOP_GATE_SLICE_REGRESSION_TOLERANCE` | `0.0` | Allowed slice e2e decrease |
| `VOICELOOP_GATE_EVAL_SETS` | `general_capability,regression` | Sets the gate asks the evaluator for |
| `VOICELOOP_GATE_AUDIT_LOG_PATH` | `reports/evalgate/promotion_audit.jsonl` | Append-only audit JSONL path |

Defaults are conservative starting points. `context.md` §14 requires lead
sign-off before changing any promotion threshold in production.

### Shadow evidence settings (`VOICELOOP_GATE_SHADOW_`)

| Variable | Default | Meaning |
|---|---|---|
| `VOICELOOP_GATE_SHADOW_MAX_REPORT_AGE_HOURS` | `24.0` | Maximum age of a P3-07 comparison report |
| `VOICELOOP_GATE_SHADOW_MIN_TURNS_OBSERVED` | `1` | Minimum observed turns for evidence sufficiency |
| `VOICELOOP_GATE_SHADOW_MIN_SUCCESSFUL_RESULTS` | `1` | Minimum successful shadow results across both sides |
| `VOICELOOP_GATE_SHADOW_REFUSAL_LOG_PATH` | `reports/evalgate/shadow_refusals.jsonl` | Append-only refusal audit JSONL path |

These are evidence-sufficiency and freshness bounds, **not** promotion
thresholds. They are kept in a separate typed settings object so existing
`VOICELOOP_GATE_` behaviour is unchanged.

## STT shadow evidence and refusals

`Gate.run(..., change_kind="stt")` builds a `ShadowEvidenceRequest` from the
evaluated artifact URIs, the current `@champion` version and the challenger
report timestamp. The source is loaded once, read-only. The validator rejects:

- missing report or missing configured source;
- wrong artifact type/schema version, missing sections, counts or metrics;
- reports older than `max_report_age_hours` or predating the evaluated report;
- anything other than synthetic, fixture-labelled provenance;
- any promotion decision, alias write, champion flip, gate edit or threshold
  change claim;
- candidate/champion artifact URIs that do not match this evaluation;
- broken call-path isolation claims or embedded-evidence disagreements.

Every failure becomes a `GateRefusal` with a stable code from `REFUSAL_CODES`
and PII-free reasons, plus one `shadow_refusal` `RefusalAuditRecord`. The
record is hash-chained and verifiable with `PromotionRefusalLog.verify_chain()`.
No alias flip, threshold change, champion change or production-readiness claim
is made on refusal. Audit records carry versions, artifact URI, actor, code,
reasons and timestamp only: no transcripts, audio bytes, guest names or raw
booking values.

## Audit and rollback

`PromotionAuditLog` records actor, UTC timestamp (`voiceloop_common.utcnow()`),
reason, before/after versions and task ID. Records are frozen dataclasses and
hash-chained; `verify_chain()` detects edits. Promotion verifies that
`@challenger` still points at the evaluated version, writes `@champion` once,
then audits; a failed audit write restores the alias. Rollback similarly
verifies that `@champion` still points at the promoted version and restores the
previous one in one step. Shadow refusals use a separate audit log and never
touch this promotion record contract.

## Contracts touched

- `schemas/eval_report.schema.json` **1.0.0** — consumed and validated; no
  schema change.
- Evaluation interface `evaluate(artifact_uri, eval_sets, config) -> report.json`
  **1.0.0** — consumed through `ArtifactEvaluator`; no signature change.
- Trainer output `artifact_uri` (contract **1.0.0**) is the conversion hook's
  source URI; raw checkpoints are never evaluated directly.
- Internal P3-07 `voiceloop.shadow.comparison` **internal-1.0.0** — consumed
  read-only through a typed source; no schema, fixture or report change. Its
  embedded `shadow_evidence` block matches the optional `eval_report` 1.0.0
  shape; no new field is required.

## Known gaps

- The shipped converter and evaluator are fakes; P3-04 supplies the real
  converted-artifact runner and decoding-settings checks. **Artifact-test
  integration is pending** until that lane lands; this package does not edit
  `services/evalgate/artifact/`.
- Evaluation report 1.0.0 has no per-slice CIs; slice regression is a point
  comparison. A contract version bump is needed for slice-level CI bounds.
- The synthetic fixtures have zero `other_languages` turns, so a real gate run
  on those fixtures fails closed until a multilingual set exists.
- Real MLflow alias backend and serving-side sync belong to P3-06. P3-08 now
  requires shadow evidence for STT changes, but all shipped evidence is a
  synthetic/fixture stand-in from P3-07: point deltas only, no paired-bootstrap
  CIs, no live traffic and no production-readiness claim. Real shadow data,
  uncertainty estimates and §14 human sign-off remain open before any real
  promotion decision.
- `change_kind` is caller-supplied. Integration must label STT changes
  accurately; parser callers keep the P2-09 behaviour.
