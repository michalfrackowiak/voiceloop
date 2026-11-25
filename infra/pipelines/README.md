# Pipelines and triggers (P2-10)

Deterministic, in-process orchestration for **build dataset → train → evaluate**.
The chain calls injected step callables; it contains no ML, MLflow, network, or
GPU code. The CLI's `--dry-run` mode runs the full chain on the synthetic
fixtures from `tests/fixtures/turns.jsonl` with fakes, so everything is testable
without heavy dependencies.

## Run

From the repository root (PowerShell examples):

```powershell
# Full fixture chain: fake dataset builder -> fake trainer -> fake evaluator.
.\.venv\Scripts\python.exe -m infra.pipelines.cli --dry-run --manual

# Fixture chain driven only by data/drift (no manual override is recorded):
.\.venv\Scripts\python.exe -m infra.pipelines.cli --dry-run --state .artifacts/pipelines/last-run.json

# Trigger policy only (what cron calls): no chain unless data/drift/manual fires.
.\.venv\Scripts\python.exe -m infra.pipelines.cli --dry-run --check

# Persist verified IDs from the last successful run for the "since last run" count.
.\.venv\Scripts\python.exe -m infra.pipelines.cli --dry-run --manual --state .artifacts/pipelines/last-run.json

# Audit-only trigger check (never runs a chain).
.\.venv\Scripts\python.exe -m infra.pipelines.cli --dry-run --check --state .artifacts/pipelines/last-run.json
```

Output order is `TRIGGER: {...}`, one `STEP <name>: {...}` line per completed
step, then `RUN RECORD: {...}` with lineage and the trigger reason. Exit codes:
0 = success/no-op/check, 1 = a step failed (partial record still printed),
2 = configuration/input error. Without `--dry-run` (or injected steps in tests)
the CLI exits 2: production adapters are not wired until the lead integrates
P2-05 and P2-06.

`crontab.example` calls the CLI with `--check`. **A calendar tick is not a
trigger** (roadmap P2-10.2 and P4-02.2; the gate in context.md section 9 is
data-driven): every invocation re-evaluates the policy, and time passing alone
never starts a chain or promotes a model.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/pipelines -q
.\.venv\Scripts\python.exe -m ruff check infra/pipelines tests/unit/pipelines
.\.venv\Scripts\python.exe -m ruff format --check infra/pipelines tests/unit/pipelines
.\.venv\Scripts\python.exe -m mypy infra/pipelines tests/unit/pipelines
```

## Trigger policy

`triggers.should_trigger(...)` returns `True` only when at least one of:

| Reason | Value | Source |
|---|---|---|
| Manual override | `manual_override` | Explicit `--manual` or `VOICELOOP_PIPELINES_MANUAL=true` only; `--dry-run` never invents a human override |
| Drift alarm | `drift_alarm` | `--drift-alarm` or `VOICELOOP_PIPELINES_DRIFT_ALARM=true` |
| New verified data | `new_verified_data` | `count_new_verified(...) >= min_new_verified` |

`last_run_at` / `checked_at` are recorded for audit but **never** influence the
decision. `decide_trigger(...)` returns all satisfied reasons; the primary
`reason` in the run record uses priority manual → drift → new data. The default
`min_new_verified=200` is the context.md section 7 rule of thumb for a parser
tier-2 retrain, not a promotion threshold (promotion belongs to P2-09/lead).
CLI overrides (`--min-new-verified`, `--fixtures`, `--eval-set`, `--state`) are
re-validated through `PipelineSettings`; invalid values such as `--min-new-verified 0`
or `-5` exit 2 before any trigger decision is printed. A manual reason requires
an explicit operator action (`--manual` or env flag); `--dry-run` alone can only
trigger on data volume or drift.

With `--state PATH`, the CLI reads `verified_ids` from the last run record,
counts current fixture IDs not in that set, and writes the new record after a
successful chain. Without `--state`, the count is against an empty baseline in
memory only. The real "verified since last run" source can be substituted by the
caller of `should_trigger`; no service owns that state yet.

## Contract and public API

- `infra.pipelines.settings.PipelineSettings` — typed env settings, prefix
  `VOICELOOP_PIPELINES_` (`fixtures_path`, `min_new_verified`, `drift_alarm`,
  `manual`, `eval_sets`, `eval_report_path`, `run_record_path`,
  `trainer_git_sha`, `trainer_image_digest`).
- `infra.pipelines.triggers.should_trigger(...)` / `decide_trigger(...)` /
  `TriggerDecision` / `count_new_verified(...)` / `TriggerReason`.
- `infra.pipelines.chain.run_chain(steps, trigger, eval_sets, clock)` —
  injectable `PipelineSteps(dataset, trainer, evaluator)`; returns `RunRecord`
  or raises `TriggerNotSatisfied` / `PipelineStepError` with a partial record.
- `infra.pipelines.chain.build_fixture_steps(...)` — contract-validating fake
  steps for the fixture chain; `read_fixture_labels(...)` and
  `verified_example_ids(...)` expose its inputs.
- `infra.pipelines.cli.main(argv, settings, steps, stdout)` — in-process CLI.

Step contracts (all values come from the injected callables; no adapters are
hard-coded):

| Step | Expected return |
|---|---|
| dataset | Dataset manifest (`schemas/dataset_manifest.schema.json` 1.0.0) with `content_hash` and `dataset_id`, or `{"manifest": ..., "verified_ids": [...]}` |
| trainer | Trainer output (`schemas/trainer_contract.schema.json` 1.0.0) with `artifact_uri`, `run_id`, `lineage.dataset_hash` equal to the manifest hash |
| evaluator | `{"report_path": str, "artifact_uri": str, "report": ...}`; when `artifact_uri` is present it must equal the trained deployable artifact (context.md section 2.3) |

The successful `RunRecord.lineage` contains `dataset_hash → trainer_run_id →
eval_report_path` plus dataset ID, artifact URI, git SHA and image digest when
the trainer supplies them (context.md section 2.9). Any step exception stops the
chain, sets `status="failed"` and `error` on the record, and is re-raised as
`PipelineStepError` (never swallowed).

## Invariants and limitations

- Synthetic fixtures are **not** the sealed golden set; the fake evaluator only
  accepts `general_capability`, never `golden` or `regression`.
- Fake dataset splits are by call ID and consent-filtered; this is plumbing
  verification only, not a replacement for P2-05's builder/exporters.
- No real promotion, alias flip, MLflow write, GPU job, network call, or file
  write happens in `--dry-run` (except an explicit `--state` record).
