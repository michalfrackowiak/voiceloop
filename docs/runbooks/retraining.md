# P4-06 retraining — SYNTHETIC/FIXTURE STAND-IN - NOT A HUMAN FOLLOW-THROUGH

> **SYNTHETIC/FIXTURE STAND-IN - NOT A HUMAN FOLLOW-THROUGH.** The chain below
> runs with fake dataset, trainer and evaluator steps over committed synthetic
> fixtures. No real GPU, MLflow, object store, network, promotion or human
> retraining decision occurs.

Dependency: CP-3-pending (parallel implementation only).

Decision guardrails: no promotion decision, no threshold change, no alias policy change, no retention decision, no compliance conclusion; golden set read-only.

Human follow-through acceptance: OPEN — no person who did not write this runbook has followed it; release blocker.

## Purpose

Rehearse the cheapest-first retraining path for one pseudonymous project
(`project-standin`). Tier 1 rules are checked before any training; the sealed
golden set stays read-only. Consent/PII and card hard-drop are enforced before
a build. Promotion thresholds and aliases are never changed here.

## Walkthrough (synthetic, ordered)

### Step RT-01 — Check data, drift or manual trigger — never calendar alone

Use the audit-only trigger check first. A clock tick is not a trigger; a real
manual override requires an explicit operator action.

- Command: `python -m infra.pipelines.cli --dry-run --check`
- Approval: admin (synthetic trigger reviewer)
- Audit: `infra/pipelines/cli.py` (`TRIGGER` line and audit-only decision)
- Expected (synthetic): a decision with no chain run unless data volume, drift or explicit manual input fires.

### Step RT-02 — Apply the cheapest-tier rule before training

Confirm tier 1 vocabulary/normalisation was considered first. The golden
regression check is read-only and never tunes the sealed set.

- Module: `services/trainers/vocab/rules.py::check_golden_regression` (tier 1 read-only check)
- Approval: ml_engineer (synthetic tier-routing reviewer)
- Audit: `infra/pipelines/chain.py` (`RunRecord` step summaries)
- Expected (synthetic): tier 1 checked before any tier 2/3 training is proposed; no golden mutation.

### Step RT-03 — Gate consent and PII before any dataset build

Apply the read-only P1-09 gates to every reviewed item, including slots and
dialogue context. A card finding hard-drops the whole item before export.

- Module: `services/datasets/filters/eligibility.py::filter_by_consent` (both consent flags required)
- Module: `services/datasets/filters/pii.py::detect_pii` (transcript, slots and context)
- Module: `services/datasets/redaction/cli.py::main(redact)` — card hard-drop for every card finding
- Approval: admin (synthetic governance role)
- Audit: `services/datasets/redaction/cli.py` (`audit_events.jsonl`)
- Expected (synthetic): only eligible rows proceed; drop records contain reasons and counts, not values.

### Step RT-04 — Build a call-split dataset with tombstone exclusions

Pass tombstone exclusions into the builder before hashing, then guard the
manifest. Golden and regression calls remain held out.

- Module: `services/datasets/builder.py::build_dataset` (whole-call splits)
- Module: `services/datasets/lineage/deletion.py::guard_manifest` (`excluded_call_ids` guard)
- Module: `services/datasets/golden/seal.py::verify_seal` (read-only verification)
- Approval: ml_engineer (synthetic build role)
- Audit: `services/datasets/manifest.py` (content hash and filter counts)
- Expected (synthetic): a contract-valid manifest; no tombstoned or sealed-golden call in any split.

### Step RT-05 — Run the fixture pipeline chain only

Run the full chain with injected fakes. Without `--dry-run` the CLI exits 2;
no production adapters are wired by this runbook.

- Command: `python -m infra.pipelines.cli --dry-run --manual`
- Approval: admin (synthetic explicit manual override)
- Audit: `infra/pipelines/cli.py` (`RUN RECORD` with dataset hash, run ID and report path)
- Expected (synthetic): dataset → train → evaluate steps succeed on fixtures; no alias write.

### Step RT-06 — Evaluate the converted artifact only

The primary metric is end-to-end booking-field accuracy from audio. WER alone
never justifies a promotion. Only fake artifacts are evaluated here.

- Command: `python -m services.evalgate.harness --output <temp-eval-report>`
- Module: `services/evalgate/gate/gate.py::Gate.evaluate` (converted-artifact hook)
- Approval: ml_engineer (synthetic evaluator role)
- Audit: `services/evalgate/report.py` (contract-valid `report.json` writer)
- Expected (synthetic): a delta-CI report labelled as a fixture stand-in; no champion claim.

### Step RT-07 — Gate review is lead-approved with no threshold or alias action

Apply the committed pass rules to the fixture report only. The gate's defaults
are engineering defaults, not approved production thresholds.

- Module: `services/evalgate/gate/rules.py::decide_gate` (e2e-only rule, no threshold edits)
- Module: `services/evalgate/gate/audit.py::PromotionAuditLog` (hash-chained records)
- Approval: lead (required for any real gate/threshold decision; none is made here)
- Audit: `services/evalgate/gate/audit.py` (`no threshold` change recorded, zero promotion writes)
- Expected (synthetic): a refusal or pass reason for fixture evidence only; aliases unchanged.

### Step RT-08 — Record the OPEN human follow-through acceptance

Run the static checker. It confirms ordered steps and citations; it does not
prove a human followed this retraining runbook.

- Module: `docs/runbooks/check_walkthroughs.py::main` (exit 1 on any missing step; `OPEN` in JSON)
- Approval: lead (independent retraining operator still required)
- Audit: `docs/runbooks/check_walkthroughs.py` (stdout JSON report)
- Expected (synthetic): `human_followthrough_claimed: false`, `release_blocker: true`.

## Verification

From the repository root:

```powershell
.\.venv\Scripts\python.exe docs\runbooks\check_walkthroughs.py
.\.venv\Scripts\python.exe -m pytest tests/unit/runbooks -q
```

No real promotion, alias flip, threshold change, golden-set change, GPU job or
network call is authorised by this document.
