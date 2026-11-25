# P4-06 deletion requests — SYNTHETIC/FIXTURE STAND-IN - NOT A HUMAN FOLLOW-THROUGH

> **SYNTHETIC/FIXTURE STAND-IN - NOT A HUMAN FOLLOW-THROUGH.** The workflow
> below uses committed fake manifest, lineage, tombstone, flag and audit seams
> with a pseudonymous call reference. No real deletion, physical erasure,
> unlearning, external service or human request is processed here.

Dependency: CP-3-pending (parallel implementation only).

Decision guardrails: no promotion decision, no threshold change, no alias policy change, no retention decision, no compliance conclusion; golden set read-only.

Human follow-through acceptance: OPEN — no person who did not write this runbook has followed it; release blocker.

## Purpose

Rehearse a deletion request for one pseudonymous call reference
(`call-ref-standin`) in one pseudonymous project (`project-standin`). The only
remedy implemented is **tombstone / flag / audit** plus exclusion from future
dataset builds. Adapters cannot be surgically unlearned; no physical deletion
or unlearning is claimed. Consent/PII screening and card hard-drop remain in
force, and counsel review is required before any real action
(`docs/compliance.md` is OPEN).

## Walkthrough (synthetic, ordered)

### Step DEL-01 — Authorise as admin using a pseudonymous call reference

Check the immutable role matrix through the default-deny authorizer. A reviewer
or ML engineer cannot authorise deletion; the audit sink records denials too.

- Module: `services/datasets/auth/policy.py::Authorizer.require` (admin-only capability)
- Module: `services/datasets/lineage/roles.py::Role.ADMIN` (pseudonymous principal)
- Approval: admin (synthetic identity only; real identity proof is out of scope)
- Audit: `services/datasets/auth/policy.py` (fixed-code allow/deny events)
- Expected (synthetic): one allow decision; pseudonymous reference only, no guest identifiers.

### Step DEL-02 — Resolve example-to-call and call-to-models lineage

Use exact IDs through the read-only lineage graph. Missing or conflicting links
fail closed; no partial ID or guessed model is accepted.

- Module: `services/datasets/lineage/graph.py::LineageGraph` (example-to-call and call to models)
- Approval: admin (synthetic lineage reader)
- Audit: `services/datasets/lineage/store.py::InMemoryAuditSink` (hash-chained, pseudonymous events)
- Expected (synthetic): exact impacted dataset IDs, content hashes and model IDs; `call to models` query complete.

### Step DEL-03 — Tombstone, flag and audit — never physical deletion or unlearning

Run the committed deletion workflow. It writes a deny-list tombstone, flags
every affected model as tainted and needing retrain, and appends a deletion
report. It never edits aliases, promotion state or the sealed golden set.

- Module: `services/datasets/lineage/deletion.py::DeletionWorkflow.delete` (tombstone, flag, audit; never physical deletion or unlearning)
- Approval: admin (required for every real deletion; none executed here)
- Audit: `DeletionReport` (`services/datasets/lineage/deletion.py`, pseudonymous call reference)
- Expected (synthetic): `never physical deletion or unlearning`; zero alias or golden writes.

### Step DEL-04 — Exclude the tombstoned call from future builds

Feed tombstone exclusions into the next build before hashing, then guard the
manifest. Already-trained adapters remain flagged; they are not unlearned.

- Module: `services/datasets/lineage/deletion.py::guard_manifest` (`excluded_call_ids` check)
- Module: `services/datasets/builder.py::build_dataset` (exclusion before content hash)
- Approval: ml_engineer (synthetic build role)
- Audit: `services/datasets/manifest.py` (new content hash and filter counts)
- Expected (synthetic): the tombstoned call absent from the new manifest; old manifest rejected by the guard.

### Step DEL-05 — Keep consent screening and card hard-drop unchanged

Deletion does not broaden eligibility. Re-run the read-only gates on any
replacement data and confirm card findings are hard-dropped.

- Module: `services/datasets/filters/eligibility.py::filter_by_consent` (both consent flags required)
- Module: `services/datasets/filters/pii.py::detect_pii` (transcript and context screening)
- Module: `services/datasets/redaction/cli.py::main(redact)` — card hard-drop for every card finding
- Approval: admin (synthetic governance role)
- Audit: `services/datasets/redaction/cli.py` (`audit_events.jsonl`)
- Expected (synthetic): no non-consented or card-bearing row enters a dataset; counts only in reports.

### Step DEL-06 — Record the required counsel review with compliance still OPEN

Point to the read-only compliance questionnaire. No legal conclusion, erasure
remedy, retention value or lawful basis is decided by this runbook.

- Module: `docs/compliance.md` (read-only questionnaire; `counsel review required`)
- Approval: counsel (human review required before a real request; not simulated here)
- Audit: `docs/compliance.md` (section 6 placeholders, all OPEN)
- Expected (synthetic): counsel review required remains OPEN; no sign-off is claimed.

### Step DEL-07 — Record the OPEN human follow-through acceptance

Run the static checker. It confirms ordered steps and citations; it does not
claim that a requester, counsel or operator followed them.

- Module: `docs/runbooks/check_walkthroughs.py::main` (exit 1 on any missing step; `OPEN` in JSON)
- Approval: lead (independent deletion-request operator still required)
- Audit: `docs/runbooks/check_walkthroughs.py` (stdout JSON report)
- Expected (synthetic): `human_followthrough_claimed: false`, `release_blocker: true`.

## Verification

From the repository root:

```powershell
.\.venv\Scripts\python.exe docs\runbooks\check_walkthroughs.py
.\.venv\Scripts\python.exe -m pytest tests/unit/runbooks -q
```

No physical deletion, unlearning, alias change, threshold change, retention
decision or compliance conclusion is authorised by this runbook.
