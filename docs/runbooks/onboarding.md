# P4-06 onboarding a project — SYNTHETIC/FIXTURE STAND-IN - NOT A HUMAN FOLLOW-THROUGH

> **SYNTHETIC/FIXTURE STAND-IN - NOT A HUMAN FOLLOW-THROUGH.** Every command
> below runs against committed in-memory fakes and read-only synthetic
> fixtures. No real project, guest, audio, external service, network or human
> operator is involved. Do not treat a completed static check as a real
> onboarding sign-off.

Dependency: CP-3-pending (parallel implementation only).

Decision guardrails: no promotion decision, no threshold change, no alias policy change, no retention decision, no compliance conclusion; golden set read-only.

Human follow-through acceptance: OPEN — no person who did not write this runbook has followed it; release blocker.

## Purpose

Onboard one **pseudonymous project stand-in** (`project-standin`) into the
existing single-project pipeline. `context.md` §16 assumes a single tenant;
cross-project scoping is P4-05 and is **not** performed here. This runbook only
records the synthetic rehearsal that an independent operator must repeat with
real adapters before production onboarding.

Consent/PII and card hard-drop are checked before any dataset eligibility; the
sealed golden set is verified read-only and never trained on or tuned against.

## Walkthrough (synthetic, ordered)

### Step ONB-01 — Scope a pseudonymous project stand-in

Record only the pseudonymous project ID and the default-deny role boundary.
The shipped authorizer is an in-memory fake; no identity provider is contacted.

- Module: `services/datasets/auth/policy.py::Authorizer.check` (fixed-code allow/deny, pseudonymous principal)
- Approval: admin (synthetic principal only; real identity proof is out of scope)
- Audit: `services/datasets/auth/policy.py` (every allow and deny decision is recorded)
- Expected (synthetic): one allow decision for the stand-in admin principal; no guest identifiers.

### Step ONB-02 — Gate consent and PII before any dataset eligibility

Apply the read-only P1-09 gates before a reviewed item can enter a dataset.
Any card finding drops the whole item; no card digits are written to drop
records or manifests.

- Module: `services/datasets/filters/eligibility.py::filter_by_consent` (both consent flags required)
- Module: `services/datasets/filters/pii.py::detect_pii` (transcript and context screening)
- Module: `services/datasets/redaction/cli.py::main(redact)` — card hard-drop for every card finding
- Approval: admin (synthetic governance role)
- Audit: `services/datasets/redaction/cli.py` (`hard_drops.jsonl` and `audit_events.jsonl`)
- Expected (synthetic): non-consented rows and card-bearing rows excluded; counts only in the report.

### Step ONB-03 — Redact and pseudonymise fixture stand-in data

Run the fake audio/transcript rehearsal. Output paths must be under a temporary
directory; the committed fixtures stay read-only.

- Command: `python -m services.datasets.redaction.cli redact --output <temp-redaction>`
- Approval: admin (synthetic policy owner)
- Audit: `services/datasets/redaction/cli.py` (`redaction_report.json` with PII-free counts)
- Expected (synthetic): zero remaining audio or text PII spans; hard drops counted, never printed as values.

### Step ONB-04 — Build a call-split synthetic dataset with golden excluded

Build from the read-only fixture pool with whole-call splits. The sealed golden
stand-in is verified, never amended, trained on or used for tuning.

- Module: `services/datasets/builder.py::build_dataset` (call-level split, golden and regression excluded)
- Module: `services/datasets/golden/seal.py::verify_seal` (read-only verification)
- Approval: ml_engineer (synthetic build role)
- Audit: `services/datasets/manifest.py` (content hash and filter counts)
- Expected (synthetic): a manifest with a stable content hash; no call in more than one split.

### Step ONB-05 — Verify ingest and SDK fail-open wiring on fixtures

Rehearse capture with the committed fixture tests only. This proves the in-process
fakes still fail open; it does not prove a live deployment.

- Command: `python -m pytest tests/unit/ingest -q`
- Command: `python -m pytest tests/unit/connectors -q`
- Approval: lead (synthetic integration reviewer)
- Audit: `services/ingest/app.py` (structured, PII-free service events)
- Expected (synthetic): fixture turns accepted; storage-failure fakes shed or buffer without raising into callers.

### Step ONB-06 — Record the OPEN human follow-through acceptance

Run the static checker. It confirms the ordered steps, citations, approvals and
audits above; it does **not** claim a person followed them.

- Module: `docs/runbooks/check_walkthroughs.py::main` (exit 1 on any missing step; `OPEN` in JSON)
- Approval: lead (an independent operator is still required for real acceptance)
- Audit: `docs/runbooks/check_walkthroughs.py` (stdout JSON report)
- Expected (synthetic): `human_followthrough_claimed: false`, `release_blocker: true`.

## Verification

From the repository root:

```powershell
.\.venv\Scripts\python.exe docs\runbooks\check_walkthroughs.py
.\.venv\Scripts\python.exe -m pytest tests/unit/runbooks -q
```

The checker fails closed on a missing or reordered step, an uncited command, a
missing approval/audit, a PII-shaped literal or a removed release-blocker line.
Production readiness is **not** established by this runbook.
