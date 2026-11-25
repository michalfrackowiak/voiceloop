# P4-06 rollback — SYNTHETIC/FIXTURE STAND-IN - NOT A HUMAN FOLLOW-THROUGH

> **SYNTHETIC/FIXTURE STAND-IN - NOT A HUMAN FOLLOW-THROUGH.** This runbook
> rehearses the committed fake alias store and fake serving stack only. It does
> not write a real MLflow alias, change gate rules or thresholds, or claim that
> a human operator restored production.

Dependency: CP-3-pending (parallel implementation only).

Decision guardrails: no promotion decision, no threshold change, no alias policy change, no retention decision, no compliance conclusion; golden set read-only.

Human follow-through acceptance: OPEN — no person who did not write this runbook has followed it; release blocker.

## Purpose

Rehearse the gate-owned one-step rollback (`context.md` §2 invariant 10) for
one pseudonymous project (`project-standin`) using the committed fakes. The
serving-side auto-revert is a separate local transaction and never flips an
alias. Consent/PII, card hard-drop and the sealed golden set are reconfirmed
after the rehearsal.

## Walkthrough (synthetic, ordered)

### Step RB-01 — Read the previous champion from the hash-chained audit

Find the recorded `before_version` from the fake promotion audit. Do not infer
or invent a version, and do not edit the audit chain.

- Module: `services/evalgate/gate/audit.py::PromotionAuditLog.verify_chain` (previous champion from a verified record)
- Approval: lead (synthetic rollback authoriser)
- Audit: `services/evalgate/gate/audit.py` (hash-chained promotion and rollback records)
- Expected (synthetic): `ok: true` on chain verification and an exact previous version, or a fail-closed refusal.

### Step RB-02 — Verify artifact hash and lineage before any serving change

Reject an artifact whose bytes do not match the recorded digest. Lineage stays
read-only; no dataset or model tag is rewritten by this step.

- Module: `services/release/sync/artifacts.py::verify_artifact` (SHA-256 check, lineage read-only)
- Approval: ml_engineer (synthetic release engineer)
- Audit: `services/release/sync/sync.py::SyncResult.to_event` (structured, PII-free event)
- Expected (synthetic): one verified artifact reference; lineage fields unchanged.

### Step RB-03 — Rehearse gate-owned one-step alias rollback against a fake

Use the committed in-memory alias store. This is a stand-in for the gate-owned
alias flip, not a real alias write and not a new alias policy.

- Module: `services/evalgate/gate/aliases.py::rollback` (restore recorded `before_version`)
- Module: `services/evalgate/gate/gate.py::Gate.rollback` (one-step rollback; no real alias write)
- Approval: lead (required before any real rollback; none executed here)
- Audit: `services/evalgate/gate/audit.py::PromotionAuditLog` (rollback record appended)
- Expected (synthetic): fake `@champion` restored in one step; `no real alias write` recorded.

### Step RB-04 — Rehearse serving-side auto-revert without an alias flip

Run the fake sync tests to prove prepare/commit/health failure reverts the
local serving stack to exact previous versions. Retries are operator-driven,
not automatic.

- Command: `python -m pytest tests/unit/release_sync -q` (fake serving only; no alias flip)
- Approval: ml_engineer (synthetic serving operator)
- Audit: `services/release/sync/sync.py` (`release_sync.tick` structured events)
- Expected (synthetic): broken artifact reverts to the previous version; `no alias flip` occurs.

### Step RB-05 — Reconfirm consent, card hard-drop and the sealed golden set

Rollback does not relax eligibility. Re-check the read-only gates and seal
before declaring the rehearsal finished.

- Module: `services/datasets/filters/eligibility.py::filter_by_consent` (both consent flags required)
- Module: `services/datasets/filters/pii.py::detect_pii` (transcript and context screening)
- Module: `services/datasets/redaction/cli.py::main(redact)` — card hard-drop for every card finding
- Module: `services/datasets/golden/seal.py::verify_seal` (read-only verification)
- Approval: reviewer (synthetic governance reviewer)
- Audit: `services/datasets/golden/audit.py` (append-only access log)
- Expected (synthetic): zero remaining PII/card findings and a verified seal; no golden mutation.

### Step RB-06 — Record the OPEN human follow-through acceptance

Run the static checker. It confirms the ordered rollback steps but does not
claim a human performed them.

- Module: `docs/runbooks/check_walkthroughs.py::main` (exit 1 on any missing step; `OPEN` in JSON)
- Approval: lead (independent rollback operator still required)
- Audit: `docs/runbooks/check_walkthroughs.py` (stdout JSON report)
- Expected (synthetic): `human_followthrough_claimed: false`, `release_blocker: true`.

## Verification

From the repository root:

```powershell
.\.venv\Scripts\python.exe docs\runbooks\check_walkthroughs.py
.\.venv\Scripts\python.exe -m pytest tests/unit/runbooks -q
```

A real rollback remains a gate-owned one-step alias flip under §2 invariant 10;
this document records no threshold, alias-policy, retention or compliance
decision.
