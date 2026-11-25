# P4-06 operations runbooks

> **SYNTHETIC/FIXTURE STAND-IN - NOT A HUMAN FOLLOW-THROUGH.** All four
> runbooks use committed fake seams and read-only synthetic fixtures. They are
> executable rehearsals, not production procedures and not evidence that a
> human followed them.

Dependency: CP-3-pending (parallel implementation only). The dependency was
relaxed for parallel documentation work; no integration checkpoint is claimed.

Decision guardrails: no promotion decision, no threshold change, no alias policy change, no retention decision, no compliance conclusion; golden set read-only.

Human follow-through acceptance: OPEN — no person who did not write this runbook has followed it; release blocker.

## Contents

| File | Rehearsal |
|---|---|
| `onboarding.md` | Stand up one pseudonymous project stand-in with consent/PII gates and a sealed golden stand-in |
| `retraining.md` | Trigger check → cheapest tier → consent/PII and card hard-drop → build → fixture chain → converted-artifact evaluation → lead gate review |
| `rollback.md` | Read audit → verify artifact → fake gate-owned alias rollback → fake serving revert → reconfirm eligibility and seal |
| `deletion_requests.md` | Admin authorisation → lineage → tombstone/flag/audit → build exclusion → consent/PII → counsel review (OPEN) |
| `check_walkthroughs.py` | Deterministic static checker for the ordered steps above |

## How to verify

From the repository root:

```powershell
.\.venv\Scripts\python.exe docs\runbooks\check_walkthroughs.py
.\.venv\Scripts\python.exe -m pytest tests/unit/runbooks -q
```

The checker parses each runbook, asserts the exact ordered step IDs and titles,
requires a real CLI/module citation plus an approval and an audit line in every
step, verifies cited files/flags exist, and fails closed (exit 1) on any missing
element. It prints `human_followthrough_claimed: false` and
`release_blocker: true` in its JSON output. It never runs the cited commands,
never contacts a service and never fabricates a human follow-through.

## Acceptance status

P4-06 implementation is complete: the runbooks and deterministic harness exist
and are scoped-tested. The roadmap's done-when condition — *each runbook was
followed by someone who did not write it* — remains **OPEN** and is a release
blocker pending an independent operator. Consent/PII and card hard-drop steps
are present in every runbook; the golden set stays read-only; deletion is
tombstone/flag/audit only with counsel review outstanding
(`docs/compliance.md` is OPEN).
