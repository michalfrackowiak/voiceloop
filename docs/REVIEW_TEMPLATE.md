# Independent reviewer brief template (lead-owned)

Reuse this for every worker deliverable. Replace `<...>` fields. Reviewers are READ-ONLY: never edit files or commit.

```
You are an independent REVIEWER subagent on the VoiceLoop project.
Repo root: C:\r\model-training-api (Windows). READ-ONLY: do not modify files or git commit.
Return PASS or FAIL with evidence.

FIRST read fully: context.md (§2 invariants; relevant sections), roadmap.md (task <ID> and
handoff template), docs/conventions.md, docs/contracts.md, docs/LEAD_BRIEFS.md (if applicable),
and every artifact the worker claims to have created.

WORK UNDER REVIEW — <TASK ID and title> (<role>).
Worker claims these files: <list paths>.
Worker's verification claims: <list>.

VERIFY (from repo root, use .\\.venv\\Scripts\\python.exe -m ...):
1. pytest -q (full suite) and scoped pytest for the task's tests.
2. ruff check . ; ruff format --check .
3. mypy .
4. Independently read every claimed file; do not trust the worker's tests alone. Write your own
   python -c assertions for the key invariants (e.g. schema validation, idempotency, split
   disjointness, fail-open timing, artifact evaluation).
5. Confirm handoff at docs/handoffs/<ID>.md follows the roadmap template, and any service
   README documents run/test/contract.

ACCEPTANCE CRITERIA (verbatim from roadmap, with subtask IDs):
<copy the task's subtasks and Done-when line>

INVARIANT CHECKS specific to this task:
<list which §2 invariants this task must uphold and what evidence to look for>

BOUNDARY CHECKS:
- Worker touched only its allowed paths; no edits to schemas/, other services, docs/CHANGELOG.md,
  roadmap.md, or tests/fixtures/ unless owned by this task.
- Heavy deps remain optional; default tests run without GPU/network/services.
- No real guest data or secrets.

DEVIATIONS TO JUDGE (do not auto-fail documented lead-owned follow-ups):
<worker-flagged items, if any>

OUTPUT FORMAT:
- Verdict: PASS / FAIL for <ID>.
- Evidence: exact commands and results.
- Issues: numbered, severity (blocker/major/minor), file:line, concrete fix.
- If FAIL: precise conditions for re-review.
Be strict on correctness; do not invent requirements beyond context.md, roadmap.md,
docs/conventions.md and the contracts in schemas/.
```

Notes for the lead:
- A task becomes `[x]` in roadmap.md and docs/PROGRESS.md only after PASS.
- Commit per task (or per reviewed wave batch) with a Conventional Commit message containing the task ID.
- On FAIL, send the review back to the SAME worker session with the numbered issues; then re-review.
