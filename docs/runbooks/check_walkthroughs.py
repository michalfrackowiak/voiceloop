r"""Deterministic static checker for the P4-06 operations runbooks.

This harness never executes a runbook command, contacts a service or simulates
a human. It proves that each runbook keeps its ordered synthetic steps, exact
CLI/module citations, approvals and audits. Any missing or altered element is
a violation and the process exits non-zero (fail closed).

Run from the repository root::

    .\\.venv\\Scripts\\python.exe docs\\runbooks\\check_walkthroughs.py
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

SYNTHETIC_LABEL: Final = "SYNTHETIC/FIXTURE STAND-IN - NOT A HUMAN FOLLOW-THROUGH"
GUARDRAIL_LINE: Final = (
    "Decision guardrails: no promotion decision, no threshold change, "
    "no alias policy change, no retention decision, no compliance conclusion; golden set read-only."
)
ACCEPTANCE_LINE: Final = (
    "Human follow-through acceptance: OPEN — no person who did not write this runbook "
    "has followed it; release blocker."
)
DEPENDENCY_LINE: Final = "Dependency: CP-3-pending (parallel implementation only)."

_STEP_HEADING: Final = re.compile(
    r"^### Step (?P<step_id>[A-Z]+-\d{2}) — (?P<title>.+?)\s*$", re.MULTILINE
)
_EVIDENCE_LINE: Final = re.compile(r"^\s*-\s*((?:Command|Module):\s*(.+?))\s*$", re.MULTILINE)
_APPROVAL_LINE: Final = re.compile(r"^\s*-\s*Approval:\s*(.+?)\s*$", re.MULTILINE)
_AUDIT_LINE: Final = re.compile(r"^\s*-\s*Audit:\s*(.+?)\s*$", re.MULTILINE)
_MODULE_PATH: Final = re.compile(r"([A-Za-z0-9_./\\-]+\.(?:py|md|jsonl|json))")
_PYTEST_TARGET: Final = re.compile(r"\b((?:tests|docs)/[A-Za-z0-9_./-]+)")
_DOTTED_MODULE: Final = re.compile(r"-m\s+([A-Za-z_][A-Za-z0-9_.]*)")
_FLAGS: Final = re.compile(r"--[a-z][a-z0-9-]*")
_EMAIL_LIKE: Final = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_CARD_LIKE: Final = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_PHONE_LIKE: Final = re.compile(r"(?<!\w)\+\d(?:[ -]?\d){7,}(?!\w)")


@dataclass(frozen=True)
class StepSpec:
    """One expected ordered synthetic step in a runbook."""

    step_id: str
    title: str
    evidence: tuple[str, ...]
    approval: str
    audit: str


@dataclass(frozen=True)
class RunbookSpec:
    """Expected ordered steps for one runbook file."""

    filename: str
    steps: tuple[StepSpec, ...]


@dataclass(frozen=True)
class Violation:
    """One fail-closed checker finding; never holds source-text values."""

    runbook: str
    step_id: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-safe representation of the finding."""
        return {"runbook": self.runbook, "step_id": self.step_id, "reason": self.reason}


@dataclass(frozen=True)
class RunbookResult:
    """Per-runbook pass/fail summary."""

    filename: str
    steps_checked: int
    ok: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation of the summary."""
        return {"filename": self.filename, "steps_checked": self.steps_checked, "ok": self.ok}


_CONSENT_EVIDENCE: Final = (
    "services/datasets/filters/eligibility.py::filter_by_consent",
    "services/datasets/filters/pii.py::detect_pii",
    "card hard-drop",
)
_HARNESS_EVIDENCE: Final = ("docs/runbooks/check_walkthroughs.py::main", "OPEN")

SPECS: Final = (
    RunbookSpec(
        filename="onboarding.md",
        steps=(
            StepSpec(
                "ONB-01",
                "Scope a pseudonymous project stand-in",
                ("services/datasets/auth/policy.py::Authorizer.check", "pseudonymous"),
                "admin",
                "services/datasets/auth/policy.py",
            ),
            StepSpec(
                "ONB-02",
                "Gate consent and PII before any dataset eligibility",
                _CONSENT_EVIDENCE,
                "admin",
                "services/datasets/redaction/cli.py",
            ),
            StepSpec(
                "ONB-03",
                "Redact and pseudonymise fixture stand-in data",
                ("services.datasets.redaction.cli redact", "--output"),
                "admin",
                "services/datasets/redaction/cli.py",
            ),
            StepSpec(
                "ONB-04",
                "Build a call-split synthetic dataset with golden excluded",
                ("build_dataset", "verify_seal", "golden"),
                "ml_engineer",
                "services/datasets/manifest.py",
            ),
            StepSpec(
                "ONB-05",
                "Verify ingest and SDK fail-open wiring on fixtures",
                ("tests/unit/ingest", "tests/unit/connectors"),
                "lead",
                "services/ingest/app.py",
            ),
            StepSpec(
                "ONB-06",
                "Record the OPEN human follow-through acceptance",
                _HARNESS_EVIDENCE,
                "lead",
                "docs/runbooks/check_walkthroughs.py",
            ),
        ),
    ),
    RunbookSpec(
        filename="retraining.md",
        steps=(
            StepSpec(
                "RT-01",
                "Check data, drift or manual trigger — never calendar alone",
                ("--dry-run --check",),
                "admin",
                "infra/pipelines/cli.py",
            ),
            StepSpec(
                "RT-02",
                "Apply the cheapest-tier rule before training",
                ("services/trainers/vocab/rules.py::check_golden_regression", "tier 1"),
                "ml_engineer",
                "infra/pipelines/chain.py",
            ),
            StepSpec(
                "RT-03",
                "Gate consent and PII before any dataset build",
                _CONSENT_EVIDENCE,
                "admin",
                "services/datasets/redaction/cli.py",
            ),
            StepSpec(
                "RT-04",
                "Build a call-split dataset with tombstone exclusions",
                ("build_dataset", "guard_manifest", "excluded_call_ids"),
                "ml_engineer",
                "services/datasets/manifest.py",
            ),
            StepSpec(
                "RT-05",
                "Run the fixture pipeline chain only",
                ("--dry-run --manual",),
                "admin",
                "infra/pipelines/cli.py",
            ),
            StepSpec(
                "RT-06",
                "Evaluate the converted artifact only",
                ("services.evalgate.harness", "Gate.evaluate"),
                "ml_engineer",
                "services/evalgate/report.py",
            ),
            StepSpec(
                "RT-07",
                "Gate review is lead-approved with no threshold or alias action",
                ("decide_gate", "PromotionAuditLog", "no threshold"),
                "lead",
                "services/evalgate/gate/audit.py",
            ),
            StepSpec(
                "RT-08",
                "Record the OPEN human follow-through acceptance",
                _HARNESS_EVIDENCE,
                "lead",
                "docs/runbooks/check_walkthroughs.py",
            ),
        ),
    ),
    RunbookSpec(
        filename="rollback.md",
        steps=(
            StepSpec(
                "RB-01",
                "Read the previous champion from the hash-chained audit",
                ("PromotionAuditLog.verify_chain", "previous champion"),
                "lead",
                "services/evalgate/gate/audit.py",
            ),
            StepSpec(
                "RB-02",
                "Verify artifact hash and lineage before any serving change",
                ("verify_artifact", "lineage"),
                "ml_engineer",
                "services/release/sync/sync.py",
            ),
            StepSpec(
                "RB-03",
                "Rehearse gate-owned one-step alias rollback against a fake",
                ("rollback", "no real alias write"),
                "lead",
                "services/evalgate/gate/audit.py",
            ),
            StepSpec(
                "RB-04",
                "Rehearse serving-side auto-revert without an alias flip",
                ("tests/unit/release_sync", "no alias flip"),
                "ml_engineer",
                "services/release/sync/sync.py",
            ),
            StepSpec(
                "RB-05",
                "Reconfirm consent, card hard-drop and the sealed golden set",
                (*_CONSENT_EVIDENCE, "verify_seal"),
                "reviewer",
                "services/datasets/golden/audit.py",
            ),
            StepSpec(
                "RB-06",
                "Record the OPEN human follow-through acceptance",
                _HARNESS_EVIDENCE,
                "lead",
                "docs/runbooks/check_walkthroughs.py",
            ),
        ),
    ),
    RunbookSpec(
        filename="deletion_requests.md",
        steps=(
            StepSpec(
                "DEL-01",
                "Authorise as admin using a pseudonymous call reference",
                ("Authorizer.require", "pseudonymous"),
                "admin",
                "services/datasets/auth/policy.py",
            ),
            StepSpec(
                "DEL-02",
                "Resolve example-to-call and call-to-models lineage",
                ("LineageGraph", "call to models"),
                "admin",
                "services/datasets/lineage/store.py",
            ),
            StepSpec(
                "DEL-03",
                "Tombstone, flag and audit — never physical deletion or unlearning",
                ("DeletionWorkflow.delete", "tombstone", "never physical deletion or unlearning"),
                "admin",
                "DeletionReport",
            ),
            StepSpec(
                "DEL-04",
                "Exclude the tombstoned call from future builds",
                ("guard_manifest", "excluded_call_ids"),
                "ml_engineer",
                "services/datasets/manifest.py",
            ),
            StepSpec(
                "DEL-05",
                "Keep consent screening and card hard-drop unchanged",
                _CONSENT_EVIDENCE,
                "admin",
                "services/datasets/redaction/cli.py",
            ),
            StepSpec(
                "DEL-06",
                "Record the required counsel review with compliance still OPEN",
                ("docs/compliance.md", "counsel review required"),
                "counsel",
                "docs/compliance.md",
            ),
            StepSpec(
                "DEL-07",
                "Record the OPEN human follow-through acceptance",
                _HARNESS_EVIDENCE,
                "lead",
                "docs/runbooks/check_walkthroughs.py",
            ),
        ),
    ),
)


def _citation_violations(line: str, runbook: str, step_id: str, repo_root: Path) -> list[Violation]:
    """Check that a citation line points at a committed file and used flags exist."""
    violations: list[Violation] = []
    if line.lstrip().startswith("Module:"):
        for module_path in _MODULE_PATH.findall(line):
            if not (repo_root / module_path).is_file():
                violations.append(Violation(runbook, step_id, "cited module path does not exist"))
        return violations

    if "pytest" in line:
        pytest_targets = _PYTEST_TARGET.findall(line)
        if not pytest_targets:
            violations.append(Violation(runbook, step_id, "command has no test path citation"))
        for target in pytest_targets:
            if not (repo_root / target).exists():
                violations.append(Violation(runbook, step_id, "cited test path does not exist"))
        return violations

    modules = [module for module in _DOTTED_MODULE.findall(line) if module != "pytest"]
    if not modules:
        violations.append(Violation(runbook, step_id, "command has no module citation"))
        return violations
    for module in modules:
        candidate = repo_root.joinpath(*module.split(".")).with_suffix(".py")
        if not candidate.is_file():
            violations.append(Violation(runbook, step_id, "cited command module does not exist"))
            continue
        try:
            source = candidate.read_text(encoding="utf-8")
        except OSError:
            violations.append(Violation(runbook, step_id, "cited command module is unreadable"))
            continue
        for flag in _FLAGS.findall(line):
            if flag not in source:
                violations.append(Violation(runbook, step_id, "cited CLI flag is not implemented"))
    return violations


def _parse_steps(text: str) -> list[tuple[str, str, list[str]]]:
    """Return (step_id, title, body_lines) triples in document order."""
    lines = text.splitlines()
    found: list[tuple[str, str, list[str]]] = []
    current: tuple[str, str, list[str]] | None = None
    for line in lines:
        heading = _STEP_HEADING.match(line)
        if heading is not None:
            if current is not None:
                found.append(current)
            current = (heading.group("step_id"), heading.group("title"), [])
            continue
        if current is not None and line.startswith("##"):
            found.append(current)
            current = None
            continue
        if current is not None:
            current[2].append(line)
    if current is not None:
        found.append(current)
    return found


def check_runbook(path: Path, spec: RunbookSpec, repo_root: Path) -> list[Violation]:
    """Check one runbook against its expected ordered synthetic walkthrough."""
    violations: list[Violation] = []
    if not path.is_file():
        return [Violation(spec.filename, "-", "runbook file is missing")]
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return [Violation(spec.filename, "-", "runbook file is unreadable")]

    for required, reason in (
        (SYNTHETIC_LABEL, "missing synthetic stand-in label"),
        (GUARDRAIL_LINE, "missing decision guardrails"),
        (ACCEPTANCE_LINE, "missing OPEN human-acceptance release blocker"),
        (DEPENDENCY_LINE, "missing CP-3-pending dependency marker"),
        ("Consent/PII", "missing consent/PII marker"),
        ("card hard-drop", "missing card hard-drop marker"),
        ("golden set read-only", "missing golden-set read-only marker"),
    ):
        if required not in text:
            violations.append(Violation(spec.filename, "-", reason))

    for pattern, reason in (
        (_EMAIL_LIKE, "email-like literal is not allowed"),
        (_CARD_LIKE, "card-like digit run is not allowed"),
        (_PHONE_LIKE, "phone-like literal is not allowed"),
    ):
        if pattern.search(text) is not None:
            violations.append(Violation(spec.filename, "-", reason))

    parsed = _parse_steps(text)
    parsed_ids = [step_id for step_id, _, _ in parsed]
    expected_ids = [step.step_id for step in spec.steps]
    if parsed_ids != expected_ids:
        violations.append(
            Violation(spec.filename, "-", "ordered step sequence does not match the spec")
        )
    if len(set(parsed_ids)) != len(parsed_ids):
        violations.append(Violation(spec.filename, "-", "duplicate step IDs"))

    by_id = {step_id: (title, body) for step_id, title, body in parsed}
    for expected in spec.steps:
        if expected.step_id not in by_id:
            violations.append(Violation(spec.filename, expected.step_id, "missing ordered step"))
            continue
        title, body = by_id[expected.step_id]
        if expected.title not in title:
            violations.append(
                Violation(spec.filename, expected.step_id, "step title does not match the spec")
            )
        body_text = "\n".join(body)
        evidence = [match.group(1) for match in _EVIDENCE_LINE.finditer(body_text)]
        approvals = [match.group(1) for match in _APPROVAL_LINE.finditer(body_text)]
        audits = [match.group(1) for match in _AUDIT_LINE.finditer(body_text)]
        if not evidence:
            violations.append(
                Violation(spec.filename, expected.step_id, "missing Command/Module citation")
            )
        evidence_text = "\n".join(evidence)
        for token in expected.evidence:
            if token not in evidence_text:
                violations.append(
                    Violation(spec.filename, expected.step_id, "missing required citation token")
                )
        for line in evidence:
            violations.extend(
                _citation_violations(line, spec.filename, expected.step_id, repo_root)
            )
        if len(approvals) != 1 or expected.approval.lower() not in approvals[0].lower():
            violations.append(
                Violation(spec.filename, expected.step_id, "missing or wrong approval")
            )
        if len(audits) != 1 or expected.audit.lower() not in audits[0].lower():
            violations.append(Violation(spec.filename, expected.step_id, "missing or wrong audit"))
    return violations


def check_all(
    root: Path, *, repo_root: Path | None = None
) -> tuple[list[RunbookResult], list[Violation]]:
    """Check every known runbook under ``root/docs/runbooks``.

    Args:
        root: Directory containing ``docs/runbooks``; tests may pass a copy.
        repo_root: Root used to verify cited files; defaults to ``root``.

    Returns:
        Per-runbook summaries and all fail-closed violations.

    """
    citation_root = repo_root if repo_root is not None else root
    docs_dir = root / "docs" / "runbooks"
    results: list[RunbookResult] = []
    violations: list[Violation] = []
    for spec in SPECS:
        found = check_runbook(docs_dir / spec.filename, spec, citation_root)
        violations.extend(found)
        results.append(
            RunbookResult(
                spec.filename, len(spec.steps), not any(v.runbook == spec.filename for v in found)
            )
        )
    if not docs_dir.is_dir():
        violations.append(Violation("-", "-", "docs/runbooks directory is missing"))
    return results, violations


def main(argv: list[str] | None = None) -> int:
    """Run the checker and print a deterministic JSON report."""
    parser = argparse.ArgumentParser(description="Check P4-06 runbook walkthroughs (static only).")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root containing docs/runbooks.",
    )
    args = parser.parse_args(argv)
    results, violations = check_all(args.root)
    payload: dict[str, object] = {
        "acceptance": ACCEPTANCE_LINE,
        "cp3_status": "CP-3-pending",
        "human_followthrough_claimed": False,
        "ok": not violations,
        "release_blocker": True,
        "runbooks": [result.to_dict() for result in results],
        "steps_checked": sum(result.steps_checked for result in results),
        "synthetic_label": SYNTHETIC_LABEL,
        "violations": [violation.to_dict() for violation in violations],
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
