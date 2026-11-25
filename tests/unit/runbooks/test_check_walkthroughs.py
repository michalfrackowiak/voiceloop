"""Tests for the P4-06 runbook walkthrough checker."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HARNESS_PATH = REPO_ROOT / "docs" / "runbooks" / "check_walkthroughs.py"
_STEP_BLOCK = re.compile(
    r"^### Step [A-Z]+-\d{2} —.*?(?=^### Step |^## |\Z)", re.MULTILINE | re.DOTALL
)


def _load_harness() -> ModuleType:
    """Load the standalone harness module without adding docs to the path."""
    spec = importlib.util.spec_from_file_location("p4_06_check_walkthroughs", HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HARNESS = _load_harness()


def _copy_runbooks(tmp_path: Path) -> Path:
    """Copy only the runbook docs into a temporary root for mutation."""
    target = tmp_path / "docs" / "runbooks"
    target.mkdir(parents=True)
    for spec in HARNESS.SPECS:
        shutil.copy2(REPO_ROOT / "docs" / "runbooks" / spec.filename, target / spec.filename)
    return tmp_path


def _remove_step(path: Path, step_id: str) -> None:
    """Remove one ordered step block from a copied runbook."""
    kept: list[str] = []
    skipping = False
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("### Step "):
            skipping = line.startswith(f"### Step {step_id} ")
        if not skipping:
            kept.append(line)
    path.write_text("".join(kept), encoding="utf-8")


def _swap_steps(path: Path, first: str, second: str) -> None:
    """Swap two ordered step blocks in a copied runbook."""
    text = path.read_text(encoding="utf-8")
    blocks = {
        match.group(0).split(" ", 3)[2]: match.group(0) for match in _STEP_BLOCK.finditer(text)
    }
    assert first in blocks and second in blocks
    text = (
        text.replace(blocks[first], "__P4_06_A__", 1)
        .replace(blocks[second], "__P4_06_B__", 1)
        .replace("__P4_06_A__", blocks[second])
        .replace("__P4_06_B__", blocks[first])
    )
    path.write_text(text, encoding="utf-8")


def test_all_runbooks_pass_the_deterministic_check() -> None:
    results, violations = HARNESS.check_all(REPO_ROOT)
    assert not violations
    assert len(results) == 4
    assert sum(result.steps_checked for result in results) == 27
    assert all(result.ok for result in results)


def test_missing_step_fails_closed(tmp_path: Path) -> None:
    root = _copy_runbooks(tmp_path)
    _remove_step(root / "docs" / "runbooks" / "retraining.md", "RT-05")
    _, violations = HARNESS.check_all(root, repo_root=REPO_ROOT)
    assert any(
        violation.step_id == "RT-05" and "missing ordered step" in violation.reason
        for violation in violations
    )


def test_reordered_steps_fail_closed(tmp_path: Path) -> None:
    root = _copy_runbooks(tmp_path)
    _swap_steps(root / "docs" / "runbooks" / "onboarding.md", "ONB-01", "ONB-02")
    _, violations = HARNESS.check_all(root, repo_root=REPO_ROOT)
    assert any("ordered step sequence" in violation.reason for violation in violations)


def test_missing_approval_and_audit_fail_closed(tmp_path: Path) -> None:
    root = _copy_runbooks(tmp_path)
    path = root / "docs" / "runbooks" / "retraining.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "- Approval: lead (required for any real gate/threshold decision; none is made here)\n",
        "- Note: lead (required for any real gate/threshold decision; none is made here)\n",
        1,
    )
    text = text.replace(
        "- Audit: `services/evalgate/gate/audit.py` "
        "(`no threshold` change recorded, zero promotion writes)\n",
        "",
        1,
    )
    path.write_text(text, encoding="utf-8")
    _, violations = HARNESS.check_all(root, repo_root=REPO_ROOT)
    assert any(
        violation.step_id == "RT-07" and "approval" in violation.reason for violation in violations
    )
    assert any(
        violation.step_id == "RT-07" and "audit" in violation.reason for violation in violations
    )


def test_nonexistent_citation_fails_closed(tmp_path: Path) -> None:
    root = _copy_runbooks(tmp_path)
    path = root / "docs" / "runbooks" / "retraining.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "services/evalgate/gate/rules.py::decide_gate",
            "services/evalgate/gate/missing.py::decide_gate",
            1,
        ),
        encoding="utf-8",
    )
    _, violations = HARNESS.check_all(root, repo_root=REPO_ROOT)
    assert any("cited module path does not exist" in violation.reason for violation in violations)


def test_pii_shaped_literals_fail_closed_without_echo(tmp_path: Path) -> None:
    root = _copy_runbooks(tmp_path)
    path = root / "docs" / "runbooks" / "onboarding.md"
    email = "user" + "@example.invalid"
    card = "1234" * 4
    path.write_text(
        path.read_text(encoding="utf-8") + f"\nForbidden probe: {email} {card}\n",
        encoding="utf-8",
    )
    _, violations = HARNESS.check_all(root, repo_root=REPO_ROOT)
    assert any("email-like literal" in violation.reason for violation in violations)
    assert any("card-like digit run" in violation.reason for violation in violations)
    assert all(
        email not in violation.reason and card not in violation.reason for violation in violations
    )


def test_harness_output_records_open_human_acceptance(capsys: pytest.CaptureFixture[str]) -> None:
    assert HARNESS.main(["--root", str(REPO_ROOT)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["human_followthrough_claimed"] is False
    assert payload["release_blocker"] is True
    assert payload["cp3_status"] == "CP-3-pending"
    assert payload["synthetic_label"] == HARNESS.SYNTHETIC_LABEL
    assert "OPEN" in payload["acceptance"]
