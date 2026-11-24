"""Tests for the contract index documentation."""

from pathlib import Path
from typing import Any

CONTRACT_FILES = (
    "trainer_contract.schema.json",
    "eval_report.schema.json",
    "dataset_manifest.schema.json",
    "review_adapter.openapi.yaml",
)


def test_contracts_doc_references_every_contract(repo_root: Path) -> None:
    """docs/contracts.md names all four contract files and their versions."""
    text = (repo_root / "docs" / "contracts.md").read_text(encoding="utf-8")
    for name in CONTRACT_FILES:
        assert name in text
    assert text.count("1.0.0") >= len(CONTRACT_FILES)


def test_contracts_doc_describes_structured_errors(repo_root: Path) -> None:
    """The documented HTTP error shape has code, message and details."""
    text = (repo_root / "docs" / "contracts.md").read_text(encoding="utf-8")
    assert "Structured errors" in text
    for field in ("code", "message", "details"):
        assert f'"{field}"' in text


def test_contracts_doc_names_primary_metric(repo_root: Any) -> None:
    """The promotion metric is called out explicitly for downstream agents."""
    text = (repo_root / "docs" / "contracts.md").read_text(encoding="utf-8")
    assert "primary promotion metric" in text
