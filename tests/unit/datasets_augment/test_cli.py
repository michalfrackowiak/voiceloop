"""One-command CLI, exclusion proof and read-only hash verification tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from services.datasets.augment.cli import main
from services.datasets.augment.manifest import validate_manifest
from services.datasets.augment.settings import STAND_IN_LABEL

GOLDEN_DIR = Path("services/datasets/golden/artifacts")
FIXTURES = Path("tests/fixtures/turns.jsonl")
FORBIDDEN_TEXT = ("Ada Fixture", "ada.fixture@example.com", "+1 555", "Mr. Smith")


def _file_hashes(paths: list[Path]) -> dict[str, str]:
    """Hash a set of read-only files before and after a CLI run."""
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _golden_content_hash() -> str:
    """Read the sealed content hash without touching the sealed files."""
    return str(
        json.loads((GOLDEN_DIR / "golden.v1.json").read_text(encoding="utf-8"))["content_hash"]
    )


def test_cli_writes_tagged_manifest_and_leaves_inputs_unchanged(
    source_manifest: dict[str, Any],
    golden_ids: tuple[str, ...],
    regression_ids: tuple[str, ...],
    tmp_path: Path,
    capsys: Any,
) -> None:
    input_path = tmp_path / "p2-05-manifest.json"
    input_path.write_text(json.dumps(source_manifest, sort_keys=True), encoding="utf-8")
    output_path = tmp_path / "augment" / "manifest.json"
    golden_files = sorted(GOLDEN_DIR.glob("*"))
    before = _file_hashes([*golden_files, FIXTURES])
    golden_hash_before = _golden_content_hash()

    code = main(
        [
            "--manifest",
            str(input_path),
            "--output",
            str(output_path),
            "--golden-ids",
            ",".join(golden_ids),
            "--regression-ids",
            ",".join(regression_ids),
            "--duration-ms",
            "80",
            "--seed",
            "3",
            "--noise-level",
            "4",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert output_path.exists()
    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    assert manifest["synthetic"] is True
    assert manifest["counts"]["variants"] > 0
    assert not set(manifest["source_call_ids"]) & (set(golden_ids) | set(regression_ids))
    assert STAND_IN_LABEL in captured.out
    assert "not_evaluated" in captured.out
    assert all(text not in captured.out for text in FORBIDDEN_TEXT)
    serialized = output_path.read_text(encoding="utf-8")
    assert all(text not in serialized for text in FORBIDDEN_TEXT)
    assert '"samples"' not in serialized
    assert _file_hashes([*golden_files, FIXTURES]) == before
    assert (
        _golden_content_hash()
        == golden_hash_before
        == ("662cdaf43ee2d5fde03faf5e69fde196114d4f312406d94c00bf53714d03da8e")
    )
    assert hashlib.sha256(FIXTURES.read_bytes()).hexdigest() == (
        "aca23a303954d035245a97f78319df3694d63611b717a2ff2131bc2e0a4453fd"
    )


def test_cli_defaults_to_sealed_golden_exclusion(
    source_manifest: dict[str, Any],
    tmp_path: Path,
    capsys: Any,
) -> None:
    input_path = tmp_path / "p2-05-manifest.json"
    input_path.write_text(json.dumps(source_manifest, sort_keys=True), encoding="utf-8")
    output_path = tmp_path / "sealed.json"
    code = main(
        [
            "--manifest",
            str(input_path),
            "--output",
            str(output_path),
            "--duration-ms",
            "80",
        ]
    )
    assert code == 0, capsys.readouterr().err
    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert manifest["counts"]["variants"] == 0
    assert manifest["source_call_ids"] == []
    assert set(manifest["evaluation_set_exclusion"]["golden_call_ids"]) == set(
        source_manifest["source_call_ids"]
    )


def test_cli_accepts_empty_golden_list_for_fixture_rehearsal(
    source_manifest: dict[str, Any],
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "p2-05-manifest.json"
    input_path.write_text(json.dumps(source_manifest, sort_keys=True), encoding="utf-8")
    output_path = tmp_path / "injected.json"
    code = main(
        [
            "--manifest",
            str(input_path),
            "--output",
            str(output_path),
            "--golden-ids",
            "",
            "--regression-ids",
            "call-0001",
            "--duration-ms",
            "80",
        ]
    )
    assert code == 0
    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert "call-0001" not in manifest["source_call_ids"]
    assert "call-0002" in manifest["source_call_ids"]
    assert manifest["evaluation_set_exclusion"]["golden_call_ids"] == []


def test_cli_reports_structured_errors(tmp_path: Path, capsys: Any) -> None:
    output_path = tmp_path / "never.json"
    code = main(["--manifest", "", "--output", str(output_path), "--duration-ms", "80"])
    assert code == 2
    assert not output_path.exists()
    error = json.loads(capsys.readouterr().err)
    assert error["code"] == "config_error"
    assert "message" in error

    bad_fixtures = tmp_path / "turns.jsonl"
    bad_fixtures.write_text("tampered\n", encoding="utf-8")
    manifest_path = tmp_path / "p2-05.json"
    manifest_path.write_text("{}", encoding="utf-8")
    code = main(
        [
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--fixtures",
            str(bad_fixtures),
            "--duration-ms",
            "80",
        ]
    )
    assert code == 2
    error = json.loads(capsys.readouterr().err)
    assert error["code"] == "validation_error"
    assert not output_path.exists()
