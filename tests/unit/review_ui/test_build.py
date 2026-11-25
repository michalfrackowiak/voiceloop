"""Static artifact build and offline CLI tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ui import cli
from ui.build import build_artifacts, render_index
from ui.go_no_go import read_json, write_json_report


def test_build_artifacts_writes_deterministic_payloads(tmp_path: Path) -> None:
    """Rebuilding with the same inputs produces byte-identical artifacts."""
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    index_a = tmp_path / "a.html"
    index_b = tmp_path / "b.html"
    decision = read_json(Path(__file__).parents[3] / "ui" / "data" / "go_no_go_report.json")

    first = build_artifacts(output_dir=out_a, decision=decision, index_path=index_a)
    build_artifacts(output_dir=out_b, decision=decision, index_path=index_b)

    assert set(first) == {
        "booking.form.json",
        "review_state.json",
        "go_no_go_report.json",
        "index.html",
    }
    for name in first:
        if name == "index.html":
            assert index_a.read_bytes() == index_b.read_bytes()
        else:
            assert (out_a / name).read_bytes() == (out_b / name).read_bytes()


def test_rendered_index_has_no_placeholders_or_external_sources() -> None:
    """The browser shell embeds only static, pseudonymous state."""
    root = Path(__file__).parents[3]
    decision = read_json(root / "ui" / "data" / "go_no_go_report.json")
    states: list[dict[str, Any]] = json.loads(
        (root / "ui" / "data" / "review_state.json").read_text(encoding="utf-8")
    )

    html = render_index(states, decision)

    assert "{{" not in html
    assert "<audio" not in html
    assert "fetch(" not in html
    assert "http://" not in html
    assert "https://" not in html


def test_cli_verify_hashes_and_go_no_go_reports_status(
    p2_04_report: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI prints pinned hash checks and a fail-closed decision."""
    metrics_path = tmp_path / "p2_04_metrics.json"
    output_path = tmp_path / "go_no_go_report.json"
    write_json_report(dict(p2_04_report), metrics_path)

    code = cli.main(
        [
            "--verify-hashes",
            "--go-no-go",
            "--metrics",
            str(metrics_path),
            "--output",
            str(output_path),
        ]
    )
    captured = capsys.readouterr()

    assert code == 0
    assert '"fixture_hash_ok": true' in captured.out
    assert '"decision": "no_go_fail_closed"' in captured.out
    assert "STATUS: NOT DONE" in captured.out
    assert read_json(output_path)["decision"] == "no_go_fail_closed"


def test_cli_missing_metrics_fails_closed(capsys: pytest.CaptureFixture[str]) -> None:
    """A go/no-go request without metrics returns a structured validation error."""
    code = cli.main(["--go-no-go"])
    captured = capsys.readouterr()

    assert code == 2
    assert json.loads(captured.err)["code"] == "validation_error"
