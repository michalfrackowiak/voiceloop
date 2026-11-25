"""CLI rehearsal tests: reports, no-PII fixtures and structured errors."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path

from services.datasets.filters import detect_pii
from services.datasets.redaction.cli import main
from services.datasets.redaction.settings import (
    FixedClock,
    InMemoryAuditSink,
    RedactionSettings,
)


def _settings(output: Path) -> RedactionSettings:
    """Return deterministic settings with output redirected to a temp dir."""
    return RedactionSettings(
        padding_ms=10,
        retention_days=7,
        pseudonym_salt="cli-test-salt",
        output_dir=str(output),
    )


def test_cli_redact_writes_pii_free_fixture_reports(
    tmp_path: Path, settings: RedactionSettings
) -> None:
    out = io.StringIO()
    err = io.StringIO()
    audit = InMemoryAuditSink()
    output = tmp_path / "redaction"
    config = RedactionSettings(
        padding_ms=settings.padding_ms,
        retention_days=settings.retention_days,
        pseudonym_salt=settings.pseudonym_salt,
        output_dir=str(output),
    )

    code = main(
        ["redact", "--output", str(output)],
        settings=config,
        stdout=out,
        stderr=err,
        audit=audit,
        clock=FixedClock(datetime(2026, 3, 1, 12, 0, tzinfo=UTC)),
    )

    assert code == 0
    assert err.getvalue() == ""
    report = json.loads((output / "redaction_report.json").read_text(encoding="utf-8"))
    assert report["turns_processed"] == 36
    assert report["turns_redacted"] == 35
    assert report["turns_dropped"] == 0
    assert report["turns_consent_dropped"] == 1
    assert report["audio_spans_remaining"] == 0
    assert report["text_pii_spans_remaining"] == 0
    assert report["synthetic"] is True
    assert json.loads(out.getvalue()) == report

    transcripts = [
        json.loads(line)
        for line in (output / "redacted_transcripts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(transcripts) == 35
    for row in transcripts:
        assert detect_pii(row["text"]) == ()
        assert row["synthetic"] is True
        assert row["redacted_uri"].startswith("fake-redacted://")

    audit_rows = (output / "audit_events.jsonl").read_text(encoding="utf-8")
    assert "PERSON_" not in audit_rows
    assert "audit_events" in report


def test_consent_denied_row_is_absent_from_all_jsonl_outputs(tmp_path: Path) -> None:
    """call-0005/turn-2 must never become training data or a redacted URI."""
    output = tmp_path / "redaction"
    code = main(
        ["redact", "--output", str(output)],
        settings=_settings(output),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        clock=FixedClock(datetime(2026, 3, 1, 12, 0, tzinfo=UTC)),
    )
    assert code == 0

    report = json.loads((output / "redaction_report.json").read_text(encoding="utf-8"))
    assert report["turns_processed"] == 36
    assert report["turns_redacted"] == 35
    assert report["turns_consent_dropped"] == 1
    drops = report["consent_drops"]
    assert len(drops) == 1
    assert drops[0]["call_id"] == "call-0005"
    assert drops[0]["turn_index"] == 2
    assert drops[0]["reasons"] == ["training_consent_denied"]
    assert "text" not in drops[0]
    assert "redacted_uri" not in drops[0]

    jsonl_files = sorted(output.glob("*.jsonl"))
    assert {path.name for path in jsonl_files} == {
        "redacted_transcripts.jsonl",
        "hard_drops.jsonl",
        "audit_events.jsonl",
    }
    for path in jsonl_files:
        content = path.read_text(encoding="utf-8")
        assert "fake-redacted://tests/fixtures/audio/call-0005/turn-002.wav" not in content
        for line in content.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            assert not (row.get("call_id") == "call-0005" and row.get("turn_index") == 2), (
                f"consent-denied training row leaked into {path.name}"
            )

    transcripts = [
        json.loads(line)
        for line in (output / "redacted_transcripts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(transcripts) == 35
    assert all(row["call_id"] != "call-0005" or row["turn_index"] != 2 for row in transcripts)


def test_cli_retention_writes_fake_report(tmp_path: Path) -> None:
    assets = tmp_path / "assets.json"
    assets.write_text(
        json.dumps(
            [
                {
                    "uri": "fake://expired",
                    "kind": "raw_audio",
                    "captured_at": "2026-01-01T00:00:00+00:00",
                },
                {
                    "uri": "fake://fresh",
                    "kind": "raw_audio",
                    "captured_at": "2026-02-28T00:00:00+00:00",
                },
                {
                    "uri": "fake://redacted",
                    "kind": "redacted_audio",
                    "captured_at": "2020-01-01T00:00:00+00:00",
                },
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "retention"
    out = io.StringIO()
    err = io.StringIO()

    code = main(
        [
            "retention",
            "--assets",
            str(assets),
            "--now",
            "2026-03-01T00:00:00+00:00",
            "--output",
            str(output),
        ],
        settings=_settings(output),
        stdout=out,
        stderr=err,
        clock=FixedClock(datetime(2026, 3, 1, tzinfo=UTC)),
    )

    assert code == 0
    assert err.getvalue() == ""
    report = json.loads((output / "retention_report.json").read_text(encoding="utf-8"))
    assert report["deleted"] == 1
    assert report["kept"] == 2
    assert report["label"].startswith("SYNTHETIC")
    assert "no real object store" in report["deletion_claim"]
    assert json.loads(out.getvalue()) == report


def test_cli_returns_structured_error_for_missing_assets(tmp_path: Path) -> None:
    out = io.StringIO()
    err = io.StringIO()
    code = main(
        ["retention", "--assets", str(tmp_path / "missing.json")],
        settings=_settings(tmp_path / "out"),
        stdout=out,
        stderr=err,
        clock=None,
    )
    assert code == 2
    assert out.getvalue() == ""
    payload = json.loads(err.getvalue())
    assert payload["code"] == "config_error"
    assert "missing.json" in payload["message"]
