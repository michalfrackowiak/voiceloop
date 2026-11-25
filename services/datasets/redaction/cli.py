"""CLI for synthetic fixture redaction and fake retention rehearsals.

Run with ``python -m services.datasets.redaction.cli``. Both subcommands use
only deterministic fake audio, in-memory stores and read-only fixture files;
no object store, network or speech recognition is contacted. Errors are
structured JSON on stderr with stable codes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from pydantic import ValidationError as PydanticValidationError

from services.datasets.filters import filter_by_consent
from voiceloop_common import ConfigError, StorageError, ValidationError

from .audio import (
    AudioStore,
    FakeAudio,
    InMemoryAudioStore,
    assert_no_unredacted_span,
    redact_audio,
)
from .retention import InMemoryRetentionStore, asset_from_dict, run_retention
from .settings import (
    COMPLIANCE_STATUS,
    POLICY_STATUS,
    SYNTHETIC_LABEL,
    AuditSink,
    Clock,
    FixedClock,
    InMemoryAuditSink,
    RedactionSettings,
    SystemClock,
)
from .spans import (
    CardNumberRejected,
    P109TranscriptDetector,
    P109WordSpanDetector,
    RedactionError,
    TimestampSpan,
    word_timings_from_text,
)
from .text import pseudonymise_text

REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve(path: str | Path) -> Path:
    """Resolve a repo-relative path without mutating fixture URIs."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from disk, failing closed on malformed input."""
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read JSON at {path}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ConfigError(f"expected a JSON object at {path}")
    return decoded


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSON Lines objects from disk, skipping blank lines."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigError(f"cannot read JSONL at {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid JSON on line {number} of {path}: {exc}") from exc
        if not isinstance(decoded, dict):
            raise ConfigError(f"line {number} of {path} must be an object")
        rows.append(decoded)
    return rows


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write one sorted JSON report."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot write JSON at {path}: {exc}") from exc


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Write sorted JSON Lines records."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    except OSError as exc:
        raise ConfigError(f"cannot write JSONL at {path}: {exc}") from exc


def _build_parser() -> argparse.ArgumentParser:
    """Build the redaction CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m services.datasets.redaction.cli",
        description=(
            "Synthetic/fake stand-in for PII audio redaction, transcript "
            "pseudonymisation and raw-audio retention rehearsal."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    redact = commands.add_parser("redact", help="Redact the read-only synthetic fixtures.")
    redact.add_argument("--fixtures", metavar="PATH", help="Labels JSONL path override.")
    redact.add_argument("--output", metavar="DIR", help="Report output directory override.")

    retention = commands.add_parser("retention", help="Run the fake retention job.")
    retention.add_argument("--assets", required=True, metavar="PATH", help="Asset metadata JSON.")
    retention.add_argument("--now", metavar="ISO8601", help="Injected UTC time override.")
    retention.add_argument("--output", metavar="DIR", help="Report output directory override.")
    return parser


def _settings_with_overrides(base: RedactionSettings, updates: dict[str, str]) -> RedactionSettings:
    """Rebuild typed settings so CLI overrides still obey field validators."""
    return RedactionSettings.model_validate({**base.model_dump(), **updates})


def _known_names(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return reviewed guest names from one synthetic label row."""
    slots = row.get("expected_slots")
    if not isinstance(slots, dict):
        return ()
    name = slots.get("guest_name")
    return (str(name),) if isinstance(name, str) and name else ()


def _audio_spans_to_dicts(spans: Sequence[TimestampSpan]) -> list[dict[str, str | int]]:
    """Return PII-free span dictionaries for reports."""
    return [span.to_dict() for span in spans]


def _run_redact(
    args: argparse.Namespace,
    settings: RedactionSettings,
    store: AudioStore,
    audit: AuditSink,
    clock: Clock,
    out: TextIO,
) -> int:
    """Redact synthetic fixtures and write PII-free transcripts and a report."""
    updates = {
        key: value
        for key, value in (
            ("fixtures_path", args.fixtures),
            ("output_dir", args.output),
        )
        if value is not None
    }
    config = _settings_with_overrides(settings, updates)
    rows = _read_jsonl(_resolve(config.fixtures_path))
    output_dir = _resolve(config.output_dir)

    transcript_rows: list[dict[str, object]] = []
    drop_rows: list[dict[str, object]] = []
    consent_drop_rows: list[dict[str, object]] = []
    total_spans = 0
    verified_text = 0
    for row in rows:
        for field in ("call_id", "turn_index", "event_path", "audio_path", "expected_transcript"):
            if field not in row:
                raise ConfigError(f"fixture row missing {field}")
        path = _resolve(str(row["event_path"]))
        event = _read_json(path)
        # Consent gate first, using the read-only P1-09 API. A non-consented
        # row never reaches audio redaction or transcript output (§2 #8).
        kept_events, consent_drops = filter_by_consent([event])
        if not kept_events:
            dropped = consent_drops[0]
            consent_drop_rows.append(
                {
                    "call_id": str(row["call_id"]),
                    "turn_index": row["turn_index"],
                    "event_id": dropped.event_id,
                    "audio_uri": dropped.audio_uri,
                    "reasons": [str(reason) for reason in dropped.reasons],
                    "synthetic": True,
                }
            )
            continue
        audio_block = event.get("audio")
        duration_ms = audio_block.get("duration_ms") if isinstance(audio_block, dict) else None
        default_rate = audio_block.get("sample_rate_hz") if isinstance(audio_block, dict) else None
        rate = row.get("sample_rate_hz", default_rate)
        if not isinstance(duration_ms, int) or not isinstance(rate, int) or rate <= 0:
            raise ConfigError(f"fixture row has invalid audio metadata: {path}")

        call_id = str(row["call_id"])
        turn_index = row["turn_index"]
        text = str(row["expected_transcript"])
        urn = str(row["audio_path"])
        names = _known_names(row)
        words = word_timings_from_text(text, duration_ms)
        detector = P109WordSpanDetector()
        spans = detector.detect_spans(text, words, known_names=names)

        fake = FakeAudio.synthetic(urn, rate, duration_ms, seed=int(turn_index) + 1)
        if isinstance(store, InMemoryAudioStore):
            store.put_raw(fake)
        audio_result = redact_audio(
            fake, spans, settings=config, store=store, audit=audit, clock=clock
        )
        assert_no_unredacted_span(audio_result.audio, spans)
        total_spans += len(spans)

        try:
            safe = pseudonymise_text(
                text,
                call_id=call_id,
                settings=config,
                known_names=names,
                audit=audit,
                clock=clock,
            )
        except CardNumberRejected as exc:
            drop_rows.append(
                {
                    "call_id": call_id,
                    "turn_index": turn_index,
                    "reason": exc.code,
                    "details": exc.to_dict()["details"],
                    "synthetic": True,
                }
            )
            continue
        residual = P109TranscriptDetector().detect(safe.text)
        if residual:
            raise RedactionError(
                "unverified_text_pii",
                "redacted fixture transcript still contains detected PII",
                {"residual_findings": len(residual)},
            )
        verified_text += 1
        transcript_rows.append(
            {
                "call_id": call_id,
                "turn_index": turn_index,
                "text": safe.text,
                "audio_uri": urn,
                "redacted_uri": audio_result.redacted_uri,
                "audio_spans": _audio_spans_to_dicts(spans),
                "synthetic": True,
            }
        )

    audit_events = audit.to_dicts() if isinstance(audit, InMemoryAuditSink) else []
    report: dict[str, object] = {
        "fixtures_path": config.fixtures_path,
        "turns_processed": len(rows),
        "turns_redacted": len(transcript_rows),
        "turns_dropped": len(drop_rows),
        "turns_consent_dropped": len(consent_drop_rows),
        "consent_drops": consent_drop_rows,
        "audio_spans_detected": total_spans,
        "audio_spans_remaining": 0,
        "text_pii_spans_remaining": 0,
        "texts_verified": verified_text,
        "audit_events": len(audit_events),
        "policy_status": POLICY_STATUS,
        "compliance_status": COMPLIANCE_STATUS,
        "synthetic": True,
        "label": SYNTHETIC_LABEL,
    }
    _write_jsonl(output_dir / "redacted_transcripts.jsonl", transcript_rows)
    _write_jsonl(output_dir / "hard_drops.jsonl", drop_rows)
    _write_jsonl(output_dir / "audit_events.jsonl", audit_events)
    _write_json(output_dir / "redaction_report.json", report)
    print(json.dumps(report, sort_keys=True), file=out)
    return 0


def _run_retention(
    args: argparse.Namespace,
    settings: RedactionSettings,
    audit: AuditSink,
    clock: Clock,
    out: TextIO,
) -> int:
    """Run the fake retention job from a metadata-only asset JSON file."""
    updates = {"output_dir": args.output} if args.output is not None else {}
    config = _settings_with_overrides(settings, updates)
    asset_path = _resolve(args.assets)
    try:
        decoded = json.loads(asset_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read asset metadata at {asset_path}: {exc}") from exc
    if not isinstance(decoded, list):
        raise ConfigError("asset metadata must be a JSON array")

    store = InMemoryRetentionStore()
    for index, item in enumerate(decoded):
        if not isinstance(item, dict):
            raise ConfigError(f"asset {index} must be an object")
        store.add(asset_from_dict(item))

    run_clock = clock
    if args.now is not None:
        try:
            run_clock = FixedClock(datetime.fromisoformat(args.now))
        except ValueError as exc:
            raise ConfigError(f"--now must be ISO-8601: {exc}") from exc
    report = run_retention(store, settings=config, clock=run_clock, audit=audit)
    payload = report.to_dict()
    if config.output_dir:
        _write_json(_resolve(config.output_dir) / "retention_report.json", payload)
    print(json.dumps(payload, sort_keys=True), file=out)
    return 0


def _error_code(exc: Exception) -> str:
    """Map a known exception to a stable structured error code."""
    if isinstance(exc, RedactionError):
        return exc.code
    if isinstance(exc, ConfigError) or isinstance(exc, PydanticValidationError):
        return "config_error"
    if isinstance(exc, ValidationError):
        return "validation_error"
    if isinstance(exc, StorageError):
        return "storage_error"
    if isinstance(exc, OSError):
        return "io_error"
    return "internal_error"


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: RedactionSettings | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    audio_store: AudioStore | None = None,
    audit: AuditSink | None = None,
    clock: Clock | None = None,
) -> int:
    """Run a redaction or retention subcommand.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.
        settings: Injected typed settings, primarily for tests.
        stdout: Injected output stream.
        stderr: Injected structured-error stream.
        audio_store: Injected fake audio store; defaults to in-memory.
        audit: Injected PII-free audit sink; defaults to in-memory.
        clock: Injected UTC clock; defaults to the shared system clock.

    Returns:
        0 on success, 2 on a structured configuration/validation failure.

    """
    args = _build_parser().parse_args(argv)
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    config = settings or RedactionSettings()
    store = audio_store if audio_store is not None else InMemoryAudioStore()
    audit_sink = audit if audit is not None else InMemoryAuditSink()
    audit_clock = clock if clock is not None else SystemClock()
    try:
        if args.command == "redact":
            return _run_redact(args, config, store, audit_sink, audit_clock, out)
        return _run_retention(args, config, audit_sink, audit_clock, out)
    except (
        RedactionError,
        ConfigError,
        ValidationError,
        StorageError,
        PydanticValidationError,
        OSError,
    ) as exc:
        error: dict[str, object] = {
            "code": _error_code(exc),
            "message": str(exc),
        }
        if isinstance(exc, RedactionError):
            error["details"] = exc.to_dict()["details"]
        print(json.dumps(error, sort_keys=True), file=err)
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised via python -m
    raise SystemExit(main())
