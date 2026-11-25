"""Read-only fixture/golden integrity checks for the P4-02 synthetic lane."""

from __future__ import annotations

import json

from infra.pipelines.drift import (
    EXPECTED_FIXTURE_SHA256,
    EXPECTED_GOLDEN_CONTENT_HASH,
    SYNTHETIC_LABEL,
    DriftSettings,
    InMemoryAuditSink,
    ManualOverride,
    RecordingPipeline,
    build_fixture_replay_sources,
    run_once,
    verify_fixture_hashes,
)
from services.datasets.golden.seal import verify_seal
from voiceloop_common import Clock

GOLDEN_ARTIFACTS = "services/datasets/golden/artifacts"

SENSITIVE_TOKENS = (
    "Ada Fixture",
    "Sam Sample",
    "Jordan Example",
    "Morgan Mock",
    "Casey Placeholder",
    "Riley Synthetic",
    "ada.fixture@example.com",
    "+1 555 010 0001",
    "4111111111111111",
    ".wav",
)


def test_sealed_golden_and_fixture_hashes_are_unchanged() -> None:
    before = verify_fixture_hashes()
    report = verify_seal(GOLDEN_ARTIFACTS)
    report.assert_ok()
    assert report.manifest is not None
    assert report.manifest.content_hash == EXPECTED_GOLDEN_CONTENT_HASH
    after = verify_fixture_hashes()
    assert (
        before
        == after
        == {
            "fixture_sha256_ok": True,
            "golden_content_hash_ok": True,
        }
    )


def test_fixture_replay_report_has_no_pii_or_audio_references(fixed_clock: Clock) -> None:
    settings = DriftSettings(min_new_verified=2)
    scores, outcomes = build_fixture_replay_sources(settings=settings)
    sink = InMemoryAuditSink()
    report = run_once(
        score_source=scores,
        outcome_source=outcomes,
        pipeline=RecordingPipeline(),
        new_verified_count=2,
        settings=settings,
        manual_override=ManualOverride("operator-1a2b3c4d", "lead_authorized"),
        audit_sink=sink,
        clock=fixed_clock,
    )
    payload = json.dumps({"report": report.to_dict(), "audit": sink.entries}, sort_keys=True)
    assert SYNTHETIC_LABEL in payload
    for token in SENSITIVE_TOKENS:
        assert token not in payload
    assert EXPECTED_FIXTURE_SHA256 not in payload
