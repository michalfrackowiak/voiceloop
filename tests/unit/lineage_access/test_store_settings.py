"""P2-12: typed settings and fail-closed injectable fakes."""

from __future__ import annotations

import pytest

from services.datasets.lineage import (
    FakeManifestSource,
    FakeTrainerLineageSource,
    FixedClock,
    InMemoryAuditSink,
    InMemoryTombstoneSink,
    LineageSettings,
    TombstoneRecord,
    pseudonymize_call_id,
)
from tests.unit.lineage_access.conftest import FIXED_NOW
from voiceloop_common import ValidationError


def test_settings_use_only_the_lineage_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    defaults = LineageSettings()
    assert defaults.pseudonym_domain == "voiceloop-lineage-v1"
    assert defaults.audit_log_path == "reports/lineage/lineage_audit.jsonl"
    assert defaults.task_id == "P2-12"

    monkeypatch.setenv("VOICELOOP_LINEAGE_PSEUDONYM_DOMAIN", "test-domain")
    monkeypatch.setenv("VOICELOOP_LINEAGE_AUDIT_LOG_PATH", "tmp/lineage.jsonl")
    monkeypatch.setenv("VOICELOOP_DATASETS_REPLAY_SIZE", "99")
    settings = LineageSettings()
    assert settings.pseudonym_domain == "test-domain"
    assert settings.audit_log_path == "tmp/lineage.jsonl"


def test_fake_manifest_source_validates_and_copies(manifest: dict[str, object]) -> None:
    source = FakeManifestSource([manifest])
    stored = source.manifests()[0]
    assert stored == manifest
    stored["content_hash"] = "mutated"  # type: ignore[index]
    assert source.manifests()[0] == manifest

    with pytest.raises(ValidationError):
        FakeManifestSource([{"content_hash": "incomplete"}])


def test_fake_trainer_source_requires_complete_lineage(
    trainer_results: tuple[dict[str, object], ...],
) -> None:
    source = FakeTrainerLineageSource(trainer_results)
    assert len(source.results()) == 2
    assert source.model_id_for_run("run-parser-v1") is None

    missing_lineage = {
        "schema_version": "1.0.0",
        "contract_part": "output",
        "artifact_uri": "fake://artifacts/no-lineage",
        "metrics": {},
        "run_id": "run-no-lineage",
    }
    with pytest.raises(ValidationError, match="dataset_hash"):
        FakeTrainerLineageSource([missing_lineage])


def test_tombstone_sink_is_append_only_and_exposes_exclusions() -> None:
    sink = InMemoryTombstoneSink()
    record = TombstoneRecord(
        call_id="call-x",
        reason="erasure_request",
        actor_id="admin-1",
        role="admin",
        timestamp="2026-01-02T03:04:05Z",
        call_ref=pseudonymize_call_id("call-x"),
        dataset_ids=("dataset-x",),
        content_hashes=("hash-x",),
        model_ids=("model-x",),
    )
    sink.deny(record)
    sink.deny(record)

    assert sink.is_denied("call-x")
    assert sink.excluded_call_ids() == frozenset({"call-x"})
    assert sink.records() == (record,)
    assert sink.record_for("call-x") == record
    assert not sink.is_denied("call-y")


def test_fixed_clock_and_audit_chain_are_deterministic() -> None:
    clock = FixedClock(FIXED_NOW)
    assert clock.now().isoformat() == "2026-01-02T03:04:05+00:00"
    sink = InMemoryAuditSink()
    assert sink.events() == ()
    assert sink.head_hash() == ""
    assert sink.verify_chain() == ()
