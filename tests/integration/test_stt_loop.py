"""P3-12.1 fake-only end-to-end STT loop rehearsal.

Synthetic labels -> dataset -> fake Whisper LoRA -> P3-04 conversion/probe ->
P1-07 evaluation -> P3-07 shadow evidence -> P3-08 STT gate -> P3-06 serving
sync -> rollback. Every derived artefact is labelled SYNTHETIC/FIXTURE STAND-IN
and no production seam is contacted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from services.datasets.golden import VerificationReport
from services.evalgate.gate import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    CollectingShadowEvidenceSource,
    Gate,
    GateSettings,
    InMemoryAliasStore,
    PrefixArtifactConverter,
    PromotionAuditLog,
    ShadowEvidenceSettings,
    StaticReportEvaluator,
    StaticShadowEvidenceSource,
    decide_gate,
)
from services.release.shadow import (
    BoundedMirrorSink,
    CollectingAuditSink,
    ShadowMirror,
    ShadowSettings,
    ShadowTurn,
    validate_shadow_report,
)
from services.release.shadow.mirror import SystemClock
from services.release.sync import SyncOutcome

from .helpers import (
    STT_CHALLENGER_VERSION,
    STT_CHAMPION_VERSION,
    STT_RELEASE_LABEL,
    STT_TASK_ID,
    SYNTHETIC_REVIEWER_ID,
    LabelledAuditSink,
    assert_labelled_outputs,
    assert_sync_swapped,
    build_dataset_bundle,
    build_sync_rehearsal,
    build_synthetic_shadow_evidence,
    evaluate_stt_artifact,
    evaluate_stt_promotion_reports,
    make_labelled_refusal_log,
    stt_artifact_settings,
    stt_training_items,
    train_fake_whisper,
    write_derived_eval_fixture,
)
from .helpers import (
    golden_call_ids as sealed_call_ids,
)


def make_shadow_settings() -> ShadowSettings:
    """Return deterministic shadow settings carrying the STT stand-in label."""
    return ShadowSettings(
        task_id=STT_TASK_ID,
        max_pending=8,
        mirror_budget_ms=50.0,
        max_turns_per_drain=8,
        report_status_label=STT_RELEASE_LABEL,
    )


def test_stt_loop_rehearsal_with_shadow_gated_promotion_and_rollback(
    tmp_path: Path,
    fixture_rows: list[dict[str, Any]],
    sealed_golden: VerificationReport,
    read_only_source_hashes: dict[str, str],
) -> None:
    """Rehearse the whole STT loop with fake-only seams and fail-closed gates."""
    golden = sealed_call_ids(sealed_golden)
    assert read_only_source_hashes

    # Synthetic labels only: no sealed golden call is used as training content.
    items = stt_training_items(fixture_rows)
    assert len(items) == len(fixture_rows)
    assert all(item.synthetic for item in items)
    assert {item.call_id for item in items}.isdisjoint(golden)

    dataset = build_dataset_bundle(
        items,
        golden_call_ids=golden,
        output_dir=tmp_path / "dataset",
        prefix="stt-loop",
    )
    assert dataset.build.examples
    assert dataset.manifest["content_hash"]

    # P3-01: fake Whisper LoRA training and conversion are both in-memory.
    training = train_fake_whisper(dataset, golden_call_ids=golden)
    assert training.converted_uri.startswith("converted://")
    assert training.converted_uri != str(training.result["artifact_uri"])
    assert len(training.logger.runs) == 1
    assert training.logger.runs[0].artifact_uri == training.result["artifact_uri"]

    # P3-04: the exact converted artifact (never the checkpoint) is probed and
    # evaluated. Committed fixture and sealed golden hashes stay unchanged.
    artifact_settings = stt_artifact_settings(
        tmp_path,
        checkpoint_uri=str(training.result["artifact_uri"]),
        behaviour="healthy",
    )
    artifact, evaluation = evaluate_stt_artifact(artifact_settings, "healthy")
    assert artifact.uri == evaluation.report["artifact_uri"]
    assert evaluation.probe_suite.status == "pass"
    assert evaluation.gate_summary["status"] == "evaluated"
    assert evaluation.report["decoding_settings"]["release_label"] == STT_RELEASE_LABEL
    assert_labelled_outputs(
        (
            Path(artifact_settings.output_path),
            Path(artifact_settings.output_path).with_name("stt_champion_report.json"),
        )
    )

    # P1-07 paired bootstrap reports drive the P2-09 rule; the artifact safety
    # run above is deliberately read-only and cannot promote by itself.
    derived = write_derived_eval_fixture(tmp_path / "eval" / "derived.jsonl", fixture_rows)
    champion_report, challenger_report = evaluate_stt_promotion_reports(
        derived,
        champion_uri="fake://evalgate/champion-synthetic-fixture-standin",
        challenger_uri=training.converted_uri,
    )
    decision = decide_gate(champion_report, challenger_report, GateSettings())
    assert decision.passed, decision.reasons

    # P3-07: only synthetic-labelled mirrors become comparison evidence.
    shadow = build_synthetic_shadow_evidence(
        tmp_path,
        fixture_rows=fixture_rows,
        candidate_artifact_uri=training.converted_uri,
        champion_artifact_uri=str(champion_report["artifact_uri"]),
    )
    validate_shadow_report(shadow.report)
    assert shadow.report["promotion_decision"] == "not_evaluated"
    assert all(receipt.accepted for receipt in shadow.receipts)
    assert all("SYNTHETIC" in event.status for event in shadow.audit.events)
    assert STT_RELEASE_LABEL in shadow.path.read_text(encoding="utf-8")

    aliases = InMemoryAliasStore(
        {
            CHAMPION_ALIAS: STT_CHAMPION_VERSION,
            CHALLENGER_ALIAS: STT_CHALLENGER_VERSION,
        }
    )
    audit = PromotionAuditLog(tmp_path / "gate" / "audit.jsonl")
    refusal_log = make_labelled_refusal_log(tmp_path / "gate" / "refusals.jsonl")
    source = CollectingShadowEvidenceSource(
        StaticShadowEvidenceSource({(STT_CHALLENGER_VERSION, STT_CHAMPION_VERSION): shadow.report})
    )
    gate = Gate(
        settings=GateSettings(),
        converter=PrefixArtifactConverter(),
        evaluator=StaticReportEvaluator({training.converted_uri: challenger_report}),
        alias_store=aliases,
        audit_log=audit,
        shadow_source=source,
        shadow_settings=ShadowEvidenceSettings(
            refusal_log_path=str(tmp_path / "gate" / "refusals.jsonl")
        ),
        refusal_log=refusal_log,
    )
    result = gate.run(
        str(training.result["artifact_uri"]),
        champion_report,
        challenger_version=STT_CHALLENGER_VERSION,
        actor_id=SYNTHETIC_REVIEWER_ID,
        reason=f"{STT_RELEASE_LABEL}; shadow-gated STT rehearsal",
        change_kind="stt",
    )
    assert result.decision.passed
    assert result.promotion is not None
    assert result.refusal is None
    assert result.shadow_evidence is not None
    assert result.shadow_evidence.synthetic
    assert aliases.get(CHAMPION_ALIAS) == STT_CHALLENGER_VERSION
    assert len(source.requests) == 1
    request = source.requests[0]
    assert request.candidate_artifact_uri == training.converted_uri
    assert request.champion_artifact_uri == champion_report["artifact_uri"]
    assert [entry.action for entry in audit.entries()] == ["promote"]
    assert audit.verify_chain() == ()
    assert refusal_log.delegate.entries() == ()

    # P3-06: the promoted alias is mirrored to both STT and parser serving only
    # after the gate promoted, and each structured event carries the label.
    sync_agent, _, serving, sync_logger = build_sync_rehearsal(
        tmp_path,
        aliases=aliases,
        artifact_uri=training.converted_uri,
        version=STT_CHALLENGER_VERSION,
        champion_artifact_uri=str(champion_report["artifact_uri"]),
    )
    first_tick = sync_agent.tick()
    assert first_tick.outcome is SyncOutcome.SWAPPED
    assert first_tick.task_id == STT_TASK_ID
    assert_sync_swapped(serving, version=STT_CHALLENGER_VERSION)
    assert sync_agent.tick().outcome is SyncOutcome.UNCHANGED
    for event in sync_logger.delegate.events_named("release_sync.tick"):
        assert event.fields["release_label"] == STT_RELEASE_LABEL

    # P3-08/P2-09 rollback is one audited alias step; the sync agent mirrors it.
    rollback = gate.rollback(
        result.promotion,
        actor_id=SYNTHETIC_REVIEWER_ID,
        reason=f"{STT_RELEASE_LABEL}; rehearsal rollback",
    )
    assert rollback.before_version == STT_CHALLENGER_VERSION
    assert rollback.after_version == STT_CHAMPION_VERSION
    assert aliases.get(CHAMPION_ALIAS) == STT_CHAMPION_VERSION
    assert [entry.action for entry in audit.entries()] == ["promote", "rollback"]
    assert audit.verify_chain() == ()
    assert_sync_swapped(serving, version=STT_CHALLENGER_VERSION)
    revert_tick = sync_agent.tick()
    assert revert_tick.outcome is SyncOutcome.SWAPPED
    assert_sync_swapped(serving, version=STT_CHAMPION_VERSION)

    derived_outputs = (
        Path(artifact_settings.output_path),
        Path(artifact_settings.output_path).with_name("stt_champion_report.json"),
        shadow.path,
        Path(tmp_path / "gate" / "audit.jsonl"),
    )
    assert all(STT_RELEASE_LABEL in path.read_text(encoding="utf-8") for path in derived_outputs)


def test_shadow_mirror_with_system_clock_is_bounded(
    fixture_rows: list[dict[str, Any]],
) -> None:
    """A real wall clock still produces a bounded, labelled mirror receipt."""
    clock = SystemClock()
    settings = make_shadow_settings()
    sink = BoundedMirrorSink(clock, settings.max_pending)
    collecting_audit = CollectingAuditSink()
    audit = LabelledAuditSink(collecting_audit)
    mirror = ShadowMirror(settings, sink, clock, audit)
    row = fixture_rows[0]
    turn = ShadowTurn(
        call_id=str(row["call_id"]),
        turn_index=int(row["turn_index"]),
        audio_uri=str(row["audio_path"]),
        synthetic=True,
        observed_at=clock.now(),
    )
    receipt = mirror.mirror(turn)
    assert receipt.accepted
    assert 0.0 <= mirror.stats.max_elapsed_seconds <= settings.mirror_budget_ms / 1000.0
    assert all(STT_RELEASE_LABEL in event.status for event in collecting_audit.events)
