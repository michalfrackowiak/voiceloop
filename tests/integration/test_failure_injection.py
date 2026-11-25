"""P2-13.2 independent failure injections for the synthetic loop rehearsal."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from services.datasets import ReviewedItem
from services.datasets.golden import VerificationReport
from services.datasets.lineage import (
    DeletedCallInBuildError,
    InMemoryTombstoneSink,
    TombstoneRecord,
    guard_manifest,
    pseudonymize_call_id,
)
from services.evalgate import ConfidenceInterval, build_report
from services.evalgate.gate import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    CRITICAL_SLICES,
    Gate,
    GateSettings,
    InMemoryAliasStore,
    PrefixArtifactConverter,
    PromotionAuditLog,
    StaticReportEvaluator,
)
from services.ingest.failopen import (
    BoundedBuffer,
    FailOpenSettings,
    HealthSnapshot,
    RecordingSink,
)

from .helpers import (
    CARD_NUMBER,
    SYNTHETIC_REVIEWER_ID,
    build_dataset_bundle,
    clone_as_fixture_standin,
    exported_text,
    fixture_events,
    train_fake_parser,
)
from .helpers import (
    golden_call_ids as sealed_call_ids,
)


def make_contract_report(
    *,
    artifact_uri: str,
    e2e_accuracy: float,
    e2e_delta_lower: float,
    wer: float,
    wer_delta_lower: float,
) -> dict[str, Any]:
    """Build a contract-valid eval_report with explicit paired delta CIs."""

    def interval(lower: float) -> ConfidenceInterval:
        """Return a finite CI whose upper bound sits above its lower bound."""
        return ConfidenceInterval(lower=lower, upper=lower + 0.02, level=0.95)

    slices = {
        name: {
            "turns": 10.0,
            "wer": float(wer),
            "e2e_booking_accuracy": float(e2e_accuracy),
        }
        for name in CRITICAL_SLICES
    }
    return build_report(
        artifact_uri=artifact_uri,
        eval_sets=["general_capability", "regression"],
        metrics={
            "wer": float(wer),
            "entity_error": 0.1,
            "slot_f1": 0.8,
            "schema_valid_rate": 0.9,
            "e2e_booking_accuracy": float(e2e_accuracy),
        },
        confidence_intervals={
            "wer": interval(wer_delta_lower),
            "entity_error": interval(0.0),
            "slot_f1": interval(0.0),
            "schema_valid_rate": interval(0.0),
            "e2e_booking_accuracy": interval(e2e_delta_lower),
        },
        slice_metrics=slices,
        decoding_settings={
            "hotwords": [],
            "initial_prompt": "",
            "condition_on_previous_text": False,
            "language": "en",
        },
    )


def fixture_source_item(row: dict[str, Any]) -> ReviewedItem:
    """Build a synthetic reviewed stand-in source from one fixture label row."""
    event = fixture_events([row])[0]
    failure_type = str(row.get("failure_type", "both_fine"))
    return ReviewedItem(
        event=event,
        corrected_transcript=str(row["expected_transcript"]),
        slots=dict(row["expected_slots"]),
        routing="both_fine" if failure_type == "clean" else failure_type,
        reviewer_id=SYNTHETIC_REVIEWER_ID,
        failure_type=failure_type,
        synthetic=True,
    )


def test_bad_data_is_excluded_before_training(
    tmp_path: Path,
    fixture_rows: list[dict[str, Any]],
    sealed_golden: VerificationReport,
) -> None:
    """Consent failures and card PII never reach exports or the fake trainer."""
    golden_call_ids = sealed_call_ids(sealed_golden)
    clean_source = fixture_source_item(
        next(row for row in fixture_rows if row.get("failure_type") == "parser_wrong")
    )
    noisy_source = fixture_source_item(
        next(row for row in fixture_rows if row.get("failure_type") == "stt_wrong")
    )
    good_clean = clone_as_fixture_standin(clean_source, call_id="bad-data-good-clean")
    good_noisy = clone_as_fixture_standin(noisy_source, call_id="bad-data-good-noisy")

    # Non-consented row also carries the malformed mapping-key case required by
    # docs/LEAD_BRIEFS.md. Consent is checked first, so it cannot reach an export.
    no_consent = clone_as_fixture_standin(
        clean_source,
        call_id="bad-data-no-consent",
        consent_training=False,
        slots={CARD_NUMBER: "malformed-card-key"},
    )
    # A consented card-bearing string value must be hard-dropped by PII screening.
    card_row = clone_as_fixture_standin(
        noisy_source,
        call_id="bad-data-card-value",
        stt_text=f"Card number {CARD_NUMBER}",
        corrected_transcript=f"Card number {CARD_NUMBER}",
        slots={"contact": {"note": CARD_NUMBER}},
    )

    bundle = build_dataset_bundle(
        [good_clean, good_noisy, no_consent, card_row],
        golden_call_ids=golden_call_ids,
        output_dir=tmp_path / "bad-data-dataset",
        prefix="bad-data",
    )
    assert bundle.build.filter_counts.consent_dropped == 1
    assert bundle.build.filter_counts.pii_dropped == 1
    dropped_calls = {drop.call_id for drop in bundle.build.dropped}
    assert dropped_calls == {no_consent.call_id, card_row.call_id}
    source_calls = {example.call_id for example in bundle.build.examples}
    assert source_calls == {good_clean.call_id, good_noisy.call_id}
    assert source_calls.isdisjoint(golden_call_ids)

    text = exported_text(bundle.exports)
    assert CARD_NUMBER not in text
    assert CARD_NUMBER.replace(" ", "") not in text

    training = train_fake_parser(bundle, golden_call_ids=golden_call_ids)
    assert float(training.result["metrics"]["synthetic_examples"]) > 0
    assert CARD_NUMBER not in bundle.exports.sft_path.read_text(encoding="utf-8")


def test_tombstoned_call_excluded_before_hash_and_guard_fails_closed(
    tmp_path: Path,
    fixture_rows: list[dict[str, Any]],
) -> None:
    """A denied call is excluded pre-hash; an unguarded manifest is rejected."""
    source = fixture_source_item(fixture_rows[1])
    denied_call = "fixture-standin-denied"
    kept_call = "fixture-standin-kept"
    denied_item = clone_as_fixture_standin(source, call_id=denied_call)
    kept_item = clone_as_fixture_standin(source, call_id=kept_call)

    tombstones = InMemoryTombstoneSink()
    tombstones.deny(
        TombstoneRecord(
            call_id=denied_call,
            reason="consent_withdrawn",
            actor_id=SYNTHETIC_REVIEWER_ID,
            role="admin",
            timestamp="2026-11-25T00:00:00Z",
            call_ref=pseudonymize_call_id(denied_call),
            dataset_ids=(),
            content_hashes=(),
            model_ids=(),
        )
    )
    assert tombstones.excluded_call_ids() == frozenset({denied_call})

    guarded = build_dataset_bundle(
        [kept_item, denied_item],
        golden_call_ids=(),
        excluded_call_ids=tombstones.excluded_call_ids(),
        output_dir=tmp_path / "tombstone-guarded",
        prefix="tombstone-guarded",
    )
    guarded_result = guard_manifest(guarded.manifest, tombstones)
    assert guarded_result.blocked_count == 0
    assert guarded_result.content_hash == guarded.manifest["content_hash"]
    assert denied_call not in guarded.manifest["source_call_ids"]
    assert denied_call not in {example.call_id for example in guarded.build.examples}
    assert guarded.build.filter_counts.regression_excluded == 1

    unguarded = build_dataset_bundle(
        [kept_item, denied_item],
        golden_call_ids=(),
        output_dir=tmp_path / "tombstone-unguarded",
        prefix="tombstone-unguarded",
    )
    assert unguarded.manifest["content_hash"] != guarded.manifest["content_hash"]
    assert denied_call in unguarded.manifest["source_call_ids"]
    with pytest.raises(DeletedCallInBuildError) as raised:
        guard_manifest(unguarded.manifest, tombstones)
    assert denied_call not in str(raised.value)
    assert "call_sha256:" in str(raised.value)


@pytest.mark.xfail(
    reason=(
        "Known P2-05/P1-09 integration defect recorded by docs/LEAD_BRIEFS.md: "
        "_iter_json_strings screens mapping values only, so a consented card "
        "number used as a slot key still reaches HF/SFT exports. Tracked as a "
        "lead note; P2-13 does not patch another lane."
    ),
    strict=True,
)
def test_card_number_as_mapping_key_is_still_leaked(
    tmp_path: Path,
    fixture_rows: list[dict[str, Any]],
) -> None:
    """Document the known mapping-key screening gap without hiding it."""
    source = fixture_source_item(fixture_rows[1])
    malformed = clone_as_fixture_standin(
        source,
        call_id="bad-data-mapping-key-leak",
        slots={CARD_NUMBER: "malformed-card-key"},
    )
    bundle = build_dataset_bundle(
        [malformed],
        golden_call_ids=(),
        output_dir=tmp_path / "mapping-key-dataset",
        prefix="mapping-key",
    )
    assert CARD_NUMBER not in exported_text(bundle.exports)


def test_regressing_model_gate_refusal_preserves_aliases_and_audit(
    tmp_path: Path,
) -> None:
    """A better WER is not enough: e2e regression fails closed with no side effects."""
    gate_settings = GateSettings()
    assert gate_settings.min_e2e_margin == 0.01
    assert gate_settings.min_ci_level == 0.95
    assert gate_settings.slice_regression_tolerance == 0.0

    source_uri = "fake://trainer-artifacts/fake-lora/regressing-fixture-standin"
    converted_uri = PrefixArtifactConverter().convert(source_uri).artifact_uri
    champion = make_contract_report(
        artifact_uri="fake://evalgate/champion-synthetic-fixture-standin",
        e2e_accuracy=0.80,
        e2e_delta_lower=0.0,
        wer=0.30,
        wer_delta_lower=0.0,
    )
    regressing = make_contract_report(
        artifact_uri=converted_uri,
        e2e_accuracy=0.70,
        e2e_delta_lower=-0.05,
        wer=0.10,
        wer_delta_lower=-0.20,
    )

    aliases = InMemoryAliasStore(
        {
            CHAMPION_ALIAS: "v3-fixture-standin",
            CHALLENGER_ALIAS: "v4-regressing-fixture-standin",
        }
    )
    audit = PromotionAuditLog(tmp_path / "gate" / "audit.jsonl")
    audit.record(
        actor_id=SYNTHETIC_REVIEWER_ID,
        action="promote",
        reason="SYNTHETIC/FIXTURE STAND-IN baseline record",
        before_version="v2-fixture-standin",
        after_version="v3-fixture-standin",
        task_id="P2-13",
    )
    aliases_before = aliases.snapshot()
    entries_before = audit.entries()
    head_before = audit.head_hash()

    gate = Gate(
        settings=gate_settings,
        converter=PrefixArtifactConverter(),
        evaluator=StaticReportEvaluator({converted_uri: regressing}),
        alias_store=aliases,
        audit_log=audit,
    )
    result = gate.run(
        source_uri,
        champion,
        challenger_version="v4-regressing-fixture-standin",
        actor_id=SYNTHETIC_REVIEWER_ID,
        reason="SYNTHETIC/FIXTURE STAND-IN regression rejection",
    )
    assert not result.decision.passed
    assert result.promotion is None
    assert any("e2e delta lower bound" in reason for reason in result.decision.reasons)
    assert aliases.snapshot() == aliases_before
    assert audit.entries() == entries_before
    assert audit.head_hash() == head_before
    assert audit.verify_chain() == ()


def test_storage_outage_is_bounded_and_visible() -> None:
    """A down storage sink cannot block the caller and is visible in health."""
    settings = FailOpenSettings(
        backend="memory",
        capacity=2,
        enqueue_timeout_ms=25,
        flush_timeout_s=0.05,
        degraded_ratio=0.75,
    )
    sink = RecordingSink(down=True)
    receipts: list[Any] = []
    with BoundedBuffer(sink, settings) as buffer:
        started = time.perf_counter()
        for index in range(3):
            receipts.append(
                buffer.enqueue(
                    {
                        "event_id": f"0000000{index}-0000-4000-8000-000000000000",
                        "call_id": "fixture-standin-outage",
                        "turn_index": index,
                        "stt": {"text": "SYNTHETIC/FIXTURE STAND-IN"},
                    }
                )
            )
        enqueue_elapsed = time.perf_counter() - started
        assert enqueue_elapsed < 0.5
        assert [receipt.accepted for receipt in receipts] == [True, True, False]
        assert receipts[-1].signal == "shed"
        assert receipts[-1].reason == "queue_full"

        started = time.perf_counter()
        flush = buffer.flush(timeout_s=0.05)
        flush_elapsed = time.perf_counter() - started
        assert flush_elapsed < 0.5
        assert flush.flushed == 0
        assert flush.storage_failures >= 1
        assert flush.remaining >= 1
        assert sink.written == []

        health: HealthSnapshot = buffer.health()
        assert health.status == "degraded"
        assert health.dropped >= 1
        assert health.storage_failures >= 1
        assert health.capacity == settings.capacity

        sink.set_down(False)
        recovered = buffer.flush(timeout_s=0.1)
        assert recovered.flushed >= 1
        assert len(sink.written) >= 1
