"""P2-13.1 happy-path loop rehearsal with fake components only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from services.datasets import assert_no_call_in_multiple_splits, content_hash
from services.datasets.golden import VerificationReport
from services.datasets.lineage import (
    FakeManifestSource,
    FakeTrainerLineageSource,
    LineageGraph,
    P205ExampleIdResolver,
)
from services.evalgate.gate import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    Gate,
    GateSettings,
    InMemoryAliasStore,
    PrefixArtifactConverter,
    PromotionAuditLog,
    StaticReportEvaluator,
    decide_gate,
)
from services.triage.queue import InMemoryScoreSource, TriageQueueSettings

from .helpers import (
    SYNTHETIC_REVIEWER_ID,
    TRAIN_GIT_SHA,
    TRAIN_IMAGE_DIGEST,
    build_dataset_bundle,
    build_queue_with_score_source,
    build_review_queue,
    deliver_via_fake_sdk,
    evaluate_converted_artifact,
    fixture_events,
    make_training_standins,
    reviewed_items_from_results,
    run_scripted_review,
    score_delivered_events,
    scripted_reviews,
    train_fake_parser,
    write_derived_eval_fixture,
)
from .helpers import (
    golden_call_ids as sealed_call_ids,
)


def test_fixture_loop_rehearsal_with_manual_promotion(
    tmp_path: Path,
    fixture_rows: list[dict[str, Any]],
    sealed_golden: VerificationReport,
    read_only_source_hashes: dict[str, str],
) -> None:
    """Rehearse ingest -> triage -> review -> dataset -> train -> gate -> promote."""
    golden_call_ids = sealed_call_ids(sealed_golden)
    assert read_only_source_hashes

    # SDK / fake ingest: the transport is the only delivery seam.
    source_events = fixture_events(fixture_rows)
    transport, delivered = deliver_via_fake_sdk(source_events)
    delivered_ids = {str(event["event_id"]) for event in delivered}
    assert transport.call_count == len(delivered) == len(fixture_rows)
    assert delivered_ids == {str(event["event_id"]) for event in source_events}

    # P2-01 scoring -> P2-02 queue with a random slice and blind items.
    store = score_delivered_events(delivered)
    queue_settings = TriageQueueSettings()
    queue_report = build_review_queue(delivered, store, settings=queue_settings)
    queue_ids = {item.event_id for item in queue_report.items}
    assert queue_ids <= delivered_ids
    assert queue_report.stats.random_slice_size >= 1
    assert queue_report.stats.blind_items >= 1
    assert any(item.slice == "random" for item in queue_report.items)
    assert any(item.slice == "failure" for item in queue_report.items)

    random_ids = {item.event_id for item in queue_report.items if item.slice == "random"}
    uniform_scores = InMemoryScoreSource({event_id: 0.0 for event_id in delivered_ids})
    uniform_queue = build_queue_with_score_source(
        delivered,
        uniform_scores,
        settings=queue_settings,
    )
    uniform_random_ids = {item.event_id for item in uniform_queue.items if item.slice == "random"}
    assert random_ids == uniform_random_ids

    # P2-03 two-stage fake review; answers come only from synthetic fixture labels.
    reviews = scripted_reviews(delivered, fixture_rows)
    results = run_scripted_review(queue_report, reviews)
    assert {result.event_id for result in results} == queue_ids
    assert all(result.reviewer_id for result in results)
    assert all(result.started_at <= result.completed_at for result in results)
    assert any(result.blind for result in results)

    # Dataset seam: P2-03 output is joined to its source event, never relabelled.
    reviewed = reviewed_items_from_results(results, delivered, fixture_rows)
    assert {item.source_id for item in reviewed} == {
        f"{result.call_id}:{result.turn_index}" for result in results
    }
    assert all(item.synthetic for item in reviewed)

    # The sealed stand-in covers all six fixture calls, so non-golden stand-in
    # call IDs are derived from the same synthetic reviewed content.
    standins = make_training_standins(reviewed, golden_call_ids=golden_call_ids)
    assert {item.call_id for item in standins}.isdisjoint(golden_call_ids)

    dataset_dir = tmp_path / "dataset"
    bundle = build_dataset_bundle(
        (*reviewed, *standins),
        golden_call_ids=golden_call_ids,
        output_dir=dataset_dir,
    )
    assert bundle.build.filter_counts.golden_excluded == len(reviewed)
    assert bundle.build.filter_counts.consent_dropped == 0
    assert bundle.build.filter_counts.pii_dropped == 0
    assert_no_call_in_multiple_splits(bundle.build.split_plan)
    source_calls = {str(call_id) for call_id in bundle.manifest["source_call_ids"]}
    assert source_calls.isdisjoint(golden_call_ids)
    assert all(example.synthetic for example in bundle.build.examples)

    digest = content_hash(bundle.build.examples)
    assert digest == bundle.manifest["content_hash"]
    assert bundle.manifest["dataset_id"].endswith(digest[:12])
    assert bundle.exports.counts["hf"] == len(bundle.build.examples)

    sft_rows = [
        json.loads(line)
        for line in bundle.exports.sft_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_parser_ids = {
        example.example_id
        for example in bundle.build.examples
        if example.kind in ("parser_clean", "parser_noisy")
    }
    assert {str(row["example_id"]) for row in sft_rows} == expected_parser_ids
    assert {str(row["call_id"]) for row in sft_rows}.isdisjoint(golden_call_ids)

    # P2-06/P2-07 fake parser training: dataset hash is carried into the run.
    training = train_fake_parser(bundle, golden_call_ids=golden_call_ids)
    assert training.result["lineage"]["dataset_hash"] == digest
    assert training.result["lineage"]["git_sha"] == TRAIN_GIT_SHA
    assert training.result["lineage"]["image_digest"] == TRAIN_IMAGE_DIGEST
    logged = training.logger.runs
    assert len(logged) == 1
    assert logged[0].dataset_hash == digest
    assert logged[0].artifact_uri == training.result["artifact_uri"]
    assert training.converted_uri.startswith("converted://")

    # P2-12 lineage: manifest 1.0.0 has no per-example IDs, so the exact P2-05
    # example index is injected explicitly. Both query directions are asserted.
    graph = LineageGraph(
        FakeManifestSource([bundle.manifest]),
        FakeTrainerLineageSource([training.result]),
        example_resolver=P205ExampleIdResolver(
            example.example_id for example in bundle.build.examples
        ),
    )
    for example in bundle.build.examples:
        answer = graph.example_to_call(example.example_id)
        assert answer.example_id == example.example_id
        assert answer.call_id == example.call_id
        assert {dataset.content_hash for dataset in answer.datasets} == {digest}
        assert len(answer.models) == 1
        assert answer.models[0].content_hash == digest
        assert answer.models[0].run_id == training.result["run_id"]
        assert answer.models[0].artifact_uri == training.result["artifact_uri"]
        assert answer.models[0].git_sha == TRAIN_GIT_SHA
        assert answer.models[0].image_digest == TRAIN_IMAGE_DIGEST
        from_call = graph.models_for_call(example.call_id)
        assert from_call == answer.models
        assert {model.content_hash for model in from_call} == {bundle.manifest["content_hash"]}

    # P1-07 evaluator on a derived fixture; the committed fixture stays untouched.
    derived_fixture = write_derived_eval_fixture(
        tmp_path / "eval" / "derived-turns.jsonl",
        fixture_rows,
    )
    champion_report, challenger_report = evaluate_converted_artifact(
        derived_fixture,
        converted_uri=training.converted_uri,
    )
    assert challenger_report["artifact_uri"] == training.converted_uri
    assert set(champion_report["eval_sets"]) == {"general_capability", "regression"}

    # P2-09 gate: no threshold changes, converted artifact only, e2e decision.
    gate_settings = GateSettings()
    assert gate_settings.min_e2e_margin == 0.01
    assert gate_settings.min_ci_level == 0.95
    assert gate_settings.slice_regression_tolerance == 0.0
    decision = decide_gate(champion_report, challenger_report, gate_settings)
    assert decision.passed
    assert decision.e2e_delta_lower is not None
    assert decision.e2e_delta_lower > gate_settings.min_e2e_margin
    assert any("e2e delta lower bound" in reason for reason in decision.reasons)

    aliases = InMemoryAliasStore(
        {
            CHAMPION_ALIAS: "v1-fixture-standin",
            CHALLENGER_ALIAS: "v2-fixture-standin",
        }
    )
    audit = PromotionAuditLog(tmp_path / "gate" / "audit.jsonl")
    gate = Gate(
        settings=gate_settings,
        converter=PrefixArtifactConverter(),
        evaluator=StaticReportEvaluator({training.converted_uri: challenger_report}),
        alias_store=aliases,
        audit_log=audit,
    )
    gate_result = gate.run(
        str(training.result["artifact_uri"]),
        champion_report,
        challenger_version="v2-fixture-standin",
        actor_id=SYNTHETIC_REVIEWER_ID,
        reason="SYNTHETIC/FIXTURE STAND-IN loop rehearsal; not a promotion decision",
    )
    assert gate_result.decision.passed
    assert gate_result.converted_artifact_uri == training.converted_uri
    assert gate_result.promotion is not None
    assert aliases.get(CHAMPION_ALIAS) == "v2-fixture-standin"
    assert [entry.action for entry in audit.entries()] == ["promote"]
    assert audit.verify_chain() == ()

    rollback = gate.rollback(
        gate_result.promotion,
        actor_id=SYNTHETIC_REVIEWER_ID,
        reason="SYNTHETIC/FIXTURE STAND-IN one-step rollback rehearsal",
    )
    assert rollback.before_version == "v2-fixture-standin"
    assert rollback.after_version == "v1-fixture-standin"
    assert aliases.get(CHAMPION_ALIAS) == "v1-fixture-standin"
    assert [entry.action for entry in audit.entries()] == ["promote", "rollback"]
    assert audit.verify_chain() == ()
