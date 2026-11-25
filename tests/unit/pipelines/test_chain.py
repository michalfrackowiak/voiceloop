"""Tests for the deterministic P2-10 chain and fake fixture steps."""

from __future__ import annotations

from typing import Any

import pytest

from infra.pipelines.chain import (
    PipelineStepError,
    StepRecord,
    TriggerNotSatisfied,
    build_fixture_steps,
    read_fixture_labels,
    run_chain,
    verified_example_ids,
)
from infra.pipelines.settings import PipelineSettings
from infra.pipelines.triggers import TriggerDecision, TriggerReason
from voiceloop_common import validate

from .conftest import (
    DATASET_HASH,
    EVAL_RESULT,
    FIXED_NOW,
    MANIFEST,
    TRAIN_RESULT,
    make_recording_steps,
)

TRIGGER = TriggerDecision(
    triggered=True,
    reasons=(TriggerReason.MANUAL,),
    new_verified_count=10,
    threshold=2,
    drift_alarm=False,
    manual=True,
    checked_at=FIXED_NOW,
)


def test_success_records_full_lineage_and_step_order(fixed_clock: Any) -> None:
    recorder = make_recording_steps()
    record = run_chain(
        steps=recorder.steps,
        trigger=TRIGGER,
        eval_sets=("general_capability",),
        clock=fixed_clock,
    )

    assert record.status == "success"
    assert [step.name for step in record.steps] == ["dataset", "train", "evaluate"]
    assert record.lineage is not None
    lineage = record.lineage.to_dict()
    assert lineage["dataset_hash"] == DATASET_HASH
    assert lineage["trainer_run_id"] == "run-test-1"
    assert lineage["eval_report_path"] == "fake://evalgate/report.json"
    assert lineage["artifact_uri"] == "fake://artifact/deployable"
    assert lineage["git_sha"] == "1" * 40
    assert record.trigger["reason"] == "manual_override"
    assert recorder.train_manifests == [MANIFEST]
    assert recorder.eval_calls == [("fake://artifact/deployable", ("general_capability",))]
    assert record.started_at == FIXED_NOW.isoformat()
    assert record.finished_at == FIXED_NOW.isoformat()


def test_unwrapped_dataset_manifest_is_supported(fixed_clock: Any) -> None:
    recorder = make_recording_steps()
    record = run_chain(
        steps=recorder.steps,
        trigger=TRIGGER,
        clock=fixed_clock,
    )
    assert record.status == "success"
    assert record.verified_ids == ()


def test_fixture_steps_validate_contracts_and_split_by_call(
    pipeline_settings: PipelineSettings,
    fixed_clock: Any,
) -> None:
    steps = build_fixture_steps(pipeline_settings, clock=fixed_clock)
    dataset_result = steps.dataset()
    manifest = dataset_result["manifest"]
    validate(manifest, "dataset_manifest.schema.json")
    assert manifest["synthetic"] is True
    split = manifest["splits"]
    assert set(split["train"]).isdisjoint(split["val"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["val"]).isdisjoint(split["test"])
    assert set(sum(split.values(), [])) == set(manifest["source_call_ids"])
    ids = verified_example_ids(read_fixture_labels(pipeline_settings.fixtures_path))
    assert list(dataset_result["verified_ids"]) == list(ids)

    record = run_chain(steps=steps, trigger=TRIGGER, clock=fixed_clock)
    assert record.status == "success"
    assert record.lineage is not None
    assert record.lineage.dataset_hash == manifest["content_hash"]
    eval_summary = next(step.summary for step in record.steps if step.name == "evaluate")
    assert eval_summary["report_path"] == pipeline_settings.eval_report_path


@pytest.mark.parametrize("fail_at", ["dataset", "train", "evaluate"])
def test_step_failure_stops_chain_and_is_reported(fail_at: str, fixed_clock: Any) -> None:
    recorder = make_recording_steps(fail_at=fail_at)
    with pytest.raises(PipelineStepError) as caught:
        run_chain(steps=recorder.steps, trigger=TRIGGER, clock=fixed_clock)

    error = caught.value
    assert error.step == fail_at
    assert fail_at in str(error)
    assert isinstance(error.__cause__, RuntimeError)
    assert error.record is not None
    assert error.record.status == "failed"
    assert error.record.error is not None and "boom" in error.record.error
    expected_steps = {"dataset": 0, "train": 1, "evaluate": 2}[fail_at]
    assert len(error.record.steps) == expected_steps
    assert error.record.lineage is None
    assert len(recorder.dataset_calls) == 1
    assert len(recorder.train_manifests) == (0 if fail_at == "dataset" else 1)
    assert len(recorder.eval_calls) == (1 if fail_at == "evaluate" else 0)


def test_trainer_lineage_mismatch_stops_chain(fixed_clock: Any) -> None:
    bad_train_result = {
        **TRAIN_RESULT,
        "lineage": {**TRAIN_RESULT["lineage"], "dataset_hash": "b" * 64},
    }
    recorder = make_recording_steps(train_result=bad_train_result)
    with pytest.raises(PipelineStepError) as caught:
        run_chain(steps=recorder.steps, trigger=TRIGGER, clock=fixed_clock)
    assert caught.value.step == "train"
    assert "does not match manifest" in str(caught.value)
    assert recorder.eval_calls == []


def test_evaluating_wrong_artifact_stops_chain(fixed_clock: Any) -> None:
    recorder = make_recording_steps(eval_result={**EVAL_RESULT, "artifact_uri": "fake://wrong"})
    with pytest.raises(PipelineStepError) as caught:
        run_chain(steps=recorder.steps, trigger=TRIGGER, clock=fixed_clock)
    assert caught.value.step == "evaluate"
    assert "not the trained artifact" in str(caught.value)


def test_missing_trainer_lineage_stops_chain(fixed_clock: Any) -> None:
    recorder = make_recording_steps(train_result={"run_id": "r", "artifact_uri": "fake://a"})
    with pytest.raises(PipelineStepError) as caught:
        run_chain(steps=recorder.steps, trigger=TRIGGER, clock=fixed_clock)
    assert caught.value.step == "train"


def test_untriggered_chain_does_not_call_any_step(fixed_clock: Any) -> None:
    recorder = make_recording_steps()
    not_triggered = TriggerDecision(
        triggered=False,
        reasons=(),
        new_verified_count=0,
        threshold=2,
        drift_alarm=False,
        manual=False,
        checked_at=FIXED_NOW,
    )
    with pytest.raises(TriggerNotSatisfied) as caught:
        run_chain(steps=recorder.steps, trigger=not_triggered, clock=fixed_clock)
    assert caught.value.record is not None
    assert caught.value.record.status == "skipped"
    assert recorder.dataset_calls == []
    assert recorder.train_manifests == []
    assert recorder.eval_calls == []


def test_repeated_runs_with_frozen_clock_are_identical(fixed_clock: Any) -> None:
    first = run_chain(steps=make_recording_steps().steps, trigger=TRIGGER, clock=fixed_clock)
    second = run_chain(steps=make_recording_steps().steps, trigger=TRIGGER, clock=fixed_clock)
    assert first.to_dict() == second.to_dict()


def test_step_record_serialises() -> None:
    assert StepRecord("dataset", {"examples": 2}).summary == {"examples": 2}
