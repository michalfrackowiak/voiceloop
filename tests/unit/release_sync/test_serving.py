"""P3-06: STT and parser share one transactional swap protocol."""

from __future__ import annotations

from pathlib import Path

import pytest

from services.release.sync import (
    DownloadedArtifact,
    FakeServingAdapter,
    FakeServingStack,
    ServingStack,
    ServingTarget,
    SwapPlan,
)
from voiceloop_common import ValidationError

HASH = "c" * 64
ARTIFACT = DownloadedArtifact("fake://mlflow/v2", Path("fake-temp-dir"))


def _plan() -> SwapPlan:
    return SwapPlan(version="v2", artifact_hash=HASH, before=())


def test_prepare_then_commit_activates_both_targets() -> None:
    stack = FakeServingStack("v1", "v1")

    prepared = stack.prepare(_plan(), ARTIFACT)
    assert all(state.staged_version == "v2" for state in stack.states())
    assert all(state.current_version == "v1" for state in stack.states())

    committed = stack.commit(prepared)
    assert committed.prepared == (ServingTarget.STT, ServingTarget.PARSER)
    assert committed.committed == (ServingTarget.STT, ServingTarget.PARSER)
    assert all(state.current_version == "v2" for state in stack.states())
    assert all(state.staged_version is None for state in stack.states())


def test_prepare_failure_restores_previous_versions() -> None:
    stack = FakeServingStack("stt-old", "parser-old")
    stt = stack.fake_adapters[ServingTarget.STT]
    parser = stack.fake_adapters[ServingTarget.PARSER]
    parser.fail_prepare = True

    with pytest.raises(RuntimeError):
        stack.prepare(_plan(), ARTIFACT)

    assert stt.rollback_calls == 1
    assert [state.current_version for state in stack.states()] == ["stt-old", "parser-old"]
    assert all(state.staged_version is None for state in stack.states())


def test_commit_failure_rolls_back_committed_and_staged_targets() -> None:
    stack = FakeServingStack("v1", "v1")
    stt = stack.fake_adapters[ServingTarget.STT]
    parser = stack.fake_adapters[ServingTarget.PARSER]
    parser.fail_commit = True

    prepared = stack.prepare(_plan(), ARTIFACT)
    with pytest.raises(RuntimeError):
        stack.commit(prepared)

    assert stt.commit_calls == 1
    assert stt.rollback_calls == 1
    assert parser.rollback_calls == 1
    assert [state.current_version for state in stack.states()] == ["v1", "v1"]
    assert all(state.staged_version is None for state in stack.states())


def test_rollback_runs_in_reverse_order() -> None:
    order: list[str] = []

    class RecordingAdapter(FakeServingAdapter):
        def rollback(self) -> None:
            """Record rollback order, then delegate to the fake."""
            order.append(self.target.value)
            super().rollback()

    adapters = {
        ServingTarget.STT: RecordingAdapter(ServingTarget.STT, "v1"),
        ServingTarget.PARSER: RecordingAdapter(ServingTarget.PARSER, "v1"),
    }
    stack = ServingStack(adapters)
    prepared = stack.prepare(_plan(), ARTIFACT)
    committed = stack.commit(prepared)

    assert stack.rollback(committed) == ()
    assert order == ["parser", "stt"]


def test_rollback_errors_are_reported_not_raised() -> None:
    stack = FakeServingStack("v1", "v1")
    stack.fake_adapters[ServingTarget.PARSER].fail_rollback = True
    plan = SwapPlan(
        version="v2",
        artifact_hash=HASH,
        before=(),
        prepared=(ServingTarget.STT, ServingTarget.PARSER),
        committed=(ServingTarget.STT, ServingTarget.PARSER),
    )

    assert stack.rollback(plan) == ("parser:RuntimeError",)


def test_rollback_without_prepared_targets_is_a_noop() -> None:
    stack = FakeServingStack("v1", "v1")

    assert stack.rollback(_plan()) == ()
    assert stack.fake_adapters[ServingTarget.STT].rollback_calls == 0
    assert [state.current_version for state in stack.states()] == ["v1", "v1"]


def test_stack_requires_both_targets() -> None:
    with pytest.raises(ValidationError):
        ServingStack({ServingTarget.STT: FakeServingAdapter(ServingTarget.STT)})
    with pytest.raises(ValidationError):
        ServingStack(
            {
                ServingTarget.STT: FakeServingAdapter(ServingTarget.PARSER),
                ServingTarget.PARSER: FakeServingAdapter(ServingTarget.PARSER),
            }
        )


def test_adapter_rejects_commit_without_prepare() -> None:
    adapter = FakeServingAdapter(ServingTarget.STT, "v1")

    with pytest.raises(ValidationError):
        adapter.commit()
    assert adapter.state().current_version == "v1"
