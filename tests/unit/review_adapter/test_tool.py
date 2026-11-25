"""Fake tools, tool selection and proof that the tool seam is swappable."""

from __future__ import annotations

import pytest

from services.review_adapter import (
    FAKE_ALTERNATE,
    FAKE_IN_MEMORY,
    AlternateReviewTool,
    InMemoryReviewStore,
    InMemoryReviewTool,
    ReviewAdapterError,
    ReviewAdapterSettings,
    ReviewCoordinator,
    ReviewTool,
    build_review_task,
    build_review_tool,
    build_slot_form,
)
from tests.unit.review_adapter.conftest import CLOCK, queue_item, scripted_review


@pytest.mark.parametrize("tool", [InMemoryReviewTool(), AlternateReviewTool()])
def test_fakes_satisfy_review_tool_protocol(tool: ReviewTool) -> None:
    """Both fake implementations satisfy the runtime-checkable protocol."""
    assert isinstance(tool, ReviewTool)


def test_create_task_is_idempotent_and_tool_ids_differ() -> None:
    """Tools own their ID scheme and reuse it for duplicate pushes."""
    item = queue_item("misheard_surname")
    task = build_review_task(item, build_slot_form())
    first = InMemoryReviewTool()
    second = AlternateReviewTool()
    assert first.create_task(task) == first.create_task(task)
    assert second.create_task(task) == second.create_task(task)
    assert first.create_task(task).startswith("memory:")
    assert second.create_task(task).startswith("alt:")


def test_stage_order_is_enforced_by_fakes() -> None:
    """Stage 2 cannot be prepared before stage 1 has been pulled."""
    item = queue_item("misheard_surname")
    task = build_review_task(item, build_slot_form())
    for tool in (InMemoryReviewTool(), AlternateReviewTool()):
        tool_task_id = tool.create_task(task)
        with pytest.raises(ReviewAdapterError, match="stage 1 must be pulled"):
            tool.prepare_stage2(tool_task_id, {})
        with pytest.raises(ReviewAdapterError, match="unknown tool task"):
            tool.pull_stage1("does-not-exist")


@pytest.mark.parametrize("tool_name", [FAKE_IN_MEMORY, FAKE_ALTERNATE])
def test_swapping_tools_changes_only_tool_module(tool_name: str) -> None:
    """The same coordinator and payload produce the same stored result on both fakes."""
    item = queue_item("misheard_surname")
    script = scripted_review("misheard_surname")
    settings = ReviewAdapterSettings(tool=tool_name)
    tool = build_review_tool(settings, reviews={item.event_id: script}, clock=CLOCK)
    store = InMemoryReviewStore()
    result = ReviewCoordinator(tool, store, settings=settings).run_item(item)
    assert result.tool_name == tool_name
    assert result.to_api_result()["corrected_transcript"] == script.corrected_transcript
    assert result.to_api_result()["routing"] == script.routing
    assert result.reviewer_id == script.reviewer_id
    assert len(store.all()) == 1


def test_default_reviewer_id_setting_is_used_by_fakes() -> None:
    """A script without a reviewer ID falls back to typed settings."""
    item = queue_item("misheard_surname")
    script = scripted_review("misheard_surname", reviewer_id="")
    settings = ReviewAdapterSettings(
        tool=FAKE_IN_MEMORY, default_reviewer_id="reviewer-from-settings"
    )
    tool = build_review_tool(settings, reviews={item.event_id: script}, clock=CLOCK)
    result = ReviewCoordinator(tool, InMemoryReviewStore(), settings=settings).run_item(item)
    assert result.reviewer_id == "reviewer-from-settings"


def test_unknown_tool_fails_closed() -> None:
    """An unimplemented tool name is rejected rather than silently faked."""
    with pytest.raises(ReviewAdapterError, match="unknown review tool"):
        build_review_tool(ReviewAdapterSettings(tool="label_studio_not_implemented"))


def test_factory_default_is_the_offline_fake() -> None:
    """No environment or Docker is needed for the default run."""
    tool = build_review_tool()
    assert isinstance(tool, InMemoryReviewTool)
    assert tool.tool_name() == FAKE_IN_MEMORY
