"""Review tool implementations and the single tool-selection seam.

Only this module knows how a tool is driven. Two deterministic in-memory fakes
implement :class:`~services.review_adapter.interface.ReviewTool`; neither
starts Docker, opens a socket, decodes audio or performs a human review. All
runs are fixture/fake stand-ins.

A real Label Studio Community or Argilla adapter belongs here too: implement
``create_task``/``pull_stage1``/``prepare_stage2``/``pull_stage2`` against the
tool's REST API and return its ``ReviewResult`` through the existing coordinator.
No other module in ``services/review_adapter`` then changes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from voiceloop_common import Clock, utcnow

from .interface import (
    ReviewAdapterError,
    ReviewTask,
    ReviewTool,
    Stage1Answer,
    Stage2Answer,
)
from .settings import ReviewAdapterSettings

FAKE_IN_MEMORY = "fake_in_memory"
"""Setting value selecting :class:`InMemoryReviewTool`."""

FAKE_ALTERNATE = "fake_alternate"
"""Setting value selecting :class:`AlternateReviewTool` (swap proof)."""


@dataclass(frozen=True)
class ScriptedReview:
    """Fake reviewer answers supplied by a caller; no fixture labels are read here."""

    corrected_transcript: str
    slots: Mapping[str, Any]
    routing: str
    reviewer_id: str = ""
    warnings: tuple[str, ...] = ()
    duration_seconds: float = 0.0


ScriptedReviews = Mapping[str, ScriptedReview]
"""Fake answers keyed by source ``event_id``."""


@dataclass
class _ToolState:
    """Mutable per-task state inside :class:`InMemoryReviewTool`."""

    task: ReviewTask
    stage1_corrected: str | None = None
    prefill: Mapping[str, Any] | None = None
    prefill_ready: bool = False


def _stage2_answer(review: ScriptedReview, clock: Clock, default_reviewer_id: str) -> Stage2Answer:
    """Build the stage 2 answer with deterministic, injected reviewer timestamps."""
    started_at = clock()
    return Stage2Answer(
        reviewer_id=review.reviewer_id or default_reviewer_id,
        slots=dict(review.slots),
        routing=review.routing,
        started_at=started_at,
        completed_at=started_at + timedelta(seconds=max(0.0, review.duration_seconds)),
        warnings=review.warnings,
    )


class InMemoryReviewTool:
    """Deterministic fake tool with an explicit two-stage state machine."""

    def __init__(
        self,
        reviews: ScriptedReviews | None = None,
        *,
        clock: Clock | None = None,
        default_reviewer_id: str = "fixture-reviewer",
    ) -> None:
        """Create an empty fake queue bound to scripted answers and a clock.

        Args:
            reviews: Fake reviewer answers keyed by ``event_id``; when absent a
                stage answer stays pending, which mirrors a real tool.
            clock: Injected UTC clock for deterministic timestamps.
            default_reviewer_id: Reviewer ID used when a script leaves it empty.

        """
        self._reviews = dict(reviews or {})
        self._clock: Clock = clock if clock is not None else utcnow
        self._default_reviewer_id = default_reviewer_id
        self._states: dict[str, _ToolState] = {}
        self._ids_by_item: dict[str, str] = {}

    def tool_name(self) -> str:
        """Identify this stand-in in stored results."""
        return FAKE_IN_MEMORY

    def create_task(self, task: ReviewTask) -> str:
        """Create or reuse the tool task for an adapter item.

        Args:
            task: Tool-neutral review task built from a queue item.

        Returns:
            A stable tool-local task ID; duplicate pushes reuse it.

        """
        existing = self._ids_by_item.get(task.item_id)
        if existing is not None:
            return existing
        tool_task_id = f"memory:{task.item_id}"
        self._states[tool_task_id] = _ToolState(task=task)
        self._ids_by_item[task.item_id] = tool_task_id
        return tool_task_id

    def pull_stage1(self, tool_task_id: str) -> Stage1Answer | None:
        """Return the scripted corrected transcript, or ``None`` while pending."""
        state = self._state(tool_task_id)
        review = self._reviews.get(state.task.event_id)
        if review is None:
            return None
        state.stage1_corrected = review.corrected_transcript
        return Stage1Answer(corrected_transcript=review.corrected_transcript)

    def prepare_stage2(self, tool_task_id: str, prefill: Mapping[str, Any] | None) -> None:
        """Attach stage 2 prefill (``None`` for blind items) after stage 1."""
        state = self._state(tool_task_id)
        if state.stage1_corrected is None:
            raise ReviewAdapterError("stage 1 must be pulled before stage 2 is prepared")
        state.prefill = dict(prefill) if prefill is not None else None
        state.prefill_ready = True

    def pull_stage2(self, tool_task_id: str) -> Stage2Answer | None:
        """Return the scripted stage 2 answer, or ``None`` while pending."""
        state = self._state(tool_task_id)
        if not state.prefill_ready:
            raise ReviewAdapterError("stage 2 prefill has not been prepared")
        review = self._reviews.get(state.task.event_id)
        if review is None:
            return None
        return _stage2_answer(review, self._clock, self._default_reviewer_id)

    def _state(self, tool_task_id: str) -> _ToolState:
        """Return tool state or raise for an unknown task ID."""
        try:
            return self._states[tool_task_id]
        except KeyError as exc:
            raise ReviewAdapterError(f"unknown tool task: {tool_task_id}") from exc


class AlternateReviewTool:
    """Second fake tool with different IDs and storage but the same behaviour.

    It exists so tests can prove that swapping tools needs no changes outside
    ``tool.py``: the coordinator sees only the ``ReviewTool`` protocol.
    """

    def __init__(
        self,
        reviews: ScriptedReviews | None = None,
        *,
        clock: Clock | None = None,
        default_reviewer_id: str = "fixture-reviewer",
    ) -> None:
        """Create an empty alternate fake queue.

        Args:
            reviews: Fake reviewer answers keyed by ``event_id``.
            clock: Injected UTC clock for deterministic timestamps.
            default_reviewer_id: Reviewer ID used when a script leaves it empty.

        """
        self._reviews = dict(reviews or {})
        self._clock: Clock = clock if clock is not None else utcnow
        self._default_reviewer_id = default_reviewer_id
        self._ids: dict[str, str] = {}
        self._tasks: dict[str, ReviewTask] = {}
        self._stage1: dict[str, str] = {}
        self._prefills: dict[str, Mapping[str, Any] | None] = {}

    def tool_name(self) -> str:
        """Identify this second stand-in in stored results."""
        return FAKE_ALTERNATE

    def create_task(self, task: ReviewTask) -> str:
        """Register the task under an alternate ID scheme, idempotently."""
        known = self._ids.get(task.item_id)
        if known is not None:
            return known
        tool_task_id = f"alt:{task.item_id}"
        self._ids[task.item_id] = tool_task_id
        self._tasks[tool_task_id] = task
        return tool_task_id

    def pull_stage1(self, tool_task_id: str) -> Stage1Answer | None:
        """Pull the scripted stage 1 answer from this tool's own state."""
        review = self._reviews.get(self._task(tool_task_id).event_id)
        if review is None:
            return None
        self._stage1[tool_task_id] = review.corrected_transcript
        return Stage1Answer(corrected_transcript=review.corrected_transcript)

    def prepare_stage2(self, tool_task_id: str, prefill: Mapping[str, Any] | None) -> None:
        """Record the form prefill for a blind or prefilled stage 2 task."""
        self._task(tool_task_id)
        if tool_task_id not in self._stage1:
            raise ReviewAdapterError("stage 1 must be pulled before stage 2 is prepared")
        self._prefills[tool_task_id] = dict(prefill) if prefill is not None else None

    def pull_stage2(self, tool_task_id: str) -> Stage2Answer | None:
        """Pull the scripted stage 2 answer from this tool's own state."""
        self._task(tool_task_id)
        if tool_task_id not in self._prefills:
            raise ReviewAdapterError("stage 2 prefill has not been prepared")
        review = self._reviews.get(self._task(tool_task_id).event_id)
        if review is None:
            return None
        return _stage2_answer(review, self._clock, self._default_reviewer_id)

    def _task(self, tool_task_id: str) -> ReviewTask:
        """Return a task by alternate tool ID or raise."""
        try:
            return self._tasks[tool_task_id]
        except KeyError as exc:
            raise ReviewAdapterError(f"unknown tool task: {tool_task_id}") from exc


def build_review_tool(
    settings: ReviewAdapterSettings | None = None,
    *,
    reviews: ScriptedReviews | None = None,
    clock: Clock | None = None,
) -> ReviewTool:
    """Select a review tool from typed settings; the only tool-choice seam.

    A real adapter (Label Studio Community or Argilla) is added here and
    selected by a new setting value; callers and the coordinator stay unchanged.

    Args:
        settings: Typed adapter settings; defaults to environment-backed values.
        reviews: Fake reviewer answers for the in-memory stand-ins.
        clock: Injected UTC clock for deterministic fake timestamps.

    Returns:
        A tool implementing the ``ReviewTool`` protocol.

    Raises:
        ReviewAdapterError: If the configured tool name is unknown.

    """
    config = settings or ReviewAdapterSettings()
    if config.tool == FAKE_IN_MEMORY:
        return InMemoryReviewTool(
            reviews, clock=clock, default_reviewer_id=config.default_reviewer_id
        )
    if config.tool == FAKE_ALTERNATE:
        return AlternateReviewTool(
            reviews, clock=clock, default_reviewer_id=config.default_reviewer_id
        )
    raise ReviewAdapterError(f"unknown review tool: {config.tool!r}")
