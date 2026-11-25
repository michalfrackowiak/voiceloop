"""P2-03 review adapter: tool-neutral review tasks, results and fake backends.

Everything in this package is offline by default. The checked-in fakes are
fixture/fake stand-ins: no human reviewed the data, no review tool was started,
and no Postgres row was written. See ``README.md`` for the real-tool seam.
"""

from __future__ import annotations

from .interface import (
    ROUTING_LABELS,
    SCHEMA_VERSION,
    LowConfidenceSpan,
    QueueItemLike,
    QueueItemSource,
    ReviewAdapterError,
    ReviewStore,
    ReviewTask,
    ReviewTool,
    RoutingLabel,
    SlotField,
    SlotForm,
    SlotParser,
    SpanSource,
    Stage1Answer,
    Stage2Answer,
)
from .results import InMemoryReviewStore, StoredReviewResult
from .settings import ReviewAdapterSettings
from .tasks import (
    STAGE1_INSTRUCTION,
    DictSlotParser,
    ExplanationSpanSource,
    QueueItemSpanSource,
    QueueReportSource,
    ReviewCoordinator,
    SequenceQueueSource,
    SimpleQueueItem,
    build_review_task,
    build_slot_form,
)
from .tool import (
    FAKE_ALTERNATE,
    FAKE_IN_MEMORY,
    AlternateReviewTool,
    InMemoryReviewTool,
    ScriptedReview,
    ScriptedReviews,
    build_review_tool,
)

__all__ = [
    "FAKE_ALTERNATE",
    "FAKE_IN_MEMORY",
    "ROUTING_LABELS",
    "SCHEMA_VERSION",
    "STAGE1_INSTRUCTION",
    "AlternateReviewTool",
    "DictSlotParser",
    "ExplanationSpanSource",
    "InMemoryReviewStore",
    "InMemoryReviewTool",
    "LowConfidenceSpan",
    "QueueItemLike",
    "QueueItemSpanSource",
    "QueueItemSource",
    "QueueReportSource",
    "ReviewAdapterError",
    "ReviewAdapterSettings",
    "ReviewCoordinator",
    "ReviewStore",
    "ReviewTask",
    "ReviewTool",
    "RoutingLabel",
    "ScriptedReview",
    "ScriptedReviews",
    "SequenceQueueSource",
    "SimpleQueueItem",
    "SlotField",
    "SlotForm",
    "SlotParser",
    "SpanSource",
    "Stage1Answer",
    "Stage2Answer",
    "StoredReviewResult",
    "build_review_task",
    "build_review_tool",
    "build_slot_form",
]
