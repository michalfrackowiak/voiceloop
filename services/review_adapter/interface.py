"""Tool-neutral types and protocols for the P2-03 review adapter.

Nothing here imports a review tool, a database driver, a model or a network
client. Real review tools implement :class:`ReviewTool`; real result storage
implements :class:`ReviewStore`. Default implementations are offline fakes and
every run must be labelled as a fixture/fake stand-in, not a human review.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .results import StoredReviewResult

SCHEMA_VERSION = "1.0.0"
"""Version of the ``schemas/review_adapter.openapi.yaml`` payload format."""

ROUTING_LABELS: tuple[str, ...] = (
    "stt_wrong",
    "parser_wrong",
    "both_fine",
    "unusable_audio",
)
"""The only routing labels accepted by the adaptation ladder (context.md §7)."""

RoutingLabel = Literal["stt_wrong", "parser_wrong", "both_fine", "unusable_audio"]
"""Typed alias for :data:`ROUTING_LABELS`."""


class ReviewAdapterError(ValueError):
    """Raised when a review payload or tool response violates the adapter contract."""


@dataclass(frozen=True)
class LowConfidenceSpan:
    """A word-level audio range highlighted for the reviewer, taken verbatim from a source."""

    start_ms: int
    end_ms: int
    text: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        """Return the ``LowConfidenceSpan`` shape from the review-adapter contract."""
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.text,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class SlotField:
    """One form field generated from a property of the booking JSON Schema."""

    name: str
    json_type: str
    required: bool = False
    enum: tuple[str, ...] = ()
    minimum: int | None = None
    description: str = ""
    properties: tuple[SlotField, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the generated field descriptor."""
        return {
            "name": self.name,
            "type": self.json_type,
            "required": self.required,
            "enum": list(self.enum),
            "minimum": self.minimum,
            "description": self.description,
            "properties": [field.to_dict() for field in self.properties],
        }


@dataclass(frozen=True)
class SlotForm:
    """Stage 2 slot form derived from the active booking JSON Schema."""

    schema_version: str
    fields: tuple[SlotField, ...]
    required: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the form sent to the review tool."""
        return {
            "schema_version": self.schema_version,
            "fields": [field.to_dict() for field in self.fields],
            "required": list(self.required),
        }


@dataclass(frozen=True)
class ReviewTask:
    """Both review stages for one queue item, with lineage and tool-neutral payloads."""

    schema_version: str
    item_id: str
    event_id: str
    call_id: str
    turn_index: int
    audio_uri: str
    hypothesis: str
    low_confidence_spans: tuple[LowConfidenceSpan, ...]
    blind: bool
    stage1_instruction: str
    slot_form: SlotForm

    def to_push_payload(self) -> dict[str, Any]:
        """Return exactly the ``ReviewItemCreate`` fields of OpenAPI 1.0.0.

        Tool-specific stage descriptors and lineage stay on the internal object;
        the contract payload has ``additionalProperties: false`` and cannot carry
        them.
        """
        return {
            "schema_version": self.schema_version,
            "call_id": self.call_id,
            "turn_index": self.turn_index,
            "audio_uri": self.audio_uri,
            "hypothesis": self.hypothesis,
            "low_confidence_spans": [span.to_dict() for span in self.low_confidence_spans],
            "blind": self.blind,
        }


@dataclass(frozen=True)
class Stage1Answer:
    """The human-corrected transcript returned by stage 1 of a review tool."""

    corrected_transcript: str


@dataclass(frozen=True)
class Stage2Answer:
    """The reviewer, slots, routing label and timings returned by stage 2."""

    reviewer_id: str
    slots: Mapping[str, Any]
    routing: str
    started_at: datetime
    completed_at: datetime
    warnings: tuple[str, ...] = ()


class LowConfidenceSpanLike(Protocol):
    """Read-only view of a span supplied by a queue item or explanation source."""

    @property
    def start_ms(self) -> int:
        """Return the start offset in the original audio, in milliseconds."""
        ...

    @property
    def end_ms(self) -> int:
        """Return the end offset in the original audio, in milliseconds."""
        ...

    @property
    def text(self) -> str:
        """Return the hypothesis text covered by the span."""
        ...

    @property
    def confidence(self) -> float:
        """Return the word confidence for the span."""
        ...


class QueueItemLike(Protocol):
    """Read-only view of the P2-02 ``QueueItem`` attributes the adapter consumes."""

    @property
    def event_id(self) -> str:
        """Return the source turn-event ID used for lineage and idempotency."""
        ...

    @property
    def call_id(self) -> str:
        """Return the source call ID."""
        ...

    @property
    def turn_index(self) -> int:
        """Return the zero-based turn index within the call."""
        ...

    @property
    def audio_uri(self) -> str:
        """Return the object-store URI of the original audio."""
        ...

    @property
    def hypothesis(self) -> str:
        """Return the STT hypothesis corrected in stage 1."""
        ...

    @property
    def low_confidence_spans(self) -> Sequence[LowConfidenceSpanLike]:
        """Return explicit low-confidence highlights; never synthesise any."""
        ...

    @property
    def no_prefill(self) -> bool:
        """Return whether stage 2 must skip parser prefill (blind item)."""
        ...

    @property
    def training_consent(self) -> bool:
        """Return whether training consent was granted for the source turn."""
        ...


class QueueItemSource(Protocol):
    """Injectable seam that supplies P2-02 queue items without importing P2-02."""

    def iter_items(self) -> Iterator[QueueItemLike]:
        """Iterate over queue items in stable order."""
        ...


class SpanSource(Protocol):
    """Injectable source of review highlights, derived from explicit signals only."""

    def spans_for(self, item: QueueItemLike) -> tuple[LowConfidenceSpan, ...]:
        """Return validated spans for ``item``; an empty tuple means no highlights."""
        ...


class SlotParser(Protocol):
    """Injectable stand-in for the current parser used for stage 2 prefill."""

    def parse(
        self, transcript: str, *, context: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        """Parse a corrected transcript into booking slots."""
        ...


@runtime_checkable
class ReviewTool(Protocol):
    """The only seam that couples the adapter to Label Studio, Argilla or a fake."""

    def tool_name(self) -> str:
        """Return a stable identifier for the selected tool and stand-in status."""
        ...

    def create_task(self, task: ReviewTask) -> str:
        """Push a two-stage task and return a tool-local task ID (idempotent)."""
        ...

    def pull_stage1(self, tool_task_id: str) -> Stage1Answer | None:
        """Return the stage 1 answer, or ``None`` while the reviewer has not finished."""
        ...

    def prepare_stage2(self, tool_task_id: str, prefill: Mapping[str, Any] | None) -> None:
        """Attach the stage 2 slot form and optional parser prefill (``None`` if blind)."""
        ...

    def pull_stage2(self, tool_task_id: str) -> Stage2Answer | None:
        """Return the stage 2 answer, or ``None`` while it is still pending."""
        ...


class ReviewStore(Protocol):
    """Persistence seam for completed results; the in-memory fake stands in for Postgres."""

    def save(self, result: StoredReviewResult) -> StoredReviewResult:
        """Insert ``result`` idempotently and return the stored record."""
        ...

    def get(self, item_id: str) -> StoredReviewResult | None:
        """Return the stored result for ``item_id``, or ``None`` when absent."""
        ...

    def all(self) -> tuple[StoredReviewResult, ...]:
        """Return every stored result in insertion order."""
        ...
