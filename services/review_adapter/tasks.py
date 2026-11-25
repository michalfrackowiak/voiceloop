"""Two-stage review task construction and orchestration for P2-03.

Queue items arrive through :class:`~services.review_adapter.interface.QueueItemSource`
and are never modified. Stage 1 returns a corrected transcript; stage 2 is a slot
form generated from the active booking JSON Schema and prefilled by an injected
parser on that corrected transcript, except for blind items. Tool choice lives
only in ``tool.py``; this module only knows the :class:`ReviewTool` protocol.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from schemas.booking import BookingValidationError, load_booking_schema, validate_booking

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
    SlotField,
    SlotForm,
    SlotParser,
    SpanSource,
    Stage2Answer,
)
from .results import InMemoryReviewStore, StoredReviewResult
from .settings import ReviewAdapterSettings

STAGE1_INSTRUCTION = (
    "Listen to the original audio and correct the STT hypothesis. "
    "Low-confidence word spans are highlighted for reference."
)
"""Fixed stage 1 instruction; the audio URI is passed by reference only."""


@dataclass(frozen=True)
class SimpleQueueItem:
    """Minimal concrete queue item for callers without a P2-02 queue object."""

    event_id: str
    call_id: str
    turn_index: int
    audio_uri: str
    hypothesis: str
    low_confidence_spans: tuple[LowConfidenceSpan, ...] = ()
    no_prefill: bool = False
    training_consent: bool = False
    signals: tuple[str, ...] = ()


class SequenceQueueSource:
    """Adapt any sequence of P2-02-style queue items into a :class:`QueueItemSource`."""

    def __init__(self, items: Iterable[QueueItemLike]) -> None:
        """Freeze the supplied items into a stable sequence.

        Args:
            items: Queue items in processing order.

        """
        self._items = tuple(items)

    def iter_items(self) -> Iterator[QueueItemLike]:
        """Iterate over the frozen queue items."""
        return iter(self._items)


class QueueReportSource:
    """Read the ``items`` attribute of a P2-02 ``QueueReport`` without importing P2-02."""

    def __init__(self, report: object) -> None:
        """Validate and freeze the report's items.

        Args:
            report: An object exposing an ``items`` sequence.

        Raises:
            ReviewAdapterError: If ``items`` is missing or is not a sequence.

        """
        items = getattr(report, "items", None)
        if not isinstance(items, Sequence):
            raise ReviewAdapterError("queue report must expose an items sequence")
        self._items = tuple(items)

    def iter_items(self) -> Iterator[QueueItemLike]:
        """Iterate over the report's queue items."""
        return iter(self._items)


class QueueItemSpanSource:
    """Use the explicit low-confidence spans already carried by a P2-02 queue item."""

    def spans_for(self, item: QueueItemLike) -> tuple[LowConfidenceSpan, ...]:
        """Validate and return the item's spans, without inventing any."""
        return _coerce_spans(item.low_confidence_spans)


class ExplanationSpanSource:
    """Prefer spans from a P2-01-style store, falling back to the queue item.

    P2-01 ``FiredSignal`` currently carries only name, weight, severity and
    reason; it has no audio timings. A future explanation that does carry
    ``low_confidence_spans`` is used verbatim, otherwise the queue item's
    explicit word-confidence spans are used. No spans are ever synthesised.
    """

    def __init__(self, store: object) -> None:
        """Bind a store exposing ``get(event_id)``.

        Args:
            store: A P2-01-style explanation store or any compatible fake.

        Raises:
            ReviewAdapterError: If the store has no callable ``get``.

        """
        getter = getattr(store, "get", None)
        if not callable(getter):
            raise ReviewAdapterError("explanation store must expose get(event_id)")
        self._get: Callable[[str], Any] = getter

    def spans_for(self, item: QueueItemLike) -> tuple[LowConfidenceSpan, ...]:
        """Return explanation spans when present, else the queue item's spans."""
        explanation = self._get(item.event_id)
        raw = (
            getattr(explanation, "low_confidence_spans", None) if explanation is not None else None
        )
        if raw:
            return _coerce_spans(raw)
        return QueueItemSpanSource().spans_for(item)


class DictSlotParser:
    """Offline parser stand-in that looks up slots by normalised transcript.

    Unknown transcripts yield an empty slot object, which the coordinator
    records as a ``parser_prefill_empty`` warning instead of fabricating slots.
    Replace this via the ``parser`` argument for the current production parser.
    """

    def __init__(self, by_transcript: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        """Index optional transcript-to-slots examples.

        Args:
            by_transcript: Exact (whitespace/case-normalised) transcript keys.

        """
        self._by_transcript = {
            str(key).strip().lower(): dict(value) for key, value in (by_transcript or {}).items()
        }

    def parse(
        self, transcript: str, *, context: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        """Return a copy of the slots registered for ``transcript``, or ``{}``.

        Args:
            transcript: The corrected transcript from stage 1.
            context: Ignored by this fake; accepted for parser protocol parity.

        """
        return dict(self._by_transcript.get(transcript.strip().lower(), {}))


def _coerce_spans(values: Iterable[Any]) -> tuple[LowConfidenceSpan, ...]:
    """Validate explicit span data from a mapping or attribute-based object.

    Args:
        values: Span records; nothing is generated when this is empty.

    Returns:
        Immutable, contract-valid spans in source order.

    Raises:
        ReviewAdapterError: If a span has missing or invalid data.

    """
    spans: list[LowConfidenceSpan] = []
    for value in values:
        if isinstance(value, Mapping):
            start, end = value.get("start_ms"), value.get("end_ms")
            text, confidence = value.get("text", ""), value.get("confidence")
        else:
            start = getattr(value, "start_ms", None)
            end = getattr(value, "end_ms", None)
            text = getattr(value, "text", "")
            confidence = getattr(value, "confidence", None)
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise ReviewAdapterError("low-confidence span needs a non-negative start_ms")
        if isinstance(end, bool) or not isinstance(end, int) or end < start:
            raise ReviewAdapterError("low-confidence span needs end_ms >= start_ms")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ReviewAdapterError("low-confidence span needs a numeric confidence")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ReviewAdapterError("low-confidence span confidence must be in [0, 1]")
        spans.append(
            LowConfidenceSpan(
                start_ms=start,
                end_ms=end,
                text=str(text),
                confidence=float(confidence),
            )
        )
    return tuple(spans)


def _slot_field(name: str, spec: Any, required: bool) -> SlotField:
    """Build one form field descriptor from a JSON Schema property."""
    if not isinstance(spec, Mapping):
        raise ReviewAdapterError(f"booking schema property {name!r} must be an object")
    raw_type = spec.get("type", "string")
    json_type = (
        "|".join(str(part) for part in raw_type)
        if isinstance(raw_type, (list, tuple))
        else str(raw_type)
    )
    raw_enum = spec.get("enum", ())
    enum = tuple(str(value) for value in raw_enum) if isinstance(raw_enum, (list, tuple)) else ()
    minimum = spec.get("minimum")
    raw_properties = spec.get("properties", {})
    properties = (
        tuple(
            _slot_field(str(child), child_spec, False)
            for child, child_spec in raw_properties.items()
        )
        if isinstance(raw_properties, Mapping)
        else ()
    )
    return SlotField(
        name=name,
        json_type=json_type,
        required=required,
        enum=enum,
        minimum=minimum if isinstance(minimum, int) and not isinstance(minimum, bool) else None,
        description=str(spec.get("description", "")),
        properties=properties,
    )


def build_slot_form(schema_path: str | Path | None = None) -> SlotForm:
    """Generate the stage 2 form from the active booking JSON Schema.

    Args:
        schema_path: Optional project-specific schema path; the loader's
            ``VOICELOOP_BOOKING_SCHEMA`` override still applies when unset.

    Returns:
        A form with every top-level property, its required flag and nested
        object properties. No slot name or enum is hard-coded here.

    Raises:
        ReviewAdapterError: If the schema has no object properties.

    """
    schema = load_booking_schema(schema_path)
    properties = schema.get("properties")
    if not isinstance(properties, Mapping) or not properties:
        raise ReviewAdapterError("booking schema needs a non-empty properties object")
    raw_required = schema.get("required", ())
    required = tuple(str(name) for name in raw_required) if isinstance(raw_required, list) else ()
    required_set = set(required)
    fields = tuple(
        _slot_field(str(name), spec, str(name) in required_set) for name, spec in properties.items()
    )
    return SlotForm(
        schema_version=str(schema.get("schema_version", SCHEMA_VERSION)),
        fields=fields,
        required=required,
    )


def build_review_task(
    item: QueueItemLike,
    slot_form: SlotForm,
    settings: ReviewAdapterSettings | None = None,
    span_source: SpanSource | None = None,
) -> ReviewTask:
    """Build a tool-neutral two-stage task from a read-only queue item.

    Args:
        item: A P2-02 queue item or compatible view.
        slot_form: The booking-schema-derived stage 2 form.
        settings: Accepted for settings parity; the form is already built from
            the active schema path by the coordinator.
        span_source: Source of explicit highlights; defaults to queue-item spans.

    Returns:
        The immutable review task with an adapter-local item ID.

    Raises:
        ReviewAdapterError: If required lineage or audio fields are malformed.

    """
    source = span_source or QueueItemSpanSource()
    event_id, call_id = item.event_id, item.call_id
    if not isinstance(event_id, str) or not event_id:
        raise ReviewAdapterError("queue item needs a non-empty event_id")
    if not isinstance(call_id, str) or not call_id:
        raise ReviewAdapterError("queue item needs a non-empty call_id")
    if isinstance(item.turn_index, bool) or not isinstance(item.turn_index, int):
        raise ReviewAdapterError("queue item needs an integer turn_index")
    if item.turn_index < 0:
        raise ReviewAdapterError("queue item turn_index must be non-negative")
    if not isinstance(item.audio_uri, str) or not item.audio_uri:
        raise ReviewAdapterError("queue item needs a non-empty audio_uri")
    if not isinstance(item.hypothesis, str):
        raise ReviewAdapterError("queue item hypothesis must be a string")
    return ReviewTask(
        schema_version=SCHEMA_VERSION,
        item_id=f"item-{event_id}",
        event_id=event_id,
        call_id=call_id,
        turn_index=item.turn_index,
        audio_uri=item.audio_uri,
        hypothesis=item.hypothesis,
        low_confidence_spans=source.spans_for(item),
        blind=bool(item.no_prefill),
        stage1_instruction=STAGE1_INSTRUCTION,
        slot_form=slot_form,
    )


def _schema_warnings(slots: Mapping[str, Any], schema_path: str | Path | None) -> tuple[str, ...]:
    """Return PII-free warnings for slots that fail the active booking schema.

    Only field paths and validator names are recorded; validator messages can
    echo guest data and are deliberately discarded (context.md §2.8).
    """
    try:
        schema = load_booking_schema(schema_path)
    except (BookingValidationError, OSError, ValueError):
        return ("booking_schema_unavailable",)
    warnings: list[str] = []
    try:
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        errors = sorted(
            validator.iter_errors(dict(slots)),
            key=lambda error: [str(part) for part in error.absolute_path],
        )
    except (BookingValidationError, ValueError):
        return ("booking_schema_unavailable",)
    for error in errors:
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        warnings.append(f"slots_schema_invalid:{location}:{error.validator}")
    try:
        validate_booking(dict(slots), schema_path)
    except BookingValidationError as exc:
        if "check_out must be after check_in" in str(exc):
            warnings.append("slots_cross_field_invalid:check_out_after_check_in")
    return tuple(dict.fromkeys(warnings))


class ReviewCoordinator:
    """Drive the two stages through any :class:`ReviewTool` and store the result."""

    def __init__(
        self,
        tool: ReviewTool,
        store: ReviewStore | None = None,
        *,
        parser: SlotParser | None = None,
        span_source: SpanSource | None = None,
        settings: ReviewAdapterSettings | None = None,
    ) -> None:
        """Bind injectable tool, store, parser, span source and typed settings.

        Args:
            tool: The selected review tool (fake stand-ins by default).
            store: Result store; defaults to the in-memory Postgres stand-in.
            parser: Stage 2 prefill parser; defaults to the fake dict parser.
            span_source: Explicit highlight source; defaults to queue-item spans.
            settings: Typed adapter settings.

        """
        self._tool = tool
        self._store: ReviewStore = store if store is not None else InMemoryReviewStore()
        self._parser: SlotParser = parser if parser is not None else DictSlotParser()
        self._span_source: SpanSource = (
            span_source if span_source is not None else QueueItemSpanSource()
        )
        self._settings = settings or ReviewAdapterSettings()
        self._form = build_slot_form(self._settings.booking_schema_path)
        self._tasks: dict[str, ReviewTask] = {}
        self._tool_ids: dict[str, str] = {}
        self._transcripts: dict[str, str] = {}
        self._warnings: dict[str, list[str]] = {}
        self._training_consent: dict[str, bool] = {}

    @property
    def slot_form(self) -> SlotForm:
        """Return the booking-schema-derived stage 2 form."""
        return self._form

    def task_for(self, item_id: str) -> ReviewTask:
        """Return the task pushed for ``item_id``.

        Raises:
            ReviewAdapterError: If the item was never pushed.

        """
        try:
            return self._tasks[item_id]
        except KeyError as exc:
            raise ReviewAdapterError(f"unknown review item: {item_id}") from exc

    def push(self, item: QueueItemLike) -> str:
        """Push one queue item to the tool and return the adapter-local item ID.

        Re-pushing the same source event is idempotent: the tool's ``create_task``
        is called again but returns the same tool task ID, and no second result
        can be stored.
        """
        task = build_review_task(item, self._form, self._settings, self._span_source)
        tool_task_id = self._tool.create_task(task)
        if not isinstance(tool_task_id, str) or not tool_task_id:
            raise ReviewAdapterError("review tool must return a non-empty task ID")
        self._tasks[task.item_id] = task
        self._tool_ids[task.item_id] = tool_task_id
        self._training_consent[task.item_id] = bool(item.training_consent)
        return task.item_id

    def push_all(self, source: QueueItemSource) -> list[str]:
        """Push every item from an injectable queue source in order."""
        return [self.push(item) for item in source.iter_items()]

    def stage1(self, item_id: str) -> str:
        """Pull the corrected transcript, prepare stage 2 and return the transcript.

        Parser prefill is skipped entirely for blind items. Parser failures and
        empty prefills are recorded as warnings; exception text is discarded
        because it may contain guest data.
        """
        tool_task_id = self._require_tool_task(item_id)
        answer = self._tool.pull_stage1(tool_task_id)
        if answer is None:
            raise ReviewAdapterError(f"stage 1 answer is pending for {item_id}")
        transcript = answer.corrected_transcript
        if not isinstance(transcript, str):
            raise ReviewAdapterError(f"stage 1 transcript for {item_id} must be a string")
        self._transcripts[item_id] = transcript
        warnings = self._warnings.setdefault(item_id, [])
        prefill: Mapping[str, Any] | None = None
        if not self._tasks[item_id].blind:
            try:
                parsed = self._parser.parse(transcript)
            except Exception as exc:  # noqa: BLE001 - tool/parser failure becomes a warning
                warnings.append(f"parser_failed:{type(exc).__name__}")
                parsed = {}
            if not isinstance(parsed, Mapping):
                warnings.append("parser_prefill_invalid")
                parsed = {}
            prefill = dict(parsed)
            if not prefill:
                warnings.append("parser_prefill_empty")
        self._tool.prepare_stage2(tool_task_id, prefill)
        return transcript

    def stage2(self, item_id: str) -> StoredReviewResult:
        """Pull stage 2 and persist the completed, warning-annotated result.

        Routing labels outside the four contract values are rejected before any
        store write. Invalid slots are kept for human review but generate
        PII-free schema warnings; the dataset lane filters downstream.

        Raises:
            ReviewAdapterError: If stage 1 was skipped, stage 2 is pending, or
                the tool response violates the contract.

        """
        tool_task_id = self._require_tool_task(item_id)
        if item_id not in self._transcripts:
            raise ReviewAdapterError(f"stage 1 must complete before stage 2 for {item_id}")
        answer: Stage2Answer | None = self._tool.pull_stage2(tool_task_id)
        if answer is None:
            raise ReviewAdapterError(f"stage 2 answer is pending for {item_id}")
        if not isinstance(answer, Stage2Answer):
            raise ReviewAdapterError("review tool returned an invalid stage 2 answer")
        if answer.routing not in ROUTING_LABELS:
            raise ReviewAdapterError(f"unknown routing label: {answer.routing!r}")
        if not isinstance(answer.reviewer_id, str) or not answer.reviewer_id:
            raise ReviewAdapterError("stage 2 answer needs a non-empty reviewer_id")
        if not isinstance(answer.slots, Mapping):
            raise ReviewAdapterError("stage 2 slots must be a mapping")

        task = self._tasks[item_id]
        warnings = list(answer.warnings) + self._warnings.pop(item_id, [])
        warnings.extend(_schema_warnings(answer.slots, self._settings.booking_schema_path))
        transcript = self._transcripts[item_id]
        if not transcript.strip():
            warnings.append("empty_corrected_transcript")
        if answer.completed_at < answer.started_at:
            warnings.append("review_timestamps_out_of_order")
        result = StoredReviewResult(
            schema_version=SCHEMA_VERSION,
            item_id=item_id,
            event_id=task.event_id,
            call_id=task.call_id,
            turn_index=task.turn_index,
            audio_uri=task.audio_uri,
            hypothesis=task.hypothesis,
            corrected_transcript=transcript,
            slots=dict(answer.slots),
            routing=answer.routing,
            reviewer_id=answer.reviewer_id,
            started_at=answer.started_at,
            completed_at=answer.completed_at,
            warnings=tuple(dict.fromkeys(warnings)),
            blind=task.blind,
            tool_name=self._tool.tool_name(),
            training_consent=self._training_consent.get(item_id, False),
        )
        return self._store.save(result)

    def pull_result(self, item_id: str) -> StoredReviewResult | None:
        """Pull a stored result by adapter-local item ID."""
        return self._store.get(item_id)

    def run_item(self, item: QueueItemLike) -> StoredReviewResult:
        """Push one item, run both stages and return its stored result."""
        item_id = self.push(item)
        self.stage1(item_id)
        return self.stage2(item_id)

    def run_all(self, source: QueueItemSource) -> tuple[StoredReviewResult, ...]:
        """Run both stages for every item in a source and return stored results."""
        return tuple(self.run_item(item) for item in source.iter_items())

    def _require_tool_task(self, item_id: str) -> str:
        """Return the tool task ID for ``item_id`` or raise a clear error."""
        try:
            return self._tool_ids[item_id]
        except KeyError as exc:
            raise ReviewAdapterError(f"unknown review item: {item_id}") from exc
