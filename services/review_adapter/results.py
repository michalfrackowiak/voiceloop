"""Completed review results and the idempotent in-memory result store.

The stored record keeps the OpenAPI 1.0.0 result fields plus lineage and
warnings that the contract does not expose: ``additionalProperties: false`` on
``ReviewResult`` means ``warnings`` and the original event ID cannot be added
there without a version bump. ``to_api_result()`` emits only contract fields;
``to_dict()`` is the internal/Postgres record.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .interface import ROUTING_LABELS, SCHEMA_VERSION, ReviewAdapterError


def _rfc3339(value: datetime) -> str:
    """Return an RFC 3339 UTC timestamp string for ``value``."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: object, field: str) -> datetime:
    """Parse a timestamp emitted by :func:`_rfc3339` or a real datetime value."""
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ReviewAdapterError(f"invalid {field}: {value!r}") from exc
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    raise ReviewAdapterError(f"{field} must be an RFC 3339 string or datetime")


@dataclass(frozen=True)
class StoredReviewResult:
    """One completed two-stage review with reviewer, timing, lineage and warnings."""

    schema_version: str
    item_id: str
    event_id: str
    call_id: str
    turn_index: int
    audio_uri: str
    hypothesis: str
    corrected_transcript: str
    slots: Mapping[str, Any]
    routing: str
    reviewer_id: str
    started_at: datetime
    completed_at: datetime
    warnings: tuple[str, ...] = ()
    blind: bool = False
    tool_name: str = ""
    training_consent: bool = False

    def __post_init__(self) -> None:
        """Reject invalid routing labels and missing identifiers before persistence."""
        if self.routing not in ROUTING_LABELS:
            raise ReviewAdapterError(f"unknown routing label: {self.routing!r}")
        for field, value in (
            ("item_id", self.item_id),
            ("event_id", self.event_id),
            ("call_id", self.call_id),
            ("reviewer_id", self.reviewer_id),
        ):
            if not isinstance(value, str) or not value:
                raise ReviewAdapterError(f"{field} must be a non-empty string")
        if not isinstance(self.slots, Mapping):
            raise ReviewAdapterError("slots must be a mapping")

    def to_api_result(self) -> dict[str, Any]:
        """Return only the fields allowed by the OpenAPI ``ReviewResult`` schema."""
        return {
            "schema_version": self.schema_version,
            "item_id": self.item_id,
            "reviewer_id": self.reviewer_id,
            "corrected_transcript": self.corrected_transcript,
            "slots": dict(self.slots),
            "routing": self.routing,
            "started_at": _rfc3339(self.started_at),
            "completed_at": _rfc3339(self.completed_at),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the full internal record, including lineage, warnings and tool name."""
        return {
            "schema_version": self.schema_version,
            "item_id": self.item_id,
            "event_id": self.event_id,
            "call_id": self.call_id,
            "turn_index": self.turn_index,
            "audio_uri": self.audio_uri,
            "hypothesis": self.hypothesis,
            "corrected_transcript": self.corrected_transcript,
            "slots": dict(self.slots),
            "routing": self.routing,
            "reviewer_id": self.reviewer_id,
            "started_at": _rfc3339(self.started_at),
            "completed_at": _rfc3339(self.completed_at),
            "warnings": list(self.warnings),
            "blind": self.blind,
            "tool_name": self.tool_name,
            "training_consent": self.training_consent,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StoredReviewResult:
        """Rebuild a stored result from :meth:`to_dict` output.

        Args:
            payload: A JSON-compatible record previously produced by this class.

        Returns:
            The reconstructed immutable result.

        Raises:
            ReviewAdapterError: If a required field or timestamp is malformed.

        """
        try:
            turn_index = int(payload["turn_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ReviewAdapterError("turn_index must be an integer") from exc
        slots = payload.get("slots")
        if not isinstance(slots, Mapping):
            raise ReviewAdapterError("slots must be an object")
        raw_warnings = payload.get("warnings", ())
        warnings = (
            tuple(str(item) for item in raw_warnings)
            if isinstance(raw_warnings, (list, tuple))
            else ()
        )
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            item_id=str(payload["item_id"]),
            event_id=str(payload["event_id"]),
            call_id=str(payload["call_id"]),
            turn_index=turn_index,
            audio_uri=str(payload.get("audio_uri", "")),
            hypothesis=str(payload.get("hypothesis", "")),
            corrected_transcript=str(payload.get("corrected_transcript", "")),
            slots=dict(slots),
            routing=str(payload["routing"]),
            reviewer_id=str(payload["reviewer_id"]),
            started_at=_parse_timestamp(payload["started_at"], "started_at"),
            completed_at=_parse_timestamp(payload["completed_at"], "completed_at"),
            warnings=warnings,
            blind=bool(payload.get("blind", False)),
            tool_name=str(payload.get("tool_name", "")),
            training_consent=bool(payload.get("training_consent", False)),
        )

    def to_dataset_view(self) -> dict[str, Any]:
        """Return a flat lineage-rich view for a P2-05 adapter; not a turn event."""
        return {
            "source_event_id": self.event_id,
            "call_id": self.call_id,
            "turn_index": self.turn_index,
            "audio_uri": self.audio_uri,
            "hypothesis": self.hypothesis,
            "corrected_transcript": self.corrected_transcript,
            "slots": dict(self.slots),
            "routing": self.routing,
            "reviewer_id": self.reviewer_id,
            "training_consent": self.training_consent,
            "warnings": list(self.warnings),
        }


class InMemoryReviewStore:
    """Deterministic fake result store standing in for a Postgres table.

    The Postgres seam is a table keyed uniquely on ``event_id`` (and indexed by
    ``(call_id, turn_index)``), with ``slots`` and ``warnings`` as JSONB and
    ``started_at``/``completed_at`` as timestamptz. ``INSERT ... ON CONFLICT DO
    NOTHING`` on that unique key gives the same idempotency as :meth:`save`.
    Audio bytes are never stored here (context.md §2.8 and §11).
    """

    def __init__(self) -> None:
        """Create empty event-ID and item-ID indexes."""
        self._by_event: dict[str, StoredReviewResult] = {}
        self._by_item: dict[str, str] = {}

    def save(self, result: StoredReviewResult) -> StoredReviewResult:
        """Insert ``result`` once per source event, returning the first stored copy.

        Args:
            result: Completed review to persist.

        Returns:
            The newly saved result, or the previously stored one for the same
            ``event_id``, so duplicate tool submissions cannot create two rows.

        Raises:
            ReviewAdapterError: If one ``item_id`` is reused for another event.

        """
        existing = self._by_event.get(result.event_id)
        if existing is not None:
            return existing
        mapped_event = self._by_item.get(result.item_id)
        if mapped_event is not None and mapped_event != result.event_id:
            raise ReviewAdapterError(
                f"item_id {result.item_id!r} is already mapped to {mapped_event!r}"
            )
        self._by_event[result.event_id] = result
        self._by_item[result.item_id] = result.event_id
        return result

    def get(self, item_id: str) -> StoredReviewResult | None:
        """Return the result for an adapter-local ``item_id``, if present."""
        event_id = self._by_item.get(item_id)
        return self._by_event.get(event_id) if event_id is not None else None

    def get_by_event(self, event_id: str) -> StoredReviewResult | None:
        """Return the result for a source event ID, if present."""
        return self._by_event.get(event_id)

    def all(self) -> tuple[StoredReviewResult, ...]:
        """Return every stored result in insertion order."""
        return tuple(self._by_event.values())
