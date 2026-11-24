"""OpenTelemetry-compatible trace shim for conversation IDs (no OTel dependency).

The observer attaches these attributes to the current span when one exists and
always records them in a small dict context that callers can export later:

| Attribute | Meaning |
| --- | --- |
| ``voiceloop.call_id`` | Call identifier used as the turn-event join key. |
| ``voiceloop.conversation_id`` | Conversation identifier; defaults to ``call_id``. |
| ``voiceloop.turn_index`` | Zero-based turn number when supplied. |

``SpanLike`` is structurally compatible with an OpenTelemetry ``Span`` (it only
needs ``set_attribute``). When opentelemetry happens to be installed,
:func:`current_span` delegates to its real ``get_current_span()``; otherwise it
returns ``None`` and the dict context is the only carrier.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, Protocol

CALL_ID_ATTRIBUTE = "voiceloop.call_id"
CONVERSATION_ID_ATTRIBUTE = "voiceloop.conversation_id"
TURN_INDEX_ATTRIBUTE = "voiceloop.turn_index"

AttributeValue = str | int | float | bool


class SpanLike(Protocol):
    """Minimal subset of an OpenTelemetry span used by this integration."""

    def set_attribute(self, key: str, value: AttributeValue) -> None:
        """Record one attribute on the span."""


class DictSpan:
    """In-memory ``SpanLike`` fake for tests and the stdlib-only example."""

    def __init__(self, attributes: MutableMapping[str, AttributeValue] | None = None) -> None:
        """Create a span backed by ``attributes`` or a fresh dict."""
        self.attributes: MutableMapping[str, AttributeValue] = (
            attributes if attributes is not None else {}
        )

    def set_attribute(self, key: str, value: AttributeValue) -> None:
        """Store an attribute; mirrors the OpenTelemetry span API."""
        self.attributes[key] = value


def current_span() -> SpanLike | None:
    """Return the active OpenTelemetry span, or ``None`` when unavailable.

    The import is deliberately lazy and optional: no dependency on
    ``opentelemetry`` is required, and a broken/incompatible installation must
    not break a conversation loop.
    """
    try:
        from opentelemetry import trace
    except Exception:
        return None
    try:
        span = trace.get_current_span()
    except Exception:
        return None
    return span if hasattr(span, "set_attribute") else None


class TraceContext:
    """Dict carrier for VoiceLoop conversation attributes."""

    def __init__(self, attributes: MutableMapping[str, AttributeValue] | None = None) -> None:
        """Create a context with an optional caller-owned attribute mapping."""
        self.attributes: MutableMapping[str, AttributeValue] = (
            attributes if attributes is not None else {}
        )

    def bind(
        self,
        call_id: str,
        conversation_id: str | None = None,
        *,
        span: SpanLike | None = None,
        turn_index: int | None = None,
    ) -> dict[str, AttributeValue]:
        """Write call/conversation IDs into this context and an optional span.

        Args:
            call_id: Non-empty call identifier.
            conversation_id: Conversation identifier; falls back to ``call_id``.
            span: Optional OpenTelemetry-compatible span to annotate.
            turn_index: Optional zero-based turn number.

        Returns:
            A snapshot of all VoiceLoop attributes now in the context.

        """
        return bind_conversation_ids(
            call_id,
            conversation_id,
            span=span,
            attributes=self.attributes,
            turn_index=turn_index,
        )

    def get(self, key: str) -> AttributeValue | None:
        """Return one attribute, or ``None`` when it has not been bound."""
        return self.attributes.get(key)


def bind_conversation_ids(
    call_id: str,
    conversation_id: str | None = None,
    *,
    span: SpanLike | None = None,
    attributes: MutableMapping[str, AttributeValue] | None = None,
    turn_index: int | None = None,
) -> dict[str, AttributeValue]:
    """Attach conversation IDs to a dict context and/or the current span.

    Any failure from an injected span or mapping is swallowed: trace enrichment
    must never raise into a live call (``context.md`` §2.1).

    Args:
        call_id: Call identifier from the turn event.
        conversation_id: Conversation identifier; defaults to ``call_id``.
        span: Optional span implementing ``set_attribute``.
        attributes: Optional mapping to update; a dict is used when omitted.
        turn_index: Optional zero-based turn number.

    Returns:
        The bound attributes (a copy of ``attributes`` when supplied).

    """
    values: dict[str, AttributeValue] = {
        CALL_ID_ATTRIBUTE: call_id,
        CONVERSATION_ID_ATTRIBUTE: conversation_id or call_id,
    }
    if turn_index is not None:
        values[TURN_INDEX_ATTRIBUTE] = turn_index
    target = attributes if attributes is not None else {}
    try:
        for key, value in values.items():
            target[key] = value
    except Exception:
        pass
    if span is not None:
        for key, value in values.items():
            try:
                span.set_attribute(key, value)
            except Exception:
                pass
    return dict(target)


def as_span(value: Any) -> SpanLike | None:
    """Return ``value`` as a ``SpanLike`` when it exposes ``set_attribute``."""
    return value if hasattr(value, "set_attribute") else None
