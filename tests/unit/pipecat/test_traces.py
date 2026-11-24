"""Tests for the dependency-free trace shim."""

from __future__ import annotations

from sdk.pipecat import (
    CALL_ID_ATTRIBUTE,
    CONVERSATION_ID_ATTRIBUTE,
    TURN_INDEX_ATTRIBUTE,
    TraceContext,
    bind_conversation_ids,
    current_span,
)


def test_current_span_is_optional() -> None:
    """Without OpenTelemetry the shim returns None; with it, a real span."""
    span = current_span()
    assert span is None or hasattr(span, "set_attribute")


def test_bind_survives_a_broken_span() -> None:
    """A failing span cannot prevent dict propagation or raise into a call."""

    class BrokenSpan:
        """Span whose set_attribute always fails."""

        def set_attribute(self, key: str, value: object) -> None:
            """Raise to simulate a broken tracer."""
            raise RuntimeError("tracer down")

    context = TraceContext()
    attributes = context.bind("call-x", "conv-x", span=BrokenSpan(), turn_index=3)
    assert attributes == {
        CALL_ID_ATTRIBUTE: "call-x",
        CONVERSATION_ID_ATTRIBUTE: "conv-x",
        TURN_INDEX_ATTRIBUTE: 3,
    }
    assert dict(context.attributes) == attributes
    assert bind_conversation_ids("call-y") == {
        CALL_ID_ATTRIBUTE: "call-y",
        CONVERSATION_ID_ATTRIBUTE: "call-y",
    }
