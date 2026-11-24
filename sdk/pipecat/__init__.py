"""Optional Pipecat integration for VoiceLoop turn capture.

No Pipecat, OpenTelemetry or network package is required: adapters implement
the small protocols in :mod:`sdk.pipecat.observer`, and the default sink calls
``sdk.log_turn`` lazily with an in-memory fallback.
"""

from .observer import (
    CapturedAudio,
    ConsentFlags,
    ConversationLike,
    DecoderStats,
    DialogueContext,
    FrameLike,
    InMemoryTurnSink,
    ParserFrameLike,
    ParserResult,
    PipecatObserver,
    SdkLogTurnSink,
    SttFrameLike,
    SttResult,
    TurnSink,
    UserAudioFrameLike,
    WordConfidence,
    WordLike,
    build_turn_event,
    default_sink,
    validate_turn_event,
)
from .traces import (
    CALL_ID_ATTRIBUTE,
    CONVERSATION_ID_ATTRIBUTE,
    TURN_INDEX_ATTRIBUTE,
    DictSpan,
    SpanLike,
    TraceContext,
    bind_conversation_ids,
    current_span,
)

__all__ = [
    "CALL_ID_ATTRIBUTE",
    "CONVERSATION_ID_ATTRIBUTE",
    "TURN_INDEX_ATTRIBUTE",
    "CapturedAudio",
    "ConsentFlags",
    "ConversationLike",
    "DecoderStats",
    "DictSpan",
    "DialogueContext",
    "FrameLike",
    "InMemoryTurnSink",
    "ParserFrameLike",
    "ParserResult",
    "PipecatObserver",
    "SdkLogTurnSink",
    "SpanLike",
    "SttFrameLike",
    "SttResult",
    "TraceContext",
    "TurnSink",
    "UserAudioFrameLike",
    "WordConfidence",
    "WordLike",
    "bind_conversation_ids",
    "build_turn_event",
    "current_span",
    "default_sink",
    "validate_turn_event",
]
