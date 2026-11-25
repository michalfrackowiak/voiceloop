"""Self-contained fail-open buffering for ingest writes.

This package depends only on the standard library and ``voiceloop_common``.
It is intentionally separate from the P1-01 ingest app; the lead wires it in
after both workers land. See ``docs/handoffs/P1-02.md`` for the integration API.
"""

from .buffer import BoundedBuffer
from .metrics import FailOpenMetrics, HealthSnapshot, HealthStatus
from .models import BufferedEvent, EnqueueResult, FlushResult, ShedReason
from .settings import FailOpenSettings
from .sinks import RecordingSink, StorageSink

__all__ = [
    "BoundedBuffer",
    "BufferedEvent",
    "EnqueueResult",
    "FailOpenMetrics",
    "FailOpenSettings",
    "FlushResult",
    "HealthSnapshot",
    "HealthStatus",
    "RecordingSink",
    "ShedReason",
    "StorageSink",
]
