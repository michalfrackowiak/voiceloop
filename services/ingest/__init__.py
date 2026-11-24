"""HTTP and OpenTelemetry ingest for VoiceLoop turn events.

The service validates events against the shared turn-event contract, then hands
them to a bounded, fail-open write queue. Metadata goes to a repository
(Postgres in production, an in-memory fake by default); audio bytes go only to
an object store. See ``README.md`` for endpoints and attribute mapping.
"""

from services.ingest.service import IngestService
from services.ingest.settings import IngestSettings

__all__ = ["IngestService", "IngestSettings"]
