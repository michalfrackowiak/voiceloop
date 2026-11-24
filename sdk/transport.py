"""Injectable transports for turn events."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Mapping
from threading import Lock
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from voiceloop_common.errors import TransientError

if TYPE_CHECKING:
    import httpx


class TransportError(TransientError):
    """A transport could not deliver a turn event."""


@runtime_checkable
class TurnTransport(Protocol):
    """Interface used by the SDK worker; implementations should fail fast."""

    def send(self, event: Mapping[str, Any], *, timeout_seconds: float) -> None:
        """Send one turn event, raising on failure.

        Args:
            event: Schema-shaped turn event.
            timeout_seconds: Upper bound the worker allows for this send.

        """


class InMemoryTransport:
    """Deterministic fake transport for unit tests and local development."""

    def __init__(
        self,
        *,
        delay_seconds: float = 0.0,
        failures_remaining: int = 0,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create a fake transport.

        Args:
            delay_seconds: Simulated server delay before each send.
            failures_remaining: Number of initial sends that raise.
            sleeper: Injectable sleep function for deterministic timing tests.

        """
        self._delay_seconds = max(0.0, delay_seconds)
        self._failures_remaining = max(0, failures_remaining)
        self._sleeper = sleeper
        self._lock = Lock()
        self._calls = 0
        self._events: list[dict[str, Any]] = []

    def send(self, event: Mapping[str, Any], *, timeout_seconds: float) -> None:
        """Sleep for the configured delay, then record or fail the send."""
        if self._delay_seconds:
            self._sleeper(min(self._delay_seconds, max(0.0, timeout_seconds)))
        with self._lock:
            self._calls += 1
            if self._failures_remaining > 0:
                self._failures_remaining -= 1
                raise TransportError("in-memory transport failure")
            self._events.append(copy.deepcopy(dict(event)))

    @property
    def call_count(self) -> int:
        """Number of send attempts made so far."""
        with self._lock:
            return self._calls

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        """Successful events in delivery order."""
        with self._lock:
            return tuple(self._events)


class HttpTransport:
    """Optional httpx-backed transport for ``POST /v1/turns``."""

    def __init__(self, endpoint: str, *, client: httpx.Client | None = None) -> None:
        """Create a transport; httpx is imported only when a send occurs.

        Args:
            endpoint: Ingest URL for turn events.
            client: Optional injected httpx client for tests or connection reuse.

        """
        self._endpoint = endpoint
        self._client = client
        self._owns_client = client is None
        self._lock = Lock()

    def send(self, event: Mapping[str, Any], *, timeout_seconds: float) -> None:
        """POST the event and raise :class:`TransportError` on any failure."""
        import httpx

        try:
            response = self._client_for_send().post(
                self._endpoint,
                json=dict(event),
                timeout=timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise TransportError(str(exc)) from exc

    def close(self) -> None:
        """Close the httpx client if this transport created it."""
        with self._lock:
            client, owns_client = self._client, self._owns_client
            self._client = None
        if owns_client and client is not None:
            client.close()

    def _client_for_send(self) -> httpx.Client:
        """Return the injected client or lazily create one."""
        import httpx

        with self._lock:
            if self._client is None:
                self._client = httpx.Client()
            return self._client
