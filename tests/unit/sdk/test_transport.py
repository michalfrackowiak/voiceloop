"""Tests for the injectable SDK transports."""

from typing import cast
from unittest.mock import Mock

import httpx
import pytest

from sdk import HttpTransport, InMemoryTransport, TransportError


def test_in_memory_transport_records_events_and_failures() -> None:
    slept: list[float] = []
    transport = InMemoryTransport(
        delay_seconds=0.1,
        failures_remaining=1,
        sleeper=slept.append,
    )

    with pytest.raises(TransportError):
        transport.send({"event_id": "first"}, timeout_seconds=1.0)
    transport.send({"event_id": "second"}, timeout_seconds=1.0)

    assert slept == [0.1, 0.1]
    assert transport.call_count == 2
    assert transport.events == ({"event_id": "second"},)


def test_http_transport_posts_json_with_timeout() -> None:
    url = "http://ingest.test/v1/turns"
    request = httpx.Request("POST", url)
    client = Mock()
    client.post.return_value = httpx.Response(200, request=request)
    transport = HttpTransport(url, client=cast(httpx.Client, client))

    transport.send({"event_id": "event-1"}, timeout_seconds=0.25)

    client.post.assert_called_once_with(url, json={"event_id": "event-1"}, timeout=0.25)


def test_http_transport_raises_transport_error_on_http_failure() -> None:
    url = "http://ingest.test/v1/turns"
    request = httpx.Request("POST", url)
    client = Mock()
    client.post.return_value = httpx.Response(503, request=request)
    transport = HttpTransport(url, client=cast(httpx.Client, client))

    with pytest.raises(TransportError):
        transport.send({"event_id": "event-1"}, timeout_seconds=0.25)
