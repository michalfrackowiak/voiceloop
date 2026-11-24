"""Fixtures that isolate shared clock and logging state between tests."""

from collections.abc import Iterator

import pytest
import structlog

from voiceloop_common import clock


@pytest.fixture(autouse=True)
def reset_shared_state() -> Iterator[None]:
    """Restore the real clock and default logging after each test."""
    yield
    clock.set_clock(None)
    structlog.reset_defaults()
