"""Shared fixtures for connector unit tests (synthetic data only)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sdk.connectors.mock_booking import FixtureEventStore, MockBookingSystem
from voiceloop_common.clock import frozen_clock, use_clock

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def fixture_dir() -> Path:
    """Return the synthetic fixture directory."""
    return REPO_ROOT / "tests" / "fixtures"


@pytest.fixture(autouse=True)
def frozen_utc() -> Iterator[None]:
    """Freeze UTC timestamps so mock booking updates are deterministic."""
    with use_clock(frozen_clock(datetime(2026, 11, 25, 12, 0, tzinfo=UTC))):
        yield


@pytest.fixture
def booking(fixture_dir: Path) -> MockBookingSystem:
    """Return a mock booking system seeded from synthetic fixtures."""
    return MockBookingSystem.seed_from_fixtures(fixture_dir)


@pytest.fixture
def events(fixture_dir: Path) -> FixtureEventStore:
    """Return an in-memory copy of the fixture turn events."""
    return FixtureEventStore.load(fixture_dir)
