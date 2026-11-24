"""Shared fixtures for ingest tests; all backends are in-memory fakes."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from services.ingest.app import create_app
from services.ingest.repository import InMemoryRepository
from services.ingest.service import IngestService
from services.ingest.settings import IngestSettings
from services.ingest.storage import InMemoryObjectStore

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def event_files() -> list[Path]:
    """Return all synthetic fixture event paths, deterministically ordered."""
    return sorted((REPO_ROOT / "tests" / "fixtures" / "events").glob("*/*.json"))


@pytest.fixture
def load_json() -> Callable[[str], Any]:
    """Return a loader for JSON files addressed by repo-relative POSIX path."""

    def _load(relative: str) -> Any:
        with (REPO_ROOT / relative).open(encoding="utf-8") as stream:
            return json.load(stream)

    return _load


@pytest.fixture
def settings() -> IngestSettings:
    """Return settings with the background worker disabled for deterministic tests."""
    return IngestSettings(autostart_worker=False)


@pytest.fixture
def repository() -> InMemoryRepository:
    """Return an empty in-memory repository fake."""
    return InMemoryRepository()


@pytest.fixture
def object_store() -> InMemoryObjectStore:
    """Return an empty in-memory object-store fake."""
    return InMemoryObjectStore()


@pytest.fixture
def service(
    settings: IngestSettings,
    repository: InMemoryRepository,
    object_store: InMemoryObjectStore,
) -> IngestService:
    """Return an ingest service wired to the fakes."""
    return IngestService(settings, repository, object_store, autostart_worker=False)


@pytest.fixture
def client(service: IngestService) -> Iterator[TestClient]:
    """Return a TestClient sharing the service fixture's queue and repository."""
    app = create_app(autostart_worker=False, service=service)
    with TestClient(app) as test_client:
        yield test_client
