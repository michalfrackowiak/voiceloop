"""Shared fixtures and read-only hash guards for P2-13 integration tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from services.datasets.golden import VerificationReport, verify_seal

from .helpers import (
    EXPECTED_FIXTURE_SHA256,
    EXPECTED_GOLDEN_CONTENT_HASH,
    FIXTURE_TURNS_PATH,
    GOLDEN_ARTIFACTS_DIR,
    read_fixture_rows,
    sha256_file,
)


@pytest.fixture
def repo_root() -> Path:
    """Return the repository root."""
    return FIXTURE_TURNS_PATH.parents[2]


@pytest.fixture
def fixture_turns_path() -> Path:
    """Return the committed synthetic label manifest path."""
    return FIXTURE_TURNS_PATH


@pytest.fixture
def fixture_rows() -> list[dict[str, Any]]:
    """Return the read-only synthetic label rows."""
    return read_fixture_rows(FIXTURE_TURNS_PATH)


@pytest.fixture
def sealed_golden() -> VerificationReport:
    """Verify the sealed synthetic golden stand-in and pin its content hash."""
    report = verify_seal(GOLDEN_ARTIFACTS_DIR)
    report.assert_ok()
    assert report.manifest is not None
    assert report.seal is not None
    assert report.manifest.content_hash == EXPECTED_GOLDEN_CONTENT_HASH
    assert report.seal.content_hash == EXPECTED_GOLDEN_CONTENT_HASH
    return report


@pytest.fixture(autouse=True)
def read_only_source_hashes() -> Iterator[dict[str, str]]:
    """Fail if any test mutates the committed fixture or golden artifacts."""
    paths = [
        FIXTURE_TURNS_PATH,
        *sorted(path for path in GOLDEN_ARTIFACTS_DIR.rglob("*") if path.is_file()),
    ]
    before = {str(path): sha256_file(path) for path in paths}
    assert before[str(FIXTURE_TURNS_PATH)] == EXPECTED_FIXTURE_SHA256
    yield before
    after = {str(path): sha256_file(path) for path in paths}
    assert after == before, "P2-13 must not mutate committed fixtures or golden artifacts"
