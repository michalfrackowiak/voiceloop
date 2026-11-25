"""Shared fixtures for the P3-09 golden maintenance rehearsal tests."""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from services.datasets.golden.maintenance import (
    CandidateSet,
    GoldenMaintenanceSettings,
    ReviewAuditLog,
    ReviewBundle,
    build_candidate_set,
    record_candidate_build,
    run_scripted_review,
)
from voiceloop_common import frozen_clock, use_clock

ENV_PREFIX = "VOICELOOP_GOLDEN_MAINT_"
FROZEN_TIME = datetime(2026, 11, 25, 0, 0, 0, tzinfo=UTC)
PURPOSE = "P3-09 synthetic rehearsal"


@pytest.fixture(autouse=True)
def scoped_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove inherited configuration so tests only use their explicit inputs."""
    for key in tuple(os.environ):
        if key.startswith(ENV_PREFIX):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def frozen_time() -> Iterator[None]:
    """Freeze voiceloop_common.utcnow() for deterministic audit timestamps."""
    with use_clock(frozen_clock(FROZEN_TIME)):
        yield


@pytest.fixture
def settings(tmp_path: Path) -> GoldenMaintenanceSettings:
    """Return settings that write only to a temporary maintenance directory."""
    return GoldenMaintenanceSettings(output_dir=str(tmp_path / "maintenance"))


@pytest.fixture
def candidate_set(settings: GoldenMaintenanceSettings) -> CandidateSet:
    """Return one deterministic synthetic extension candidate set."""
    return build_candidate_set(settings)


@pytest.fixture
def reviewed_candidate(
    settings: GoldenMaintenanceSettings, candidate_set: CandidateSet
) -> tuple[CandidateSet, ReviewBundle, ReviewAuditLog]:
    """Return a candidate set with a build entry and two scripted reviews."""
    audit = ReviewAuditLog(Path(settings.output_dir) / "review_audit.jsonl")
    record_candidate_build(audit, candidate_set, settings, actor_id="data-steward", purpose=PURPOSE)
    bundle = run_scripted_review(candidate_set, settings, audit)
    return candidate_set, bundle, audit
