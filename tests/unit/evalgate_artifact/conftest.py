"""Shared fixtures for P3-04 artifact safety tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from services.evalgate.artifact.settings import ArtifactSettings


@pytest.fixture()
def settings(tmp_path: Path) -> ArtifactSettings:
    """Return deterministic settings whose outputs stay inside ``tmp_path``."""
    return ArtifactSettings(
        output_path=str(tmp_path / "report.json"),
        artifact_store_dir=str(tmp_path / "artifacts"),
        bootstrap_iterations=50,
        bootstrap_seed=42,
    )
