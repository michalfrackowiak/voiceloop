"""Shared fixtures for P3-05 benchmark tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from services.evalgate.benchmarks.settings import BenchmarkSettings


@pytest.fixture
def repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[3]


@pytest.fixture
def benchmark_settings(tmp_path: Path) -> BenchmarkSettings:
    """Return settings with small deterministic bootstrap settings and a temp output."""
    return BenchmarkSettings(
        fixtures_path="tests/fixtures/turns.jsonl",
        golden_artifact_dir="services/datasets/golden/artifacts",
        output_path=str(tmp_path / "comparison.json"),
        bootstrap_iterations=40,
        bootstrap_seed=11,
    )
