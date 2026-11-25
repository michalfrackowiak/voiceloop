"""Shared fixtures for synthetic augmentation tests.

All P2-05, golden and fixture inputs are read-only; the only writes in these
tests go to pytest ``tmp_path`` directories.
"""

from __future__ import annotations

from typing import Any

import pytest

from services.datasets import build_dataset_from_fixtures
from services.datasets.augment.cli import AugmentationRun, build_augmentation
from services.datasets.augment.settings import AugmentSettings
from services.datasets.manifest import build_manifest as build_dataset_manifest


@pytest.fixture
def settings() -> AugmentSettings:
    """Fast deterministic settings using short 8 kHz stand-in carriers."""
    return AugmentSettings(duration_ms=80, sample_rate_hz=8000, noise_level=4, seed=3)


@pytest.fixture(scope="module")
def source_manifest() -> dict[str, Any]:
    """A real P2-05 manifest built read-only from the synthetic fixture pool."""
    build = build_dataset_from_fixtures()
    return build_dataset_manifest(build, created_at="2026-11-25T00:00:00Z")


@pytest.fixture
def golden_ids() -> tuple[str, ...]:
    """Injected sealed-call exclusion subset for non-vacuous tests."""
    return ("call-0001", "call-0002")


@pytest.fixture
def regression_ids() -> tuple[str, ...]:
    """Injected regression-call exclusion subset; disjoint from golden."""
    return ("call-0003",)


@pytest.fixture
def run(
    settings: AugmentSettings,
    source_manifest: dict[str, Any],
    golden_ids: tuple[str, ...],
    regression_ids: tuple[str, ...],
) -> AugmentationRun:
    """One deterministic augmentation batch with injected exclusion lists."""
    return build_augmentation(
        settings,
        dataset_manifest=source_manifest,
        golden_call_ids=golden_ids,
        regression_call_ids=regression_ids,
        created_at="2026-11-25T00:00:00Z",
    )
