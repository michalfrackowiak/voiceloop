"""Deterministic, in-memory fixtures for the isolated P4-05 seam tests."""

from __future__ import annotations

import pytest

from services.projects import (
    InMemoryProjectRegistry,
    ProjectConfig,
    ProjectId,
    ProjectScope,
    ProjectSettings,
)


@pytest.fixture
def settings() -> ProjectSettings:
    """Return settings with the two default synthetic project IDs."""
    return ProjectSettings()


@pytest.fixture
def alpha_id() -> ProjectId:
    """Return the first validated synthetic project ID."""
    return ProjectId.parse("synthetic-alpha")


@pytest.fixture
def beta_id() -> ProjectId:
    """Return the second validated synthetic project ID."""
    return ProjectId.parse("synthetic-beta")


@pytest.fixture
def disabled_id() -> ProjectId:
    """Return a registered but disabled synthetic project ID."""
    return ProjectId.parse("synthetic-disabled")


@pytest.fixture
def alpha_scope(alpha_id: ProjectId) -> ProjectScope:
    """Return a validated scope for the first synthetic project."""
    return ProjectScope.for_project(alpha_id)


@pytest.fixture
def beta_scope(beta_id: ProjectId) -> ProjectScope:
    """Return a validated scope for the second synthetic project."""
    return ProjectScope.for_project(beta_id)


@pytest.fixture
def registry(
    alpha_id: ProjectId, beta_id: ProjectId, disabled_id: ProjectId
) -> InMemoryProjectRegistry:
    """Return a fake registry with two enabled projects and one disabled one."""
    return InMemoryProjectRegistry(
        [
            ProjectConfig(alpha_id, "schemas/booking.schema.json", ("golden", "regression")),
            ProjectConfig(
                beta_id, "schemas/booking-beta.schema.json", ("golden", "general-capability")
            ),
            ProjectConfig(
                disabled_id, "schemas/booking-disabled.schema.json", ("golden",), enabled=False
            ),
        ]
    )
