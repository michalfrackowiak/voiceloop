"""Tests for the fail-closed in-memory project registry fake."""

from __future__ import annotations

from typing import cast

import pytest

from services.projects import (
    INTERNAL_SCOPE_LABEL,
    INTERNAL_SCOPE_VERSION,
    STANDIN_LABEL,
    DisabledProjectError,
    DuplicateProjectError,
    InMemoryProjectRegistry,
    InvalidProjectConfigError,
    InvalidProjectIdError,
    ProjectConfig,
    ProjectId,
    ProjectRegistry,
    UnknownProjectError,
)


def test_config_exposes_schema_path_eval_sets_enabled_flag_and_label(
    alpha_id: ProjectId,
) -> None:
    config = ProjectConfig(
        alpha_id, "schemas/booking.schema.json", ("golden", "regression"), enabled=True
    )
    assert config.project_id == alpha_id
    assert config.booking_schema_path == "schemas/booking.schema.json"
    assert config.eval_set_names == ("golden", "regression")
    assert config.enabled is True
    assert config.tag.project_id == alpha_id
    payload = config.to_dict()
    assert payload["label"] == STANDIN_LABEL
    assert payload["scope_label"] == INTERNAL_SCOPE_LABEL
    assert payload["scope_version"] == INTERNAL_SCOPE_VERSION


def test_registry_resolves_two_projects_without_crossing(
    registry: InMemoryProjectRegistry,
    alpha_id: ProjectId,
    beta_id: ProjectId,
    disabled_id: ProjectId,
) -> None:
    alpha = registry.require(alpha_id)
    beta = registry.require(beta_id)
    assert isinstance(registry, ProjectRegistry)
    assert alpha.project_id != beta.project_id
    assert alpha.booking_schema_path != beta.booking_schema_path
    assert alpha.eval_set_names != beta.eval_set_names
    alpha_view = cast(str, alpha.to_dict()["project_id"])
    beta_view = cast(str, beta.to_dict()["project_id"])
    assert beta.project_id.value not in alpha_view
    assert alpha.project_id.value not in beta_view
    assert registry.get(alpha_id) == alpha
    assert registry.get("synthetic-beta") == beta
    assert registry.enabled_ids() == (alpha_id, beta_id)
    assert disabled_id not in registry.enabled_ids()
    assert registry.registered_ids() == (alpha_id, beta_id, disabled_id)


def test_unknown_disabled_and_invalid_ids_fail_closed(
    registry: InMemoryProjectRegistry,
) -> None:
    with pytest.raises(UnknownProjectError):
        registry.require("synthetic-missing")
    with pytest.raises(DisabledProjectError):
        registry.require("synthetic-disabled")
    assert registry.get("synthetic-missing") is None
    assert registry.get("synthetic-disabled") is None
    bad_values: tuple[object, ...] = (
        None,
        "",
        "Synthetic-Alpha",
        "ada.fixture@example.com",
        "+1-555-010-0001",
        "4111111111111111",
        123,
    )
    for bad in bad_values:
        candidate = cast("str | ProjectId | None", bad)
        with pytest.raises(InvalidProjectIdError):
            registry.require(candidate)
        assert registry.get(candidate) is None


def test_error_messages_never_echo_rejected_values(
    registry: InMemoryProjectRegistry,
) -> None:
    for bad in ("ada.fixture@example.com", "4111111111111111"):
        with pytest.raises(InvalidProjectIdError) as exc:
            registry.require(bad)
        message = str(exc.value)
        assert bad not in message
        assert "@" not in message
        assert bad[:4] not in message


def test_duplicate_registration_is_rejected(
    registry: InMemoryProjectRegistry, alpha_id: ProjectId
) -> None:
    with pytest.raises(DuplicateProjectError):
        registry.register(ProjectConfig(alpha_id, "schemas/booking.schema.json", ("golden",)))
    assert registry.get(alpha_id) is not None


@pytest.mark.parametrize(
    "path",
    [
        "schemas/../booking.schema.json",
        "/schemas/booking.schema.json",
        "schemas\\booking.schema.json",
        "booking.schema.json",
        "schemas/booking.schema.txt",
        "schemas/booking schema.json",
        "schemas/user@host.json",
    ],
)
def test_unsafe_booking_schema_paths_are_rejected(alpha_id: ProjectId, path: str) -> None:
    with pytest.raises(InvalidProjectConfigError):
        ProjectConfig(alpha_id, path, ("golden",))


@pytest.mark.parametrize(
    "names",
    [(), ("golden", "golden"), ("Ada Fixture",), ("4111111111111111",)],
)
def test_invalid_eval_set_names_are_rejected(alpha_id: ProjectId, names: tuple[str, ...]) -> None:
    with pytest.raises((InvalidProjectConfigError, InvalidProjectIdError)):
        ProjectConfig(alpha_id, "schemas/booking.schema.json", names)


def test_non_config_registration_is_rejected() -> None:
    registry = InMemoryProjectRegistry()
    with pytest.raises(InvalidProjectConfigError):
        registry.register(cast(ProjectConfig, object()))
    assert registry.registered_ids() == ()
