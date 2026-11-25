"""Tests for typed, fail-closed P4-05 project seam settings."""

from __future__ import annotations

import pytest

from services.projects import (
    CONTRACT_NOTE,
    INTERNAL_SCOPE_LABEL,
    INTERNAL_SCOPE_VERSION,
    INVALID_PROJECT_SETTINGS_MESSAGE,
    STANDIN_LABEL,
    InvalidProjectIdError,
    ProjectId,
    ProjectScope,
    ProjectSettings,
    all_namespaces,
    load_project_settings,
)
from voiceloop_common import ConfigError


def test_defaults_are_pinned_and_labelled() -> None:
    settings = load_project_settings()
    assert settings.task_id == "P4-05"
    assert settings.evidence_label == STANDIN_LABEL
    assert settings.internal_contract_label == INTERNAL_SCOPE_LABEL
    assert settings.internal_contract_version == INTERNAL_SCOPE_VERSION
    assert settings.contract_note == CONTRACT_NOTE
    assert "no cross-service contract version bump" in CONTRACT_NOTE
    assert settings.default_project_id == ""


def test_synthetic_project_ids_are_distinct_and_valid() -> None:
    settings = load_project_settings()
    alpha, beta = settings.synthetic_project_ids()
    assert alpha != beta
    assert alpha.value == "synthetic-alpha"
    assert beta.value == "synthetic-beta"
    assert all_namespaces(ProjectScope.for_project(alpha)) != all_namespaces(
        ProjectScope.for_project(beta)
    )


def test_default_project_id_is_fail_closed_when_unset(settings: ProjectSettings) -> None:
    with pytest.raises(InvalidProjectIdError):
        settings.require_default_project_id()
    configured = load_project_settings(default_project_id="synthetic-alpha")
    assert configured.require_default_project_id() == ProjectId.parse("synthetic-alpha")


def test_invalid_synthetic_project_id_from_env_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    rejected = "ada.fixture@example.com"
    monkeypatch.setenv("VOICELOOP_PROJECTS_ALPHA_PROJECT_ID", rejected)
    with pytest.raises(ConfigError) as exc:
        load_project_settings()
    message = str(exc.value)
    assert message == INVALID_PROJECT_SETTINGS_MESSAGE
    assert rejected not in message


@pytest.mark.parametrize(
    "rejected",
    [
        "ada.fixture@example.com",
        "Ada Fixture",
        "+1-555-010-0001",
        "4111111111111111",
        "4111-1111-1111-1111",
    ],
)
def test_rejected_names_emails_phones_and_cards_never_leak(
    monkeypatch: pytest.MonkeyPatch, rejected: str
) -> None:
    monkeypatch.setenv("VOICELOOP_PROJECTS_ALPHA_PROJECT_ID", rejected)
    with pytest.raises(ConfigError) as exc:
        load_project_settings()
    message = str(exc.value)
    assert message == INVALID_PROJECT_SETTINGS_MESSAGE
    assert rejected not in message
    assert rejected not in repr(exc.value)
    assert "input_value" not in message
    assert "line_errors" not in message
    assert "validation error" not in message.lower()
    assert exc.value.__cause__ is None
    assert exc.value.__suppress_context__ is True


def test_rejected_card_in_override_is_withheld() -> None:
    rejected = "4111111111111111"
    with pytest.raises(ConfigError) as exc:
        load_project_settings(default_project_id=rejected)
    message = str(exc.value)
    assert message == INVALID_PROJECT_SETTINGS_MESSAGE
    assert rejected not in message
    assert rejected not in repr(exc.value)


def test_mandatory_labels_cannot_be_relabelled_via_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for variable, rejected in (
        ("VOICELOOP_PROJECTS_EVIDENCE_LABEL", "PRODUCTION-MULTI-PROJECT"),
        ("VOICELOOP_PROJECTS_INTERNAL_CONTRACT_VERSION", "2.0.0"),
        ("VOICELOOP_PROJECTS_INTERNAL_CONTRACT_LABEL", INTERNAL_SCOPE_VERSION),
    ):
        monkeypatch.setenv(variable, rejected)
        with pytest.raises(ConfigError) as exc:
            load_project_settings()
        message = str(exc.value)
        assert message == INVALID_PROJECT_SETTINGS_MESSAGE
        assert rejected not in message
        monkeypatch.delenv(variable, raising=False)


def test_identical_synthetic_project_ids_are_rejected_when_read() -> None:
    settings = load_project_settings(beta_project_id="synthetic-alpha")
    with pytest.raises(InvalidProjectIdError):
        settings.synthetic_project_ids()
