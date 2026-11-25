"""Tests for typed, fail-closed P4-05 project seam settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from services.projects import (
    CONTRACT_NOTE,
    INTERNAL_SCOPE_LABEL,
    INTERNAL_SCOPE_VERSION,
    STANDIN_LABEL,
    InvalidProjectIdError,
    ProjectId,
    ProjectScope,
    ProjectSettings,
    all_namespaces,
)


def test_defaults_are_pinned_and_labelled() -> None:
    settings = ProjectSettings()
    assert settings.task_id == "P4-05"
    assert settings.evidence_label == STANDIN_LABEL
    assert settings.internal_contract_label == INTERNAL_SCOPE_LABEL
    assert settings.internal_contract_version == INTERNAL_SCOPE_VERSION
    assert settings.contract_note == CONTRACT_NOTE
    assert "no cross-service contract version bump" in CONTRACT_NOTE
    assert settings.default_project_id == ""


def test_synthetic_project_ids_are_distinct_and_valid() -> None:
    settings = ProjectSettings()
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
    configured = ProjectSettings(default_project_id="synthetic-alpha")
    assert configured.require_default_project_id() == ProjectId.parse("synthetic-alpha")


def test_invalid_synthetic_project_id_from_env_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICELOOP_PROJECTS_ALPHA_PROJECT_ID", "ada.fixture@example.com")
    with pytest.raises(ValidationError):
        ProjectSettings()


def test_invalid_default_project_id_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ProjectSettings(default_project_id="4111111111111111")


def test_mandatory_labels_cannot_be_relabelled_via_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VOICELOOP_PROJECTS_EVIDENCE_LABEL", "PRODUCTION-MULTI-PROJECT")
    with pytest.raises(ValidationError):
        ProjectSettings()
    monkeypatch.setenv("VOICELOOP_PROJECTS_INTERNAL_CONTRACT_VERSION", "2.0.0")
    with pytest.raises(ValidationError):
        ProjectSettings()
    monkeypatch.setenv("VOICELOOP_PROJECTS_INTERNAL_CONTRACT_LABEL", INTERNAL_SCOPE_VERSION)
    with pytest.raises(ValidationError):
        ProjectSettings()


def test_identical_synthetic_project_ids_are_rejected_when_read() -> None:
    settings = ProjectSettings(beta_project_id="synthetic-alpha")
    with pytest.raises(InvalidProjectIdError):
        settings.synthetic_project_ids()
