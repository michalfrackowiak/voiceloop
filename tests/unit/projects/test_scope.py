"""Tests for pure project namespaces and fail-closed scope guards."""

from __future__ import annotations

from typing import Any, cast

import pytest

from services.projects import (
    INTERNAL_SCOPE_LABEL,
    INTERNAL_SCOPE_VERSION,
    STANDIN_LABEL,
    STANDIN_TOKEN,
    InvalidProjectIdError,
    LabelledValue,
    MissingProjectTagError,
    ProjectId,
    ProjectScope,
    ProjectScopeError,
    ProjectScopeMismatchError,
    ProjectTag,
    all_namespaces,
    artifact_uri,
    assert_same_project,
    audio_object_key,
    dataset_id,
    mlflow_alias,
    mlflow_model_name,
    report_path,
)


@pytest.mark.parametrize(
    "value", ["synthetic-alpha", "ab", "p-1", "proj-007", "a-b-c", "room-beta-2"]
)
def test_valid_pseudonymous_slug_is_accepted(value: str) -> None:
    parsed = ProjectId.parse(value)
    assert parsed.value == value
    assert str(parsed) == value


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "",
        " ",
        "Alpha",
        "ada.fixture@example.com",
        "+1-555-010-0001",
        "4111111111111111",
        "4111-1111-1111-1111",
        "John Doe",
        "proj_1",
        "-alpha",
        "alpha-",
        "alpha--beta",
        "x" * 65,
    ],
)
def test_invalid_or_missing_ids_fail_closed_without_echo(bad: object) -> None:
    with pytest.raises(InvalidProjectIdError) as exc:
        ProjectId.parse(cast("str | ProjectId | None", bad))
    message = str(exc.value)
    if isinstance(bad, str) and len(bad) > 1 and not bad.isspace():
        assert bad not in message


def test_tampered_project_id_is_revalidated() -> None:
    tampered = ProjectId("synthetic-alpha")
    object.__setattr__(tampered, "value", "Ada@example.com")
    with pytest.raises(InvalidProjectIdError):
        ProjectId.parse(tampered)


def test_internal_tag_label_and_version_are_pinned(alpha_id: ProjectId) -> None:
    tag = ProjectTag(alpha_id)
    assert tag.label == STANDIN_LABEL
    assert tag.scope_label == INTERNAL_SCOPE_LABEL
    assert tag.scope_version == INTERNAL_SCOPE_VERSION
    assert tag.to_dict()["project_id"] == alpha_id.value
    tampered = ProjectTag(alpha_id)
    object.__setattr__(tampered, "label", "PRODUCTION-EVIDENCE")
    with pytest.raises(ProjectScopeError):
        ProjectScope(tampered)
    with pytest.raises(ProjectScopeError):
        assert_same_project(tampered)


def test_missing_or_malformed_scope_fails_closed(alpha_id: ProjectId) -> None:
    with pytest.raises(MissingProjectTagError):
        ProjectScope(cast(ProjectTag, None))
    with pytest.raises(InvalidProjectIdError):
        ProjectScope.for_project(None)
    assert ProjectScope.for_project(alpha_id).project_id == alpha_id


def test_assert_same_project_accepts_shared_tags(alpha_scope: ProjectScope) -> None:
    values = all_namespaces(alpha_scope)
    assert assert_same_project(*values.values(), alpha_scope.tag) == alpha_scope.project_id
    assert assert_same_project(alpha_scope) == alpha_scope.project_id
    assert assert_same_project(values["audio_object_key"], values["mlflow_alias"])


def test_assert_same_project_fails_on_mismatch_missing_and_foreign_tags(
    alpha_scope: ProjectScope, beta_scope: ProjectScope
) -> None:
    alpha_value = all_namespaces(alpha_scope)["dataset_id"]
    beta_value = all_namespaces(beta_scope)["dataset_id"]
    with pytest.raises(ProjectScopeMismatchError):
        assert_same_project(alpha_value, beta_value)
    with pytest.raises(MissingProjectTagError):
        assert_same_project(alpha_value, None)
    with pytest.raises(MissingProjectTagError):
        assert_same_project(alpha_value, cast(Any, object()))
    with pytest.raises(MissingProjectTagError):
        assert_same_project()


def test_every_namespace_output_carries_the_internal_label(
    alpha_scope: ProjectScope,
) -> None:
    for name, value in all_namespaces(alpha_scope).items():
        assert isinstance(value, LabelledValue), name
        assert value.label == STANDIN_LABEL
        assert str(value) == value.value
        payload = value.to_dict()
        assert payload["label"] == STANDIN_LABEL
        assert payload["scope_label"] == INTERNAL_SCOPE_LABEL
        assert payload["scope_version"] == INTERNAL_SCOPE_VERSION
        assert payload["project_id"] == alpha_scope.project_id.value
        assert STANDIN_TOKEN in value.value
        assert alpha_scope.project_id.value in value.value


def test_two_synthetic_projects_are_provably_disjoint(
    alpha_scope: ProjectScope, beta_scope: ProjectScope
) -> None:
    alpha = all_namespaces(alpha_scope)
    beta = all_namespaces(beta_scope)
    assert set(alpha) == set(beta)
    assert not (set(alpha.values()) & set(beta.values()))
    for key in alpha:
        left, right = alpha[key].value, beta[key].value
        assert left != right
        assert beta_scope.project_id.value not in left
        assert alpha_scope.project_id.value not in right
        assert not left.startswith(right)
        assert not right.startswith(left)
    audio = alpha["audio_object_key"].value
    assert audio.startswith(f"{STANDIN_TOKEN}/{alpha_scope.project_id.value}/audio/")
    assert audio.endswith("/call-0001/turn-001.wav")


def test_builders_are_pure_and_fail_closed_on_local_inputs(
    alpha_scope: ProjectScope,
) -> None:
    first = audio_object_key(alpha_scope, "call-0001", 1)
    second = audio_object_key(alpha_scope, "call-0001", 1)
    assert first == second
    assert first.tag is alpha_scope.tag
    with pytest.raises(InvalidProjectIdError):
        audio_object_key(alpha_scope, "Ada Fixture@example.com", 1)
    with pytest.raises(InvalidProjectIdError):
        audio_object_key(alpha_scope, "call-0001", 1, extension="wav;rm")
    with pytest.raises(ProjectScopeError):
        audio_object_key(alpha_scope, "call-0001", -1)
    with pytest.raises(MissingProjectTagError):
        audio_object_key(cast(ProjectScope, None), "call-0001", 1)


def test_all_builder_families_are_distinct_namespaces(alpha_scope: ProjectScope) -> None:
    values = all_namespaces(alpha_scope)
    assert dataset_id(alpha_scope, "parser-clean", "v1") == values["dataset_id"]
    assert artifact_uri(alpha_scope, "parser", "v1") == values["artifact_uri"]
    assert report_path(alpha_scope, "eval", "v1") == values["report_path"]
    assert mlflow_alias(alpha_scope, "challenger") == values["mlflow_alias"]
    assert mlflow_model_name(alpha_scope, "parser", "v1") == values["mlflow_model_name"]
