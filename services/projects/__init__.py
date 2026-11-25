"""P4-05 isolated fake project-scope seam.

SYNTHETIC/FIXTURE STAND-IN - NOT A MULTI-PROJECT PRODUCTION GUARANTEE.

This package is an internal rehearsal seam only. It uses the internal artefact
label ``voiceloop.project_scope`` / ``internal-1.0.0``; no cross-service
contract version bump is implied or performed. STATUS: NOT DONE / BLOCKED
pending the lead-owned contract bump and cross-service wiring described in
``docs/handoffs/P4-05.md``. Nothing here is imported by, or imports from, an
active service lane.
"""

from __future__ import annotations

from .ids import InvalidProjectIdError, ProjectId, validate_project_id
from .registry import (
    DisabledProjectError,
    DuplicateProjectError,
    InMemoryProjectRegistry,
    InvalidProjectConfigError,
    ProjectConfig,
    ProjectRegistry,
    ProjectRegistryError,
    UnknownProjectError,
)
from .scope import (
    CONTRACT_NOTE,
    INTERNAL_SCOPE_LABEL,
    INTERNAL_SCOPE_VERSION,
    STANDIN_LABEL,
    STANDIN_TOKEN,
    LabelledValue,
    MissingProjectTagError,
    ProjectScope,
    ProjectScopeError,
    ProjectScopeMismatchError,
    ProjectTag,
    Tagged,
    all_namespaces,
    artifact_uri,
    assert_same_project,
    audio_object_key,
    dataset_id,
    mlflow_alias,
    mlflow_model_name,
    report_path,
)
from .settings import ProjectSettings

__all__ = [
    "CONTRACT_NOTE",
    "INTERNAL_SCOPE_LABEL",
    "INTERNAL_SCOPE_VERSION",
    "STANDIN_LABEL",
    "STANDIN_TOKEN",
    "DisabledProjectError",
    "DuplicateProjectError",
    "InvalidProjectConfigError",
    "InvalidProjectIdError",
    "InMemoryProjectRegistry",
    "LabelledValue",
    "MissingProjectTagError",
    "ProjectConfig",
    "ProjectId",
    "ProjectRegistry",
    "ProjectRegistryError",
    "ProjectScope",
    "ProjectScopeError",
    "ProjectScopeMismatchError",
    "ProjectSettings",
    "ProjectTag",
    "Tagged",
    "UnknownProjectError",
    "all_namespaces",
    "artifact_uri",
    "assert_same_project",
    "audio_object_key",
    "dataset_id",
    "mlflow_alias",
    "mlflow_model_name",
    "report_path",
    "validate_project_id",
]
