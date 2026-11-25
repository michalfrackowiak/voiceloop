"""Typed settings for the isolated P4-05 project-scope fake seam.

SYNTHETIC/FIXTURE STAND-IN - NOT A MULTI-PROJECT PRODUCTION GUARANTEE.
The internal artefact label is ``voiceloop.project_scope`` / ``internal-1.0.0``.
No cross-service contract version bump is implied or performed. Only
``VOICELOOP_PROJECTS_`` environment variables are read; no secrets, real tenant
names, emails, phone numbers or card values exist here. Callers must use
:func:`load_project_settings`; the settings class is a typed internal detail
whose raw ``ValidationError`` would echo rejected values.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError, ValidationInfo, field_validator

from voiceloop_common import BaseSettings, ConfigError, settings_config

from .ids import InvalidProjectIdError, ProjectId, validate_project_id
from .scope import CONTRACT_NOTE, INTERNAL_SCOPE_LABEL, INTERNAL_SCOPE_VERSION, STANDIN_LABEL

DEFAULT_ALPHA_PROJECT_ID = "synthetic-alpha"
DEFAULT_BETA_PROJECT_ID = "synthetic-beta"
INVALID_PROJECT_SETTINGS_MESSAGE = "project settings are invalid; rejected values are withheld"


class ProjectSettings(BaseSettings):
    """Typed internal configuration for the two synthetic rehearsal projects.

    Construct through :func:`load_project_settings`, which replaces pydantic
    errors with a fixed-code :class:`~voiceloop_common.errors.ConfigError`.
    """

    model_config = settings_config("VOICELOOP_PROJECTS_")

    task_id: str = "P4-05"
    alpha_project_id: str = DEFAULT_ALPHA_PROJECT_ID
    beta_project_id: str = DEFAULT_BETA_PROJECT_ID
    default_project_id: str = ""
    evidence_label: str = STANDIN_LABEL
    internal_contract_label: str = INTERNAL_SCOPE_LABEL
    internal_contract_version: str = INTERNAL_SCOPE_VERSION

    @field_validator("alpha_project_id", "beta_project_id")
    @classmethod
    def _synthetic_ids_are_pseudonyms(cls, value: str) -> str:
        """Require both synthetic project IDs to validate as pseudonyms."""
        validate_project_id(value, field="project_id")
        return value

    @field_validator("default_project_id")
    @classmethod
    def _default_is_optional(cls, value: str) -> str:
        """Allow an unset default, but reject any non-empty invalid ID."""
        if value:
            validate_project_id(value, field="project_id")
        return value

    @field_validator("evidence_label")
    @classmethod
    def _evidence_label_is_pinned(cls, value: str) -> str:
        """Refuse any attempt to relabel outputs as production evidence."""
        if value != STANDIN_LABEL:
            raise ValueError("evidence_label must remain the mandatory stand-in label")
        return value

    @field_validator("internal_contract_label", "internal_contract_version")
    @classmethod
    def _internal_contract_is_pinned(cls, value: str, info: ValidationInfo) -> str:
        """Refuse any change to the internal-only scope label or version."""
        expected = (
            INTERNAL_SCOPE_LABEL
            if info.field_name == "internal_contract_label"
            else INTERNAL_SCOPE_VERSION
        )
        if value != expected:
            raise ValueError("internal scope label and version are pinned")
        return value

    @field_validator("task_id")
    @classmethod
    def _task_id_is_nonblank(cls, value: str) -> str:
        """Require a non-blank roadmap task ID."""
        if not value.strip():
            raise ValueError("task_id must be non-blank")
        return value

    def require_default_project_id(self) -> ProjectId:
        """Return the configured default ID or fail closed when it is unset.

        Returns:
            The validated default project ID.

        Raises:
            InvalidProjectIdError: If no default was configured.

        """
        return ProjectId.parse(self.default_project_id, field="default_project_id")

    def synthetic_project_ids(self) -> tuple[ProjectId, ProjectId]:
        """Return the two distinct validated synthetic project IDs.

        Returns:
            The alpha and beta project IDs.

        Raises:
            InvalidProjectIdError: If the configured IDs are identical.

        """
        alpha = ProjectId.parse(self.alpha_project_id, field="alpha_project_id")
        beta = ProjectId.parse(self.beta_project_id, field="beta_project_id")
        if alpha == beta:
            raise InvalidProjectIdError("synthetic projects must be distinct")
        return alpha, beta

    @property
    def contract_note(self) -> str:
        """Return the explicit no-contract-bump note attached to this seam."""
        return CONTRACT_NOTE


def load_project_settings(**overrides: Any) -> ProjectSettings:
    """Load validated project settings without echoing rejected values.

    Args:
        **overrides: Keyword values forwarded to :class:`ProjectSettings` for
            deterministic tests; production callers pass none.

    Returns:
        Validated settings from ``VOICELOOP_PROJECTS_`` sources and overrides.

    Raises:
        ConfigError: If any setting is invalid. The message is fixed and never
            contains the rejected value, pydantic details or error locations.

    """
    try:
        return ProjectSettings(**overrides)
    except ValidationError:
        # Suppress the pydantic context so neither str(exc) nor a traceback can
        # disclose rejected names, emails, phone numbers, cards or locations.
        raise ConfigError(INVALID_PROJECT_SETTINGS_MESSAGE) from None
