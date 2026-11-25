"""Fail-closed project registry protocol and in-memory fake for P4-05.

SYNTHETIC/FIXTURE STAND-IN - NOT A MULTI-PROJECT PRODUCTION GUARANTEE.
This maps a validated pseudonymous project ID to a booking schema path, a set
of eval-set names and an enabled flag. It is an internal fake seam only: the
internal artefact label is ``voiceloop.project_scope`` / ``internal-1.0.0`` and
no cross-service contract version bump is implied or performed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .ids import InvalidProjectIdError, ProjectId, validate_project_id
from .scope import INTERNAL_SCOPE_LABEL, INTERNAL_SCOPE_VERSION, STANDIN_LABEL, ProjectTag

_SCHEMA_SEGMENT_RE = re.compile(r"[A-Za-z0-9_.-]+")


class ProjectRegistryError(ValueError):
    """Base error for invalid or unresolved project registry entries."""


class UnknownProjectError(ProjectRegistryError):
    """Raised when a project ID is not registered."""


class DisabledProjectError(ProjectRegistryError):
    """Raised when a registered project is disabled."""


class DuplicateProjectError(ProjectRegistryError):
    """Raised when a project ID is registered twice."""


class InvalidProjectConfigError(ProjectRegistryError):
    """Raised when a project config carries an unsafe or missing value."""


def _validate_schema_path(value: object) -> str:
    """Return a safe repo-relative JSON schema path, failing closed.

    Args:
        value: Candidate booking schema path.

    Returns:
        The validated POSIX-style relative path.

    Raises:
        InvalidProjectConfigError: If the path is absolute, traverses upwards,
            carries unsafe characters, or is not a JSON file.

    """
    if not isinstance(value, str) or not value or len(value) > 200:
        raise InvalidProjectConfigError("booking_schema_path must be a non-empty relative path")
    if (
        "\\" in value
        or value.startswith("/")
        or ".." in value
        or "@" in value
        or any(char.isspace() for char in value)
    ):
        raise InvalidProjectConfigError("booking_schema_path has an unsafe path shape")
    segments = value.split("/")
    if len(segments) < 2 or any(_SCHEMA_SEGMENT_RE.fullmatch(part) is None for part in segments):
        raise InvalidProjectConfigError("booking_schema_path has an unsafe segment")
    if not value.endswith(".json"):
        raise InvalidProjectConfigError("booking_schema_path must point to a JSON schema")
    return value


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Booking schema, eval sets and enabled flag for one synthetic project."""

    project_id: ProjectId
    booking_schema_path: str
    eval_set_names: tuple[str, ...]
    enabled: bool = True
    tag: ProjectTag = field(init=False, repr=False)
    label: str = field(init=False, repr=False, default=STANDIN_LABEL)

    def __post_init__(self) -> None:
        """Validate all fields and attach the mandatory internal tag."""
        if not isinstance(self.project_id, ProjectId):
            raise InvalidProjectConfigError("project_id must be a validated ProjectId")
        object.__setattr__(
            self, "booking_schema_path", _validate_schema_path(self.booking_schema_path)
        )
        if not isinstance(self.eval_set_names, tuple) or not self.eval_set_names:
            raise InvalidProjectConfigError("at least one eval set name is required")
        names = tuple(
            validate_project_id(name, field="eval_set_name") for name in self.eval_set_names
        )
        if len(set(names)) != len(names):
            raise InvalidProjectConfigError("eval set names must be unique")
        object.__setattr__(self, "eval_set_names", names)
        if not isinstance(self.enabled, bool):
            raise InvalidProjectConfigError("enabled must be a boolean")
        object.__setattr__(self, "tag", ProjectTag(self.project_id))
        object.__setattr__(self, "label", STANDIN_LABEL)

    def to_dict(self) -> dict[str, object]:
        """Return a labelled JSON-safe view of this configuration."""
        return {
            "booking_schema_path": self.booking_schema_path,
            "enabled": self.enabled,
            "eval_set_names": list(self.eval_set_names),
            **self.tag.to_dict(),
            "label": STANDIN_LABEL,
        }


@runtime_checkable
class ProjectRegistry(Protocol):
    """Structural contract for looking up synthetic project configurations."""

    def get(self, project_id: str | ProjectId | None) -> ProjectConfig | None:
        """Return an enabled config or ``None`` for invalid, unknown or disabled IDs."""
        ...

    def require(self, project_id: str | ProjectId | None) -> ProjectConfig:
        """Return an enabled config or raise a fixed-code error."""
        ...

    def enabled_ids(self) -> tuple[ProjectId, ...]:
        """Return all enabled project IDs in deterministic order."""
        ...


class InMemoryProjectRegistry:
    """Deterministic in-memory fake implementing :class:`ProjectRegistry`."""

    def __init__(self, configs: Iterable[ProjectConfig] = ()) -> None:
        """Build a registry from validated configs, rejecting duplicate IDs.

        Args:
            configs: Project configurations to register.

        Raises:
            DuplicateProjectError: If a project ID appears more than once.

        """
        self._configs: dict[ProjectId, ProjectConfig] = {}
        for config in configs:
            self.register(config)

    def register(self, config: ProjectConfig) -> None:
        """Add one validated config, failing closed on a duplicate ID.

        Args:
            config: Project configuration to register.

        Raises:
            InvalidProjectConfigError: If the value is not a validated config.
            DuplicateProjectError: If the project ID is already registered.

        """
        if not isinstance(config, ProjectConfig):
            raise InvalidProjectConfigError("registry accepts only validated ProjectConfig values")
        if config.project_id in self._configs:
            raise DuplicateProjectError("project_id is already registered")
        self._configs[config.project_id] = config

    def get(self, project_id: str | ProjectId | None) -> ProjectConfig | None:
        """Return an enabled config or ``None`` without raising.

        Args:
            project_id: Candidate project ID.

        Returns:
            The config when the ID is valid, registered and enabled; otherwise
            ``None``. Callers must treat ``None`` as a refusal to serve.

        """
        try:
            parsed = ProjectId.parse(project_id)
        except InvalidProjectIdError:
            return None
        config = self._configs.get(parsed)
        return config if config is not None and config.enabled else None

    def require(self, project_id: str | ProjectId | None) -> ProjectConfig:
        """Return an enabled config or fail closed with a fixed-code error.

        Args:
            project_id: Candidate project ID.

        Returns:
            The validated config for the enabled project.

        Raises:
            InvalidProjectIdError: If the ID is missing, malformed or unsafe.
            UnknownProjectError: If the ID is not registered.
            DisabledProjectError: If the project is registered but disabled.

        """
        parsed = ProjectId.parse(project_id)
        config = self._configs.get(parsed)
        if config is None:
            raise UnknownProjectError("project_id is not registered")
        if not config.enabled:
            raise DisabledProjectError("project is disabled")
        return config

    def enabled_ids(self) -> tuple[ProjectId, ...]:
        """Return enabled project IDs in deterministic order."""
        return tuple(sorted(pid for pid, config in self._configs.items() if config.enabled))

    def registered_ids(self) -> tuple[ProjectId, ...]:
        """Return every registered project ID in deterministic order."""
        return tuple(sorted(self._configs))


__all__ = [
    "INTERNAL_SCOPE_LABEL",
    "INTERNAL_SCOPE_VERSION",
    "STANDIN_LABEL",
    "DisabledProjectError",
    "DuplicateProjectError",
    "InvalidProjectConfigError",
    "InMemoryProjectRegistry",
    "ProjectConfig",
    "ProjectRegistry",
    "ProjectRegistryError",
    "UnknownProjectError",
]
