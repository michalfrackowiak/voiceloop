"""Pure, fail-closed namespace builders for the isolated P4-05 fake seam.

SYNTHETIC/FIXTURE STAND-IN - NOT A MULTI-PROJECT PRODUCTION GUARANTEE.

The internal artefact label is ``voiceloop.project_scope`` / ``internal-1.0.0``.
No cross-service contract version bump is implied or performed: these builders
are internal string namespaces only and are not consumed by any active lane.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Final, Protocol, runtime_checkable

from .ids import ProjectId, validate_project_id

STANDIN_LABEL: Final = "SYNTHETIC/FIXTURE STAND-IN - NOT A MULTI-PROJECT PRODUCTION GUARANTEE"
INTERNAL_SCOPE_LABEL: Final = "voiceloop.project_scope"
INTERNAL_SCOPE_VERSION: Final = "internal-1.0.0"
CONTRACT_NOTE: Final = (
    "Internal artefact label only; no cross-service contract version bump is implied or performed."
)


def _label_token(label: str) -> str:
    """Return a deterministic slug derived from an internal label text."""
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


STANDIN_TOKEN: Final = _label_token(STANDIN_LABEL)


class ProjectScopeError(ValueError):
    """Base error for malformed or relabelled internal project scopes."""


class MissingProjectTagError(ProjectScopeError):
    """Raised when a required internal project tag is absent."""


class ProjectScopeMismatchError(ProjectScopeError):
    """Raised when values claiming one scope actually carry different projects."""


def _assert_internal_label(label: str, scope_label: str, scope_version: str) -> None:
    """Require the exact internal label and version, without echoing inputs."""
    if label != STANDIN_LABEL:
        raise ProjectScopeError("scope label must be the mandatory stand-in label")
    if scope_label != INTERNAL_SCOPE_LABEL or scope_version != INTERNAL_SCOPE_VERSION:
        raise ProjectScopeError("unsupported internal scope label or version")


@dataclass(frozen=True, slots=True)
class ProjectTag:
    """Internal-only tag carried by every labelled output of this seam."""

    project_id: ProjectId
    label: str = STANDIN_LABEL
    scope_label: str = INTERNAL_SCOPE_LABEL
    scope_version: str = INTERNAL_SCOPE_VERSION

    def __post_init__(self) -> None:
        """Fail closed on a missing project ID or a relabelled internal tag."""
        if not isinstance(self.project_id, ProjectId):
            raise MissingProjectTagError("project tag has no valid project_id")
        _assert_internal_label(self.label, self.scope_label, self.scope_version)

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-safe, labelled description of this tag."""
        return {
            "project_id": self.project_id.value,
            "label": self.label,
            "scope_label": self.scope_label,
            "scope_version": self.scope_version,
        }


@dataclass(frozen=True, slots=True)
class ProjectScope:
    """Validated synthetic scope used by all pure namespace builders."""

    tag: ProjectTag
    prefix: str = field(init=False, repr=False, default="")

    def __post_init__(self) -> None:
        """Derive the namespace prefix from a validated internal tag."""
        if not isinstance(self.tag, ProjectTag):
            raise MissingProjectTagError("project scope has no valid project tag")
        _assert_internal_label(self.tag.label, self.tag.scope_label, self.tag.scope_version)
        object.__setattr__(self, "prefix", f"{STANDIN_TOKEN}/{self.tag.project_id.value}")

    @property
    def project_id(self) -> ProjectId:
        """Return the validated pseudonymous project ID."""
        return self.tag.project_id

    @classmethod
    def for_project(cls, project_id: str | ProjectId | None) -> ProjectScope:
        """Build a scope from a candidate ID, failing closed on invalid input."""
        return cls(ProjectTag(ProjectId.parse(project_id)))


@dataclass(frozen=True, slots=True)
class LabelledValue:
    """A namespace string with its mandatory internal stand-in label."""

    value: str
    tag: ProjectTag
    label: str = field(init=False, repr=False, default=STANDIN_LABEL)

    def __post_init__(self) -> None:
        """Verify that the string is labelled and carries its project slug."""
        if not isinstance(self.tag, ProjectTag):
            raise MissingProjectTagError("labelled value has no valid project tag")
        _assert_internal_label(self.tag.label, self.tag.scope_label, self.tag.scope_version)
        if not isinstance(self.value, str) or not self.value:
            raise ProjectScopeError("labelled value must be a non-empty string")
        if STANDIN_TOKEN not in self.value or self.tag.project_id.value not in self.value:
            raise ProjectScopeError("labelled value is outside its project namespace")
        object.__setattr__(self, "label", STANDIN_LABEL)

    def __str__(self) -> str:
        """Return the namespace string itself."""
        return self.value

    def to_dict(self) -> dict[str, str]:
        """Return the namespace string with all internal label metadata."""
        return {"value": self.value, **self.tag.to_dict(), "label": STANDIN_LABEL}


@runtime_checkable
class Tagged(Protocol):
    """Structural type for values carrying an internal project tag."""

    @property
    def tag(self) -> ProjectTag:
        """Return the internal project tag."""
        ...


def _tag_of(item: Tagged | ProjectTag | None) -> ProjectTag:
    """Extract and re-validate a tag from a tagged value, failing closed."""
    if item is None:
        raise MissingProjectTagError("missing project tag")
    if isinstance(item, ProjectTag):
        tag = item
    else:
        candidate = getattr(item, "tag", None)
        if not isinstance(candidate, ProjectTag):
            raise MissingProjectTagError("tagged value has no valid project tag")
        tag = candidate
    _assert_internal_label(tag.label, tag.scope_label, tag.scope_version)
    if not isinstance(tag.project_id, ProjectId):
        raise MissingProjectTagError("project tag has no valid project_id")
    return tag


def assert_same_project(*items: Tagged | ProjectTag | None) -> ProjectId:
    """Return the shared project ID, failing closed on missing or differing tags.

    Args:
        *items: Labelled values, explicit tags, or ``None`` (which fails closed).

    Returns:
        The single validated project ID shared by every supplied tag.

    Raises:
        MissingProjectTagError: If nothing is supplied or a tag is absent.
        ProjectScopeError: If a tag carries a foreign label or version.
        ProjectScopeMismatchError: If the tags name different projects.

    """
    if not items:
        raise MissingProjectTagError("at least one project tag is required")
    tags = tuple(_tag_of(item) for item in items)
    expected = tags[0].project_id
    if any(tag.project_id != expected for tag in tags[1:]):
        raise ProjectScopeMismatchError("project tags do not agree")
    return expected


def _segment(value: object, field: str) -> str:
    """Validate a local namespace segment with the pseudonym-safe slug rules."""
    return validate_project_id(value, field=field)


def _index(value: object, field: str) -> str:
    """Validate a non-negative index and render it as a zero-padded segment."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 999_999:
        raise ProjectScopeError(f"{field} must be a non-negative integer at most 999999")
    return f"{value:03d}"


def audio_object_key(
    scope: ProjectScope,
    call_id: str,
    turn_index: int,
    *,
    extension: str = "wav",
) -> LabelledValue:
    """Build a labelled, project-scoped synthetic audio object key.

    Args:
        scope: Validated synthetic project scope.
        call_id: Pseudonymous synthetic call ID.
        turn_index: Non-negative turn index.
        extension: Low-risk file extension from the pseudonym slug alphabet.

    Returns:
        The labelled object key.

    """
    if not isinstance(scope, ProjectScope):
        raise MissingProjectTagError("namespace builder requires a validated project scope")
    call = _segment(call_id, "call_id")
    turn = _index(turn_index, "turn_index")
    ext = _segment(extension, "extension")
    value = f"{scope.prefix}/audio/{call}/turn-{turn}.{ext}"
    return LabelledValue(value, scope.tag)


def dataset_id(scope: ProjectScope, dataset_name: str, version: str) -> LabelledValue:
    """Build a labelled, project-scoped synthetic dataset ID."""
    if not isinstance(scope, ProjectScope):
        raise MissingProjectTagError("namespace builder requires a validated project scope")
    name = _segment(dataset_name, "dataset_name")
    ver = _segment(version, "version")
    return LabelledValue(f"{STANDIN_TOKEN}-{scope.project_id.value}-{name}-{ver}", scope.tag)


def artifact_uri(scope: ProjectScope, artifact_name: str, version: str) -> LabelledValue:
    """Build a labelled, project-scoped fake artifact URI."""
    if not isinstance(scope, ProjectScope):
        raise MissingProjectTagError("namespace builder requires a validated project scope")
    name = _segment(artifact_name, "artifact_name")
    ver = _segment(version, "version")
    return LabelledValue(
        f"fake://voiceloop-artifacts/{scope.prefix}/{name}/{ver}",
        scope.tag,
    )


def report_path(scope: ProjectScope, report_name: str, version: str) -> LabelledValue:
    """Build a labelled, project-scoped synthetic report path."""
    if not isinstance(scope, ProjectScope):
        raise MissingProjectTagError("namespace builder requires a validated project scope")
    name = _segment(report_name, "report_name")
    ver = _segment(version, "version")
    return LabelledValue(f"reports/{scope.prefix}/{name}/{ver}.json", scope.tag)


def mlflow_alias(scope: ProjectScope, alias: str) -> LabelledValue:
    """Build a labelled, project-scoped fake MLflow alias name."""
    if not isinstance(scope, ProjectScope):
        raise MissingProjectTagError("namespace builder requires a validated project scope")
    name = _segment(alias, "alias")
    return LabelledValue(f"{STANDIN_TOKEN}-{scope.project_id.value}-{name}", scope.tag)


def mlflow_model_name(scope: ProjectScope, model_name: str, version: str) -> LabelledValue:
    """Build a labelled, project-scoped fake registered model name."""
    if not isinstance(scope, ProjectScope):
        raise MissingProjectTagError("namespace builder requires a validated project scope")
    name = _segment(model_name, "model_name")
    ver = _segment(version, "version")
    return LabelledValue(f"{STANDIN_TOKEN}-{scope.project_id.value}-{name}-{ver}", scope.tag)


def all_namespaces(scope: ProjectScope) -> dict[str, LabelledValue]:
    """Return one labelled output from every builder for a fixed local name set.

    Args:
        scope: Validated synthetic project scope.

    Returns:
        Mapping from builder name to its labelled output, used by disjointness
        tests to compare two synthetic projects across every namespace.

    """
    return {
        "audio_object_key": audio_object_key(scope, "call-0001", 1),
        "dataset_id": dataset_id(scope, "parser-clean", "v1"),
        "artifact_uri": artifact_uri(scope, "parser", "v1"),
        "report_path": report_path(scope, "eval", "v1"),
        "mlflow_alias": mlflow_alias(scope, "challenger"),
        "mlflow_model_name": mlflow_model_name(scope, "parser", "v1"),
    }
