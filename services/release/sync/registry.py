"""Read-only champion alias polling seam.

> [!IMPORTANT]
> **FAKE ONLY unless an adapter is injected.** This package never writes an
> alias and never promotes or demotes a model. The gate's alias state is an
> input to :class:`ChampionRegistry`; ``context.md`` section 2 invariant 10
> keeps promotion and rollback as gate-owned one-step alias flips.

A real MLflow adapter implements :meth:`ChampionRegistry.poll` with
``MlflowClient.get_model_version_by_alias(...)``, translating the returned
version, source URI and SHA-256 tag into a :class:`ChampionSnapshot`. MLflow is
deliberately not imported here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from voiceloop_common import ValidationError

from .artifacts import ArtifactRef

CHAMPION_ALIAS = "@champion"


class AliasState(StrEnum):
    """State of the observed champion alias, independent of its version."""

    BOUND = "bound"
    UNBOUND = "unbound"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ChampionSnapshot:
    """One immutable observation of the champion alias and its artifact."""

    alias: str = CHAMPION_ALIAS
    version: str | None = None
    state: AliasState = AliasState.UNBOUND
    artifact: ArtifactRef | None = None

    def __post_init__(self) -> None:
        """Keep version and alias state mutually consistent."""
        if not isinstance(self.alias, str) or not self.alias.strip():
            raise ValidationError("alias must be a non-blank string")
        if not isinstance(self.state, AliasState):
            raise ValidationError("state must be an AliasState")
        if self.state is AliasState.BOUND:
            if not isinstance(self.version, str) or not self.version.strip():
                raise ValidationError("a bound champion alias must name a version")
        elif self.version is not None:
            raise ValidationError("an unbound or unknown alias cannot name a version")

    def identity(self) -> tuple[str, str | None, AliasState]:
        """Return the change-detection key: alias, version and alias state."""
        return (self.alias, self.version, self.state)

    def changed_from(self, previous: ChampionSnapshot | None) -> bool:
        """Return True when version or alias state differs from ``previous``."""
        return previous is None or self.identity() != previous.identity()

    @classmethod
    def bound(
        cls, version: str, artifact: ArtifactRef | None, alias: str = CHAMPION_ALIAS
    ) -> ChampionSnapshot:
        """Build a snapshot for an alias bound to ``version``."""
        return cls(alias=alias, version=version, state=AliasState.BOUND, artifact=artifact)

    @classmethod
    def unbound(cls, alias: str = CHAMPION_ALIAS) -> ChampionSnapshot:
        """Build a snapshot for an alias with no bound version."""
        return cls(alias=alias, version=None, state=AliasState.UNBOUND)


class ChampionRegistry(Protocol):
    """Poll-only seam for champion alias observations."""

    def poll(self) -> ChampionSnapshot:
        """Return the current champion alias observation."""
        ...


class ReadOnlyAliasStore(Protocol):
    """Read-only subset of the gate's ``AliasStore`` (``get`` only)."""

    def get(self, alias: str) -> str | None:
        """Return the version bound to ``alias`` or None when unbound."""
        ...


class GateAliasChampionRegistry:
    """Read-only adapter over the P2-09 gate alias store.

    The injected store must expose ``get`` only, so this adapter cannot promote
    or roll back anything. Artifact metadata is supplied out of band because
    the gate's alias store holds versions, not artifact URIs or hashes.
    """

    def __init__(
        self,
        aliases: ReadOnlyAliasStore,
        artifacts: Mapping[str, ArtifactRef],
        alias: str = CHAMPION_ALIAS,
    ) -> None:
        """Initialise from a read-only alias store and version metadata."""
        if not isinstance(alias, str) or not alias.strip():
            raise ValidationError("alias must be a non-blank string")
        self._aliases = aliases
        self._artifacts = dict(artifacts)
        self._alias = alias

    def poll(self) -> ChampionSnapshot:
        """Read the champion version without writing to the alias store."""
        version = self._aliases.get(self._alias)
        if version is None:
            return ChampionSnapshot.unbound(self._alias)
        return ChampionSnapshot.bound(version, self._artifacts.get(version), self._alias)


class FakeChampionRegistry:
    """Deterministic FAKE registry whose snapshots change only via ``publish``."""

    def __init__(self, initial: ChampionSnapshot | None = None) -> None:
        """Initialise with an optional first snapshot (unbound by default)."""
        self._snapshot = initial if initial is not None else ChampionSnapshot.unbound()
        self.poll_count = 0

    def publish(self, snapshot: ChampionSnapshot) -> None:
        """Replace the snapshot returned by future polls."""
        if not isinstance(snapshot, ChampionSnapshot):
            raise ValidationError("snapshot must be a ChampionSnapshot")
        self._snapshot = snapshot

    def bind(self, version: str, artifact: ArtifactRef | None = None) -> None:
        """Publish a bound champion alias with optional artifact metadata."""
        self.publish(ChampionSnapshot.bound(version, artifact))

    def unbind(self) -> None:
        """Publish an unbound champion alias."""
        self.publish(ChampionSnapshot.unbound())

    def poll(self) -> ChampionSnapshot:
        """Return the current snapshot and count the poll."""
        self.poll_count += 1
        return self._snapshot
