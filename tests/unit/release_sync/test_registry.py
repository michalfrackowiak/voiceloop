"""P3-06: alias change detection is version + alias state, read-only."""

from __future__ import annotations

import pytest

from services.evalgate.gate import InMemoryAliasStore
from services.release.sync import (
    AliasState,
    ArtifactRef,
    ChampionSnapshot,
    FakeChampionRegistry,
    GateAliasChampionRegistry,
)
from voiceloop_common import ValidationError

HASH_A = "a" * 64
HASH_B = "b" * 64


def test_identity_uses_version_and_alias_state() -> None:
    ref_a = ArtifactRef("fake://a", HASH_A)
    ref_b = ArtifactRef("fake://b", HASH_B)
    v1 = ChampionSnapshot.bound("v1", ref_a)
    v1_other_artifact = ChampionSnapshot.bound("v1", ref_b)
    v2 = ChampionSnapshot.bound("v2", ref_a)
    unbound = ChampionSnapshot.unbound()

    assert v1.identity() == ("@champion", "v1", AliasState.BOUND)
    assert v1.changed_from(None)
    assert not v1.changed_from(v1)
    assert not v1.changed_from(v1_other_artifact)  # artifact metadata is not identity
    assert v2.changed_from(v1)
    assert unbound.changed_from(v1)
    assert ChampionSnapshot.bound("v1", ref_a, alias="@other").changed_from(v1)


def test_snapshot_validation() -> None:
    with pytest.raises(ValidationError):
        ChampionSnapshot.bound("  ", None)
    with pytest.raises(ValidationError):
        ChampionSnapshot(version="v1", state=AliasState.UNBOUND)
    with pytest.raises(ValidationError):
        ChampionSnapshot(alias="", version=None, state=AliasState.UNBOUND)


def test_gate_alias_adapter_is_read_only() -> None:
    store = InMemoryAliasStore({"@champion": "v2", "@challenger": "v3"})
    metadata = {"v2": ArtifactRef("fake://mlflow/v2", HASH_A)}
    registry = GateAliasChampionRegistry(store, metadata)

    snapshot = registry.poll()

    assert snapshot.version == "v2"
    assert snapshot.state is AliasState.BOUND
    assert snapshot.artifact == metadata["v2"]
    assert store.snapshot() == {"@champion": "v2", "@challenger": "v3"}

    store.set("@champion", "v3")
    assert registry.poll() == ChampionSnapshot.bound("v3", None)


def test_gate_alias_adapter_handles_unbound_alias() -> None:
    registry = GateAliasChampionRegistry(InMemoryAliasStore({}), {})

    assert registry.poll() == ChampionSnapshot.unbound()


def test_fake_registry_changes_only_on_publish() -> None:
    registry = FakeChampionRegistry()
    assert registry.poll() == ChampionSnapshot.unbound()

    registry.bind("v4", ArtifactRef("fake://mlflow/v4", HASH_B))
    first = registry.poll()
    second = registry.poll()

    assert first == second
    assert first.version == "v4"
    assert registry.poll_count == 3

    registry.unbind()
    assert registry.poll().state is AliasState.UNBOUND
