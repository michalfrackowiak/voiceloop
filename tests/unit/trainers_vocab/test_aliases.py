"""Alias derivation and canonicalisation tests."""

from __future__ import annotations

from typing import Any

import pytest

from services.trainers.vocab import AliasTable, derive_alias_table
from voiceloop_common import ValidationError, load_schema


def test_name_alias_derived_from_misheard_surname_fixture(alias_table: AliasTable) -> None:
    names = [entry for entry in alias_table.entries if entry.kind == "name"]
    assert len(names) == 1
    entry = names[0]
    assert entry.surface == "smythe"
    assert entry.canonical == "Smith"
    assert entry.source_case_id == "misheard_surname"
    assert entry.source_call_id == "call-0002"
    assert entry.source_turn_index == 2
    assert "read-only" in entry.derivation


def test_name_alias_fixes_the_fixture_slot_without_touching_clean_names(
    alias_table: AliasTable,
) -> None:
    assert alias_table.apply_text("Mr. Smythe.") == "Mr. Smith."
    assert alias_table.apply_text("SMYTHE") == "SMITH"
    assert alias_table.canonicalise_slots({"guest_name": "Mr. Smythe"})["guest_name"] == (
        "Mr. Smith"
    )
    assert alias_table.canonicalise_slots({"guest_name": "Ada Fixture"})["guest_name"] == (
        "Ada Fixture"
    )


def test_room_aliases_come_from_observed_transcript_phrases(alias_table: AliasTable) -> None:
    rooms = {
        entry.surface: entry.canonical for entry in alias_table.entries if entry.kind == "room_type"
    }
    assert rooms == {"deluxe room": "deluxe", "standard room": "standard"}
    assert alias_table.canonicalise_slots({"room_type": "deluxe room"})["room_type"] == "deluxe"
    assert alias_table.canonicalise_slots({"room_type": "suite"})["room_type"] == "suite"


def test_longest_surface_wins_and_inputs_are_not_mutated() -> None:
    table = AliasTable(entries=(*derive_alias_table("tests/fixtures/turns.jsonl").entries,))
    original: dict[str, Any] = {"guest_name": "Mr. Smythe", "room_type": "deluxe room"}
    snapshot = dict(original)
    result = table.canonicalise_slots(original)
    assert result == {"guest_name": "Mr. Smith", "room_type": "deluxe"}
    assert original == snapshot


def test_manual_overrides_are_labelled_with_settings_provenance() -> None:
    table = derive_alias_table(
        "tests/fixtures/turns.jsonl",
        name_overrides={"Foo": "Bar"},
        room_overrides={"delux": "deluxe"},
    )
    foo = [entry for entry in table.entries if entry.surface == "foo"]
    deluxe = [entry for entry in table.entries if entry.surface == "delux"]
    assert foo[0].canonical == "Bar"
    assert deluxe[0].canonical == "deluxe"
    assert all("VOICELOOP_VOCAB_" in entry.derivation for entry in (foo[0], deluxe[0]))


def test_room_override_rejects_canonical_outside_booking_schema() -> None:
    with pytest.raises(ValidationError, match="invalid room_type"):
        derive_alias_table(
            "tests/fixtures/turns.jsonl",
            room_overrides={"foo": "not-a-room"},
        )


def test_room_override_accepts_schema_room_type_and_defaults_stay_valid() -> None:
    table = derive_alias_table(
        "tests/fixtures/turns.jsonl",
        room_overrides={"suite please": "SUITE"},
    )
    assert table.canonicalise_slots({"room_type": "suite please"})["room_type"] == "suite"
    schema = load_schema("booking")
    enum = frozenset(str(value).casefold() for value in schema["properties"]["room_type"]["enum"])
    assert enum == {"standard", "deluxe", "suite"}
    assert all(
        entry.canonical.casefold() in enum for entry in table.entries if entry.kind == "room_type"
    )
