"""Fixture-derived alias tables for guest names and room types.

Every entry records the read-only synthetic fixture turn it came from. The
default tables are derived at call time from ``tests/fixtures/turns.jsonl`` and
its event files; no fixture data is invented here. Name aliases come from the
``misheard_surname`` labelled failure (observed STT vs expected slot value),
and room-type aliases come from surface phrases in expected transcripts paired
with their labelled ``room_type`` slot.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal

from voiceloop_common import ValidationError, load_schema

AliasKind = Literal["name", "room_type"]
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIXTURES = "tests/fixtures/turns.jsonl"
_TOKEN_RE = re.compile(r"[A-Za-z]+")


@dataclass(frozen=True)
class AliasEntry:
    """One observed surface form mapped to its canonical fixture value."""

    surface: str
    canonical: str
    kind: AliasKind
    source_case_id: str | None
    source_call_id: str
    source_turn_index: int
    derivation: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record including its fixture provenance."""
        return {
            "surface": self.surface,
            "canonical": self.canonical,
            "kind": self.kind,
            "source_case_id": self.source_case_id,
            "source_call_id": self.source_call_id,
            "source_turn_index": self.source_turn_index,
            "derivation": self.derivation,
        }


def _resolve(path: str | Path, root: Path | None) -> Path:
    """Resolve a repo-relative path without modifying it."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (root or REPO_ROOT) / candidate


def _load_fixture_rows(path: str | Path, root: Path | None) -> list[dict[str, Any]]:
    """Read the labelled fixture manifest as plain JSON Lines."""
    rows: list[dict[str, Any]] = []
    for line in _resolve(path, root).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        decoded = json.loads(line)
        if not isinstance(decoded, dict):
            raise ValueError("fixture rows must be JSON objects")
        rows.append(decoded)
    return rows


def _read_event(event_path: str, root: Path | None) -> dict[str, Any]:
    """Read one read-only fixture event file."""
    decoded = json.loads(_resolve(event_path, root).read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("fixture event must be a JSON object")
    return decoded


def _preserve_case(matched: str, canonical: str) -> str:
    """Apply canonical spelling while keeping the matched token's casing style."""
    if matched.isupper():
        return canonical.upper()
    if matched[:1].isupper():
        return canonical[:1].upper() + canonical[1:]
    return canonical.lower()


def _replace_with_canonical(match: re.Match[str], canonical: str) -> str:
    """Return one regex match rewritten to its canonical form."""
    return _preserve_case(match.group(0), canonical)


def _booking_room_types() -> frozenset[str]:
    """Read case-folded room types from the committed booking schema (read-only)."""
    schema = load_schema("booking")
    properties = schema.get("properties")
    room_type = properties.get("room_type") if isinstance(properties, dict) else None
    values = room_type.get("enum") if isinstance(room_type, dict) else None
    if not isinstance(values, list) or not values:
        raise ValidationError("booking schema has no room_type enum")
    if not all(isinstance(value, str) and value for value in values):
        raise ValidationError("booking schema room_type enum must contain non-empty strings")
    return frozenset(value.casefold() for value in values)


@dataclass(frozen=True)
class AliasTable:
    """Immutable alias entries plus deterministic text/slot canonicalisation."""

    entries: tuple[AliasEntry, ...]

    def for_kind(self, kind: AliasKind) -> tuple[AliasEntry, ...]:
        """Return entries of one kind in replacement-safe (longest-first) order."""
        return tuple(
            sorted(
                (entry for entry in self.entries if entry.kind == kind),
                key=lambda entry: (-len(entry.surface), entry.surface.casefold()),
            )
        )

    def apply_text(
        self,
        text: str,
        *,
        kinds: Sequence[AliasKind] = ("name", "room_type"),
    ) -> str:
        """Replace known alias surfaces in ``text`` with their canonical forms.

        Word boundaries prevent partial-token edits, and longer surfaces are
        tried first. Casing is preserved from the observed text.
        """
        result = text
        for kind in kinds:
            for entry in self.for_kind(kind):
                pattern = re.compile(rf"(?<!\w){re.escape(entry.surface)}(?!\w)", re.IGNORECASE)
                result = pattern.sub(
                    partial(_replace_with_canonical, canonical=entry.canonical), result
                )
        return result

    def canonicalise_slots(self, slots: Mapping[str, Any]) -> dict[str, Any]:
        """Canonicalise only ``guest_name`` and ``room_type`` string slots."""
        result = dict(slots)
        name = result.get("guest_name")
        if isinstance(name, str):
            result["guest_name"] = self.apply_text(name, kinds=("name",))
        room = result.get("room_type")
        if isinstance(room, str):
            result["room_type"] = self.apply_text(room, kinds=("room_type",)).lower()
        return result

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable table with provenance for every entry."""
        return {"entries": [entry.to_dict() for entry in self.entries]}


def derive_alias_table(
    path: str | Path = DEFAULT_FIXTURES,
    *,
    root: Path | None = None,
    name_overrides: Mapping[str, str] | None = None,
    room_overrides: Mapping[str, str] | None = None,
) -> AliasTable:
    """Derive aliases from read-only fixtures; never from the sealed golden set.

    Args:
        path: Labelled synthetic fixture manifest path.
        root: Base directory for repo-relative paths; defaults to repo root.
        name_overrides: Extra ``surface -> canonical`` name aliases.
        room_overrides: Extra ``surface -> canonical`` room-type aliases.

    Returns:
        A deterministic table with source call/turn provenance per entry.

    Raises:
        ValidationError: If any room alias, including an override, maps to a
            canonical value outside the committed booking schema's room_type enum.

    """
    rows = _load_fixture_rows(path, root)
    collected: dict[tuple[AliasKind, str], AliasEntry] = {}

    for row in rows:
        if row.get("case_id") != "misheard_surname":
            continue
        event = _read_event(str(row["event_path"]), root)
        observed = str(event.get("stt", {}).get("text", ""))
        expected_slots = row.get("expected_slots") or {}
        expected_name = str(expected_slots.get("guest_name", ""))
        observed_tokens = _TOKEN_RE.findall(observed)
        expected_tokens = _TOKEN_RE.findall(expected_name)
        for observed_token, expected_token in zip(observed_tokens, expected_tokens, strict=False):
            if observed_token.casefold() == expected_token.casefold():
                continue
            if len(observed_token) < 3 or len(expected_token) < 3:
                continue
            key: tuple[AliasKind, str] = ("name", observed_token.casefold())
            collected.setdefault(
                key,
                AliasEntry(
                    surface=observed_token.casefold(),
                    canonical=expected_token,
                    kind="name",
                    source_case_id=str(row.get("case_id")),
                    source_call_id=str(row["call_id"]),
                    source_turn_index=int(row["turn_index"]),
                    derivation=(
                        "Token-wise diff between the read-only misheard_surname STT "
                        "hypothesis and its labelled guest_name slot."
                    ),
                ),
            )

    room_types = sorted(
        {
            str((row.get("expected_slots") or {}).get("room_type", "")).casefold()
            for row in rows
            if (row.get("expected_slots") or {}).get("room_type")
        }
    )
    room_pattern_re = (
        re.compile(
            rf"\b({'|'.join(re.escape(room) for room in room_types)})\b(?:\s+room)?\b",
            re.IGNORECASE,
        )
        if room_types
        else None
    )
    if room_pattern_re is not None:
        for row in sorted(rows, key=lambda item: (str(item["call_id"]), int(item["turn_index"]))):
            transcript = str(row.get("expected_transcript", ""))
            for match in room_pattern_re.finditer(transcript):
                surface = " ".join(match.group(0).casefold().split())
                canonical = " ".join(match.group(1).casefold().split())
                if surface == canonical:
                    continue
                key = ("room_type", surface)
                collected.setdefault(
                    key,
                    AliasEntry(
                        surface=surface,
                        canonical=canonical,
                        kind="room_type",
                        source_case_id=row.get("case_id"),
                        source_call_id=str(row["call_id"]),
                        source_turn_index=int(row["turn_index"]),
                        derivation=(
                            "Room-type surface phrase in a read-only expected transcript "
                            "paired with that turn's labelled room_type slot."
                        ),
                    ),
                )

    override_groups: tuple[tuple[Mapping[str, str] | None, AliasKind, str], ...] = (
        (name_overrides, "name", "VOICELOOP_VOCAB_NAME_ALIAS_OVERRIDES"),
        (room_overrides, "room_type", "VOICELOOP_VOCAB_ROOM_ALIAS_OVERRIDES"),
    )
    for overrides, kind, label in override_groups:
        for surface, canonical in (overrides or {}).items():
            override_key: tuple[AliasKind, str] = (kind, surface.casefold())
            collected[override_key] = AliasEntry(
                surface=surface.casefold(),
                canonical=canonical,
                kind=kind,
                source_case_id=None,
                source_call_id="settings",
                source_turn_index=-1,
                derivation=f"Manual alias from {label}.",
            )

    valid_room_types = _booking_room_types()
    for entry in collected.values():
        if entry.kind == "room_type" and entry.canonical.casefold() not in valid_room_types:
            raise ValidationError(
                f"room alias {entry.surface!r} maps to invalid room_type "
                f"{entry.canonical!r}; expected one of {sorted(valid_room_types)}"
            )

    entries = tuple(
        sorted(collected.values(), key=lambda entry: (entry.kind, entry.surface, entry.canonical))
    )
    return AliasTable(entries=entries)
