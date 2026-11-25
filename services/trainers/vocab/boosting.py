"""Phrase-boosting / LM-fusion config generator for an injected STT seam.

The emitted JSON is a **configuration proposal** for the current STT runtime
(``faster-whisper-ctranslate2`` by roadmap default). This package never starts
that runtime, decodes audio, or claims that boosting was applied in production.
:class:`FakeSttRuntime` is an explicitly synthetic stand-in used only to show
how a runtime could consume the config and to exercise the harness in-memory.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .aliases import derive_alias_table
from .settings import VocabSettings

CONFIG_SCHEMA_VERSION = "voiceloop.vocab-boost.v1"
SYNTHETIC_WARNING = (
    "SYNTHETIC/FIXTURE STAND-IN: config derived from synthetic fixtures; "
    "no real STT runtime or GPU applied phrase boosting or LM fusion."
)
_OBSERVED_GUEST_RE = re.compile(r"\b(\w+\s+guests)\b", re.IGNORECASE)
_WORD_RE = re.compile(r"\w+")


@dataclass(frozen=True)
class Hotword:
    """One boost phrase with its fixture-derived provenance."""

    phrase: str
    weight: float
    source_case_id: str | None
    source_call_id: str
    source_turn_index: int
    derivation: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable hotword record."""
        return {
            "phrase": self.phrase,
            "weight": self.weight,
            "source_case_id": self.source_case_id,
            "source_call_id": self.source_call_id,
            "source_turn_index": self.source_turn_index,
            "derivation": self.derivation,
        }


@dataclass(frozen=True)
class Substitution:
    """A read-only fixture mishearing used by the synthetic runtime stand-in."""

    observed: str
    canonical: str
    source_case_id: str
    source_call_id: str
    source_turn_index: int
    derivation: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable substitution record."""
        return {
            "observed": self.observed,
            "canonical": self.canonical,
            "source_case_id": self.source_case_id,
            "source_call_id": self.source_call_id,
            "source_turn_index": self.source_turn_index,
            "derivation": self.derivation,
        }


@dataclass(frozen=True)
class BoostConfig:
    """JSON config payload for a runtime seam, with explicit synthetic labels."""

    runtime: str
    language: str
    hotwords: tuple[Hotword, ...]
    substitutions: tuple[Substitution, ...]
    max_hotwords: int
    boost: float
    initial_prompt: str
    condition_on_previous_text: bool
    low_confidence_threshold: float
    provenance: tuple[str, ...]
    schema_version: str = CONFIG_SCHEMA_VERSION
    synthetic: bool = True
    warning: str = SYNTHETIC_WARNING

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable config with all provenance fields."""
        return {
            "schema_version": self.schema_version,
            "runtime": self.runtime,
            "language": self.language,
            "synthetic": self.synthetic,
            "warning": self.warning,
            "hotwords": [hotword.to_dict() for hotword in self.hotwords],
            "substitutions": [item.to_dict() for item in self.substitutions],
            "max_hotwords": self.max_hotwords,
            "boost": self.boost,
            "initial_prompt": self.initial_prompt,
            "condition_on_previous_text": self.condition_on_previous_text,
            "low_confidence_threshold": self.low_confidence_threshold,
            "provenance": list(self.provenance),
        }

    def to_json(self) -> str:
        """Return canonical, deterministic JSON text for the runtime seam."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=2, ensure_ascii=False) + "\n"


@runtime_checkable
class SttRuntime(Protocol):
    """Injectable seam: consume a config and return a hypothesised transcript."""

    def transcribe(
        self,
        transcript: str,
        words: Sequence[Mapping[str, Any]],
        config: BoostConfig,
    ) -> str:
        """Return the transcript after any runtime-specific boosting."""
        ...


def _read_rows(path: str) -> list[dict[str, Any]]:
    """Read fixture rows from a read-only JSONL manifest."""
    from .aliases import REPO_ROOT

    candidate = Path(path)
    source = candidate if candidate.is_absolute() else REPO_ROOT / candidate
    rows: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            decoded = json.loads(line)
            if not isinstance(decoded, dict):
                raise ValueError("fixture rows must be JSON objects")
            rows.append(decoded)
    return rows


def _read_event(path: str) -> dict[str, Any]:
    """Read one read-only fixture event."""
    from .aliases import REPO_ROOT

    candidate = Path(path)
    source = candidate if candidate.is_absolute() else REPO_ROOT / candidate
    decoded = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("fixture event must be a JSON object")
    return decoded


def _clean_phrase(value: str) -> str:
    """Strip trailing punctuation from a fixture surface phrase, keeping case."""
    return value.strip().strip(".,!?;:\"' ")


def derive_substitutions(fixtures_path: str) -> tuple[Substitution, ...]:
    """Derive narrow mishearing substitutions from named failure fixtures.

    Only the read-only ``misheard_surname`` and ``misheard_guests`` cases are
    used. Each substitution cites its source call/turn; no phrase is invented.
    """
    substitutions: list[Substitution] = []
    for row in _read_rows(fixtures_path):
        case_id = row.get("case_id")
        if case_id not in {"misheard_surname", "misheard_guests"}:
            continue
        event = _read_event(str(row["event_path"]))
        observed_text = str(event.get("stt", {}).get("text", ""))
        expected_text = str(row.get("expected_transcript", ""))
        expected_slots = row.get("expected_slots") or {}
        if case_id == "misheard_surname":
            observed = _clean_phrase(observed_text)
            canonical_value = expected_slots.get("guest_name")
            canonical = _clean_phrase(str(canonical_value)) if canonical_value else ""
        else:
            observed_match = _OBSERVED_GUEST_RE.search(observed_text)
            expected_match = _OBSERVED_GUEST_RE.search(expected_text)
            observed = _clean_phrase(observed_match.group(1)) if observed_match else ""
            canonical = _clean_phrase(expected_match.group(1)) if expected_match else ""
        if not observed or not canonical or observed == canonical:
            continue
        substitutions.append(
            Substitution(
                observed=observed,
                canonical=canonical,
                source_case_id=str(case_id),
                source_call_id=str(row["call_id"]),
                source_turn_index=int(row["turn_index"]),
                derivation=(
                    "Read-only named fixture failure: observed STT hypothesis "
                    "paired with the expected transcript/slot from its label."
                ),
            )
        )
    return tuple(
        sorted(substitutions, key=lambda item: (item.source_call_id, item.source_turn_index))
    )


def generate_boosting_config(settings: VocabSettings | None = None) -> BoostConfig:
    """Generate a synthetic runtime config from read-only fixtures.

    Failed-case canonical phrases are listed first, followed by room-type
    phrases observed in fixture transcripts. The result is capped by
    ``settings.max_hotwords`` and always labelled synthetic.
    """
    settings = settings or VocabSettings()
    substitutions = derive_substitutions(settings.fixtures_path)
    aliases = derive_alias_table(settings.fixtures_path)
    candidates: list[Hotword] = []
    seen: set[str] = set()

    def add(
        phrase: str, source_case_id: str | None, source: str, turn: int, derivation: str
    ) -> None:
        key = phrase.casefold()
        if not key or key in seen:
            return
        seen.add(key)
        candidates.append(
            Hotword(
                phrase=phrase,
                weight=settings.hotword_boost,
                source_case_id=source_case_id,
                source_call_id=source,
                source_turn_index=turn,
                derivation=derivation,
            )
        )

    for item in substitutions:
        add(
            item.canonical,
            item.source_case_id,
            item.source_call_id,
            item.source_turn_index,
            "Canonical phrase from a read-only named fixture failure.",
        )
    for entry in aliases.for_kind("room_type"):
        add(
            entry.surface,
            entry.source_case_id,
            entry.source_call_id,
            entry.source_turn_index,
            "Room-type phrase derived from a read-only fixture transcript.",
        )
    room_values = sorted(
        {
            str((row.get("expected_slots") or {}).get("room_type", "")).casefold()
            for row in _read_rows(settings.fixtures_path)
            if (row.get("expected_slots") or {}).get("room_type")
        }
    )
    for room in room_values:
        add(room, None, "fixtures", -1, "Canonical room type from read-only fixture slots.")
    return BoostConfig(
        runtime=settings.runtime,
        language=settings.language,
        hotwords=tuple(candidates[: settings.max_hotwords]),
        substitutions=substitutions,
        max_hotwords=settings.max_hotwords,
        boost=settings.hotword_boost,
        initial_prompt=settings.initial_prompt,
        condition_on_previous_text=settings.condition_on_previous_text,
        low_confidence_threshold=settings.low_confidence_threshold,
        provenance=(f"fixtures:{settings.fixtures_path}", f"runtime_label:{settings.runtime}"),
    )


def write_boosting_config(config: BoostConfig, path: str | Path) -> Path:
    """Write the generated JSON config to ``path`` and return it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(config.to_json(), encoding="utf-8")
    return destination


def _preserve_case(matched: str, canonical: str) -> str:
    """Keep observed casing style while applying a canonical phrase."""
    if matched.isupper():
        return canonical.upper()
    if matched[:1].isupper():
        return canonical[:1].upper() + canonical[1:]
    return canonical.lower()


def _replace_with_canonical(match: re.Match[str], canonical: str) -> str:
    """Return one regex match rewritten to its canonical phrase."""
    return _preserve_case(match.group(0), canonical)


class FakeSttRuntime:
    """SYNTHETIC stand-in that applies only low-confidence fixture corrections.

    This is not an STT decoder. It consumes the generated config and replaces a
    fixture-derived misheard phrase only when at least one word of that phrase
    has confidence below the configured threshold. Production runtimes must be
    injected separately; no real boosting is claimed here.
    """

    def transcribe(
        self,
        transcript: str,
        words: Sequence[Mapping[str, Any]],
        config: BoostConfig,
    ) -> str:
        """Return a transcript with synthetic, config-driven phrase corrections."""
        result = transcript
        confidences = {
            _clean_phrase(str(word.get("w", ""))).casefold(): float(word.get("conf", 1.0))
            for word in words
            if isinstance(word, Mapping)
        }
        for item in config.substitutions:
            observed_tokens = _WORD_RE.findall(item.observed)
            low_confidence = any(
                confidences.get(token.casefold(), 1.0) < config.low_confidence_threshold
                for token in observed_tokens
            )
            if not low_confidence:
                continue
            pattern = re.compile(rf"(?<!\w){re.escape(item.observed)}(?!\w)", re.IGNORECASE)
            result = pattern.sub(partial(_replace_with_canonical, canonical=item.canonical), result)
        return result


def apply_boosting(
    transcript: str,
    words: Sequence[Mapping[str, Any]],
    config: BoostConfig,
    runtime: SttRuntime | None = None,
) -> str:
    """Run one injectable runtime seam over a transcript and its word timings."""
    return (runtime or FakeSttRuntime()).transcribe(transcript, words, config)
