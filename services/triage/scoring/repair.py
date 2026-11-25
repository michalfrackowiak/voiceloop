"""Repair-behaviour detection from turn events and dialogue context."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .settings import TriageScoringSettings
from .signals import Observation, _as_mapping

GUEST_CORRECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bno,?\s+i\s+(?:said|mean|meant)\b",
        r"\bi\s+mean\b",
        r"\bactually\b",
        r"\bthat'?s\s+(?:wrong|not\s+what\s+i\s+said)\b",
        r"\bcorrection\b",
    )
)
AGENT_REASK_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:could|can|would)\s+you\s+(?:please\s+)?(?:repeat|say\s+that\s+again)\b",
        r"\bwhat\s+was\s+that\b",
        r"\bcome\s+again\b",
        r"\bsay\s+(?:that|it)\s+again\b",
        r"\b(?:sorry|pardon)[,?]?\s+what\b",
    )
)


@dataclass(frozen=True)
class RepairContext:
    """Dialogue evidence available for repair detection on one turn."""

    prior_guest_texts: tuple[str, ...] = ()
    prior_agent_texts: tuple[str, ...] = ()
    status: str | None = None

    @property
    def transferred(self) -> bool:
        """Return whether the outcome records a transfer to a human."""
        return self.status == "transferred"


def load_prior_turns(
    event: Mapping[str, Any], *, root: Path | None = None
) -> tuple[dict[str, Any], ...]:
    """Read the optional ``dialogue_context.prior_turns_ref`` JSONL file.

    Args:
        event: A schema-valid turn event.
        root: Base directory for repo-relative URIs; ``None`` reads the path as
            given (normally an absolute path supplied by the caller).

    Returns:
        Decoded prior-turn records, or an empty tuple when no reference exists.

    Raises:
        ValueError: If the referenced file is unreadable or contains invalid JSON.

    """
    dialogue = _as_mapping(event.get("dialogue_context"))
    reference = dialogue.get("prior_turns_ref") if dialogue is not None else None
    if not isinstance(reference, str) or not reference:
        return ()
    path = Path(reference)
    if not path.is_absolute() and root is not None:
        path = root / path
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read prior turns: {path}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid prior-turn JSON at {path}:{line_number}") from exc
        if not isinstance(decoded, dict):
            raise ValueError(f"prior-turn record must be an object at {path}:{line_number}")
        records.append(cast(dict[str, Any], decoded))
    return tuple(records)


def _text_field(record: Mapping[str, Any], *keys: str) -> str | None:
    """Return the first non-empty string among ``keys`` in a prior-turn record."""
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def build_repair_context(
    event: Mapping[str, Any], prior_turns: Sequence[Mapping[str, Any]] | None = None
) -> RepairContext:
    """Build repair evidence from an event and optional prior-turn records.

    Args:
        event: A schema-valid turn event; its ``outcome.status`` supplies transfer.
        prior_turns: Decoded context records. Guest hypotheses and agent
            utterances are classified by ``speaker`` when present, otherwise by
            the key that carries text.

    Returns:
        Immutable repair context for :func:`detect_repair`.

    """
    guest_texts: list[str] = []
    agent_texts: list[str] = []
    for record in prior_turns or ():
        speaker = record.get("speaker")
        speaker_name = speaker.strip().lower() if isinstance(speaker, str) else ""
        guest_text = _text_field(record, "hypothesis", "text", "transcript")
        agent_text = _text_field(record, "agent_text", "agent_utterance")
        if speaker_name == "agent":
            if agent_text:
                agent_texts.append(agent_text)
            elif guest_text:
                agent_texts.append(guest_text)
        elif speaker_name == "guest":
            if guest_text:
                guest_texts.append(guest_text)
            elif agent_text:
                agent_texts.append(agent_text)
        else:
            if guest_text:
                guest_texts.append(guest_text)
            if agent_text:
                agent_texts.append(agent_text)

    outcome = _as_mapping(event.get("outcome"))
    status = outcome.get("status") if outcome is not None else None
    return RepairContext(
        prior_guest_texts=tuple(guest_texts),
        prior_agent_texts=tuple(agent_texts),
        status=status if isinstance(status, str) else None,
    )


def detect_repair(
    event: Mapping[str, Any],
    settings: TriageScoringSettings,
    context: RepairContext | None = None,
) -> list[Observation]:
    """Detect guest corrections, agent re-asks and transfers.

    Args:
        event: A schema-valid turn event.
        settings: Weights for this run.
        context: Optional dialogue evidence; built from ``event`` when omitted.

    Returns:
        One observation per distinct repair behaviour.

    """
    repair_context = context if context is not None else build_repair_context(event)
    observations: list[Observation] = []

    stt = _as_mapping(event.get("stt"))
    current_text = stt.get("text", "") if stt is not None else ""
    guest_texts = (
        (current_text,) + repair_context.prior_guest_texts
        if isinstance(current_text, str)
        else repair_context.prior_guest_texts
    )
    for patterns, texts, name, label in (
        (
            GUEST_CORRECTION_PATTERNS,
            guest_texts,
            "guest_correction",
            "guest correction phrase",
        ),
        (
            AGENT_REASK_PATTERNS,
            repair_context.prior_agent_texts,
            "agent_reask",
            "agent re-ask phrase",
        ),
    ):
        for text in texts:
            match = next((found for pattern in patterns if (found := pattern.search(text))), None)
            if match is not None:
                observations.append(
                    Observation(
                        name=name,
                        weight=settings.weight_repair_behaviour,
                        severity=1.0,
                        reason=f"{label} {match.group(0)!r}",
                    )
                )
                break

    if repair_context.transferred:
        observations.append(
            Observation(
                name="transfer",
                weight=settings.weight_repair_behaviour,
                severity=1.0,
                reason="outcome.status='transferred' indicates a human handoff",
            )
        )
    return observations
