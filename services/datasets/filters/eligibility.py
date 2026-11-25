"""Dataset eligibility filters for consent and PII (P1-09.1/P1-09.2)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from services.ingest.consent.consent import is_eligible

from .pii import PiiFinding, detect_pii, has_card_number

TurnEvent = Mapping[str, Any]


class DropReason(StrEnum):
    """Stable, auditable reasons an event was excluded from a dataset."""

    CONSENT_MISSING = "consent_missing"
    RECORDING_CONSENT_DENIED = "recording_consent_denied"
    TRAINING_CONSENT_DENIED = "training_consent_denied"
    CARD_NUMBER_PII = "card_number_pii"


@dataclass(frozen=True, slots=True)
class DroppedEvent:
    """PII-free audit record tying a dropped row to its source call and reasons."""

    event_id: str
    call_id: str
    turn_index: int
    audio_uri: str
    reasons: tuple[DropReason, ...]
    pii_findings: tuple[PiiFinding, ...] = ()


def consent_reasons(event: TurnEvent) -> tuple[DropReason, ...]:
    """Return why an event fails the consent gate, or an empty tuple.

    Args:
        event: A schema-valid turn event, read-only.

    Returns:
        ``()`` when :func:`is_eligible` would return ``True``; otherwise one or
        more stable reasons. Missing flags fail closed.

    """
    if is_eligible(event):
        return ()
    consent = event.get("consent")
    if not isinstance(consent, Mapping):
        return (DropReason.CONSENT_MISSING,)
    reasons: list[DropReason] = []
    if consent.get("recording") is not True:
        reasons.append(DropReason.RECORDING_CONSENT_DENIED)
    if consent.get("training") is not True:
        reasons.append(DropReason.TRAINING_CONSENT_DENIED)
    return tuple(reasons)


def filter_by_consent(
    events: Iterable[TurnEvent],
) -> tuple[list[TurnEvent], list[DroppedEvent]]:
    """Split events by consent eligibility only.

    Args:
        events: Schema-valid turn events, never mutated.

    Returns:
        ``(kept, dropped_with_reasons)``; dropped records carry source ids.

    """
    kept: list[TurnEvent] = []
    dropped: list[DroppedEvent] = []
    for event in events:
        reasons = consent_reasons(event)
        if reasons:
            dropped.append(_dropped_event(event, reasons))
        else:
            kept.append(event)
    return kept, dropped


def filter_eligible_events(
    events: Iterable[TurnEvent],
) -> tuple[list[TurnEvent], list[DroppedEvent]]:
    """Split events by consent and hard-drop PII rules.

    Consent is checked first; card-number PII is a hard drop based on a
    dedicated card scan, independent of other detectors' overlap resolution,
    and is never redacted into training data. Other detected PII (email, phone,
    person-name patterns) is reported as a finding for the later redaction/pseudonymisation
    stage (P3-10) and does not by itself exclude a row. Events and their nested
    structures are treated as read-only.

    Args:
        events: Schema-valid turn events, never mutated.

    Returns:
        ``(kept, dropped_with_reasons)``. Every dropped record lists all
        triggered reasons and PII offsets without matched values, so exclusions
        are auditable and no raw PII is copied into logs or manifests.

    """
    kept: list[TurnEvent] = []
    dropped: list[DroppedEvent] = []
    for event in events:
        reasons = list(consent_reasons(event))
        text = _transcript_text(event)
        findings = detect_pii(text)
        if has_card_number(text):
            reasons.append(DropReason.CARD_NUMBER_PII)
        if reasons:
            dropped.append(_dropped_event(event, tuple(reasons), findings))
        else:
            kept.append(event)
    return kept, dropped


def _transcript_text(event: TurnEvent) -> str:
    """Return the STT transcript text, or an empty string when absent."""
    stt = event.get("stt")
    if not isinstance(stt, Mapping):
        return ""
    text = stt.get("text")
    return text if isinstance(text, str) else ""


def _dropped_event(
    event: TurnEvent,
    reasons: tuple[DropReason, ...],
    findings: tuple[PiiFinding, ...] = (),
) -> DroppedEvent:
    """Build an auditable drop record using only identifiers and offsets."""
    audio = event.get("audio")
    raw_index = event.get("turn_index")
    return DroppedEvent(
        event_id=str(event.get("event_id", "")),
        call_id=str(event.get("call_id", "")),
        turn_index=(
            raw_index if isinstance(raw_index, int) and not isinstance(raw_index, bool) else -1
        ),
        audio_uri=(str(audio.get("uri", "")) if isinstance(audio, Mapping) else ""),
        reasons=reasons,
        pii_findings=findings,
    )
