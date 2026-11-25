"""Name pseudonymisation and non-name PII masking for training transcripts.

Text screening delegates to the committed, read-only P1-09 ``detect_pii`` API;
card detection is never reimplemented here. Card findings are a hard drop:
they raise a structured error and produce no redacted transcript. Names become
stable per-call salted pseudonyms; email/phone findings become category masks;
all other text is preserved byte-for-byte. Matched values are never stored in
results, audit events or logs.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from services.datasets.filters import PiiCategory, PiiFinding
from voiceloop_common import StorageError

from .settings import (
    COMPLIANCE_STATUS,
    POLICY_STATUS,
    SYNTHETIC_LABEL,
    AuditEvent,
    AuditSink,
    Clock,
    InMemoryAuditSink,
    RedactionSettings,
    SystemClock,
)
from .spans import (
    CardNumberRejected,
    P109TranscriptDetector,
    RedactionError,
    SpanCategory,
    TranscriptPiiDetector,
    UnverifiedRedactionError,
    resolve_char_spans,
)

_MASK_BY_CATEGORY: dict[SpanCategory, str] = {
    SpanCategory.EMAIL: "[EMAIL_REDACTED]",
    SpanCategory.PHONE: "[PHONE_REDACTED]",
    SpanCategory.CARD_NUMBER: "[CARD_REDACTED]",
    SpanCategory.PII: "[PII_REDACTED]",
    SpanCategory.MIXED: "[PII_REDACTED]",
}


@dataclass(frozen=True, slots=True)
class TextReplacement:
    """One PII replacement: offsets, category and safe token; no matched text."""

    start: int
    end: int
    category: SpanCategory
    token: str

    def to_dict(self) -> dict[str, str | int]:
        """Return offsets, category and token only."""
        return {
            "start": self.start,
            "end": self.end,
            "category": str(self.category),
            "token": self.token,
        }


@dataclass(frozen=True, slots=True)
class PseudonymisedText:
    """Transcript safe for training, with PII-free provenance metadata."""

    call_id: str
    text: str
    findings: tuple[PiiFinding, ...]
    replacements: tuple[TextReplacement, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe report; findings carry offsets only."""
        return {
            "call_id": self.call_id,
            "text": self.text,
            "findings": [finding.as_dict() for finding in self.findings],
            "replacements": [item.to_dict() for item in self.replacements],
            "policy_status": POLICY_STATUS,
            "compliance_status": COMPLIANCE_STATUS,
            "synthetic": True,
            "label": SYNTHETIC_LABEL,
        }


@dataclass(frozen=True, slots=True)
class TranscriptRecord:
    """In-memory transcript input; never carries audio or real guest data."""

    key: str
    call_id: str
    text: str
    known_names: tuple[str, ...] = ()


class TranscriptSource(Protocol):
    """Injectable source of transcript text and known reviewed names."""

    def load(self, key: str) -> TranscriptRecord:
        """Load one transcript record by synthetic key."""
        ...


@dataclass(slots=True)
class InMemoryTranscriptSource:
    """Fake transcript source for tests and synthetic CLI rehearsals."""

    records: dict[str, TranscriptRecord] = field(default_factory=dict)

    def add(self, record: TranscriptRecord) -> None:
        """Store one synthetic transcript record under its key."""
        self.records[record.key] = record

    def load(self, key: str) -> TranscriptRecord:
        """Load a record, failing closed when the key is unknown."""
        if key not in self.records:
            raise StorageError(f"unknown synthetic transcript key: {key}")
        return self.records[key]


def make_pseudonym(call_id: str, name: str, settings: RedactionSettings | None = None) -> str:
    """Derive a stable per-call salted pseudonym from a detected name.

    The digest uses the per-call namespace salt, the call ID and the casefolded
    matched name, so the same person maps to the same token across turns of one
    call, different calls never share a token, and replay is deterministic. The
    original name is not retained in the token or any report.

    Args:
        call_id: Synthetic call identifier used as a per-call salt component.
        name: Matched name text, used only in the transient hash input.
        settings: Typed redaction settings; defaults to env-backed settings.

    Returns:
        A bracketed pseudonym token such as ``[PERSON_1a2b3c4d5e6f]``.

    Raises:
        RedactionError: ``call_id`` or ``name`` is empty (fail closed).

    """
    if not call_id or not name:
        raise RedactionError(
            "invalid_pseudonym_input", "call_id and matched name must be non-empty"
        )
    config = settings or RedactionSettings()
    digest_input = f"{config.pseudonym_salt}\x00{call_id}\x00{name.casefold()}".encode()
    digest = hashlib.sha256(digest_input).hexdigest()[:12]
    return f"[{config.pseudonym_prefix}_{digest}]"


def _category_counts(findings: Sequence[PiiFinding]) -> dict[str, int]:
    """Count findings by category name without retaining matched values."""
    counts: dict[str, int] = {}
    for finding in findings:
        key = f"finding_{finding.category.value}"
        counts[key] = counts.get(key, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def _name_still_present(text: str, known_names: Sequence[str]) -> int:
    """Count known names still present as whole words in transformed text."""
    remaining = 0
    for raw_name in known_names:
        name = raw_name.strip()
        if not name:
            continue
        pattern = re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)", re.IGNORECASE)
        remaining += len(pattern.findall(text))
    return remaining


def pseudonymise_text(
    text: str,
    *,
    call_id: str,
    settings: RedactionSettings | None = None,
    detector: TranscriptPiiDetector | None = None,
    known_names: Sequence[str] = (),
    audit: AuditSink | None = None,
    clock: Clock | None = None,
) -> PseudonymisedText:
    """Pseudonymise names and mask other PII in a training transcript.

    Card findings are hard-dropped via a structured error, never pseudonymised.
    Replacements are applied right-to-left with offsets from the read-only
    detector, then verified: the output must have no residual PII finding and
    no caller-supplied known name left as a whole word.

    Args:
        text: Original transcript text, treated as read-only.
        call_id: Synthetic call ID used for stable per-call pseudonyms.
        settings: Typed ``VOICELOOP_REDACTION_`` settings.
        detector: Injected transcript detector; defaults to read-only P1-09.
        known_names: Reviewed name values covering names P1-09 does not cue.
        audit: Injected PII-free audit sink; defaults to an in-memory fake.
        clock: Injected UTC clock for audit timestamps.

    Returns:
        Safe transcript plus offsets/categories and pseudonym tokens.

    Raises:
        CardNumberRejected: Any card finding was detected.
        UnverifiedRedactionError: PII or a known name survived the transformation.

    """
    config = settings or RedactionSettings()
    screen = detector or P109TranscriptDetector()
    audit_sink = audit or InMemoryAuditSink()
    audit_clock = clock or SystemClock()

    findings = screen.detect(text)
    cards = tuple(item for item in findings if item.category is PiiCategory.CARD_NUMBER)
    if cards:
        raise CardNumberRejected(
            "card number findings are hard-dropped, never pseudonymised",
            {
                "call_id": call_id,
                "card_findings": len(cards),
                "first_start": cards[0].start,
                "first_end": cards[0].end,
            },
        )

    spans = resolve_char_spans(findings, text, known_names)
    transformed = text
    replacements: list[TextReplacement] = []
    for span in reversed(spans):
        matched = text[span.start : span.end]
        if span.category is SpanCategory.PERSON_NAME:
            token = make_pseudonym(call_id, matched, config)
        else:
            token = _MASK_BY_CATEGORY.get(span.category, "[PII_REDACTED]")
        transformed = transformed[: span.start] + token + transformed[span.end :]
        replacements.append(TextReplacement(span.start, span.end, span.category, token))
    replacements.reverse()

    remaining = screen.detect(transformed)
    if remaining:
        raise UnverifiedRedactionError(
            "unverified_text_pii",
            "transcript still contains a detected PII span after redaction",
            {"remaining_findings": len(remaining), "first_start": remaining[0].start},
        )
    leaks = _name_still_present(transformed, known_names)
    if leaks:
        raise UnverifiedRedactionError(
            "name_leak",
            "a known name survived transcript pseudonymisation",
            {"remaining_names": leaks},
        )

    details: dict[str, str | int | float | bool | None] = {
        "call_id": call_id,
        "replacements": len(replacements),
        "policy_status": POLICY_STATUS,
        "label": SYNTHETIC_LABEL,
    }
    details.update(_category_counts(findings))
    audit_sink.emit(
        AuditEvent(
            event_type="text_pseudonymised",
            occurred_at=audit_clock.now(),
            details=details,
        )
    )
    return PseudonymisedText(
        call_id=call_id,
        text=transformed,
        findings=findings,
        replacements=tuple(replacements),
    )


def pseudonymise_transcript(
    source: TranscriptSource,
    key: str,
    *,
    settings: RedactionSettings | None = None,
    detector: TranscriptPiiDetector | None = None,
    audit: AuditSink | None = None,
    clock: Clock | None = None,
) -> PseudonymisedText:
    """Load a transcript from an injected source and pseudonymise it.

    Args:
        source: Injected transcript source seam.
        key: Synthetic key for one transcript record.
        settings: Typed redaction settings.
        detector: Injected transcript detector.
        audit: Injected PII-free audit sink.
        clock: Injected UTC clock.

    Returns:
        The pseudonymised transcript for that record.

    """
    record = source.load(key)
    return pseudonymise_text(
        record.text,
        call_id=record.call_id,
        settings=settings,
        detector=detector,
        known_names=record.known_names,
        audit=audit,
        clock=clock,
    )


__all__ = [
    "InMemoryTranscriptSource",
    "PseudonymisedText",
    "TextReplacement",
    "TranscriptRecord",
    "TranscriptSource",
    "make_pseudonym",
    "pseudonymise_text",
    "pseudonymise_transcript",
]
