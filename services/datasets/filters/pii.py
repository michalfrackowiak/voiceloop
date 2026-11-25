"""Deterministic transcript PII detection (P1-09.2).

Card-like runs are detected by digit count alone (at least 13 digits, with any
repeated unicode whitespace, hyphen/dash, dot, slash or underscore separators),
not by Luhn, issuer tables, group lengths or neighbouring letters. Longer runs
are also hard-dropped: this deliberately prefers false-positive drops over
misses. Card screening uses a dedicated position-by-position scan so a greedy
match beginning earlier in the text cannot prevent a card-like run from
triggering the hard drop; the hard-drop decision never depends on ``finditer``
non-overlap or overlap ordering with other detectors.
Detected spans are character offsets into the supplied text, with no raw values
attached so findings can be logged safely.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

_CARD_RE = re.compile(r"(?<!\d)(\d+(?:[\s\u2010-\u2015./_-]+\d+)*)(?!\d)")
_EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})(?![A-Za-z0-9_%+-])"
)
_PHONE_INTL_RE = re.compile(r"(?<!\w)(\+\d{1,3}(?:[ .-]?\d{2,4}){2,4})(?!\w)")
_PHONE_NATIONAL_RE = re.compile(r"(?<!\w)((?:\(\d{3}\)\s?|\d{3}[ .-])\d{3}[ .-]\d{4})(?!\w)")
_NAME_RE = re.compile(
    r"\b(?:(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+([A-Z][a-z]+)"
    r"|name\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?))\b"
)


class PiiCategory(StrEnum):
    """Categories reported by transcript PII screening."""

    CARD_NUMBER = "card_number"
    EMAIL = "email"
    PHONE = "phone"
    PERSON_NAME = "person_name"


@dataclass(frozen=True, slots=True)
class PiiFinding:
    """One PII span: category plus half-open character offsets, no raw text.

    Attributes:
        category: Detected PII category.
        start: Inclusive character offset into the screened text.
        end: Exclusive character offset into the screened text.

    """

    category: PiiCategory
    start: int
    end: int

    def as_dict(self) -> dict[str, str | int]:
        """Return a JSON-safe audit representation without matched values."""
        return {"category": str(self.category), "start": self.start, "end": self.end}


@dataclass(frozen=True, slots=True)
class _ScoredFinding:
    """Internal finding with detector priority for overlap resolution."""

    finding: PiiFinding
    priority: int


def _is_card_like(candidate: str) -> bool:
    """Return whether a candidate contains at least 13 digits after separators."""
    return sum(char.isdigit() for char in candidate) >= 13


def _card_findings(text: str) -> tuple[PiiFinding, ...]:
    """Scan every digit offset for cards, independent of other detector matches.

    ``finditer`` yields non-overlapping matches, so a greedy candidate beginning
    inside a phone number could otherwise consume the following card. Trying a
    fresh match at each digit offset examines the later card directly. The regex
    consumes whole digit/separator runs, including repeated unicode whitespace,
    dot, slash and underscore, so no separator multiplicity can hide a card.
    Redundant candidates wholly contained in a longer accepted candidate are
    removed.

    Args:
        text: Transcript text to scan read-only.

    Returns:
        Card findings with category ``CARD_NUMBER`` and absolute character
        offsets; no matched text is retained.

    """
    candidates: list[PiiFinding] = []
    for start, char in enumerate(text):
        if not char.isdigit():
            continue
        match = _CARD_RE.match(text, start)
        if match is None or not _is_card_like(match.group(1)):
            continue
        candidates.append(PiiFinding(PiiCategory.CARD_NUMBER, *match.span(1)))
    selected: list[PiiFinding] = []
    for finding in sorted(candidates, key=lambda item: (item.start, -(item.end - item.start))):
        if any(finding.start >= kept.start and finding.end <= kept.end for kept in selected):
            continue
        selected.append(finding)
    return tuple(selected)


def _collect(text: str) -> list[_ScoredFinding]:
    """Collect candidate findings from every deterministic detector."""
    collected: list[_ScoredFinding] = [
        _ScoredFinding(finding, 0) for finding in _card_findings(text)
    ]
    detectors: tuple[tuple[PiiCategory, re.Pattern[str]], ...] = (
        (PiiCategory.EMAIL, _EMAIL_RE),
        (PiiCategory.PHONE, _PHONE_INTL_RE),
        (PiiCategory.PHONE, _PHONE_NATIONAL_RE),
        (PiiCategory.PERSON_NAME, _NAME_RE),
    )
    for priority, (category, pattern) in enumerate(detectors, start=1):
        for match in pattern.finditer(text):
            group = match.lastindex or 0
            collected.append(_ScoredFinding(PiiFinding(category, *match.span(group)), priority))
    return collected


def detect_pii(text: str) -> tuple[PiiFinding, ...]:
    """Detect PII spans and categories in transcript text.

    Args:
        text: Exact transcript text to screen. No mutation or redaction occurs.

    Returns:
        Non-overlapping findings ordered by start offset. Card findings take
        absolute precedence when spans overlap, regardless of start or length.

    """
    selected: list[_ScoredFinding] = []
    ordered = sorted(
        _collect(text),
        key=lambda item: (
            item.finding.category is not PiiCategory.CARD_NUMBER,
            item.finding.start,
            -(item.finding.end - item.finding.start),
            item.priority,
        ),
    )
    for item in ordered:
        if any(
            item.finding.start < kept.finding.end and kept.finding.start < item.finding.end
            for kept in selected
        ):
            continue
        selected.append(item)
    return tuple(
        sorted(
            (item.finding for item in selected),
            key=lambda finding: (finding.start, finding.end, str(finding.category)),
        )
    )


def has_card_number(text: str) -> bool:
    """Return whether transcript text contains a hard-drop card number.

    This uses the dedicated card scan directly, so the answer is independent of
    overlap resolution against phone, email or person-name detectors.

    Args:
        text: Transcript text to scan read-only.

    Returns:
        ``True`` when any digit/separator run with at least 13 digits is present.

    """
    return bool(_card_findings(text))
