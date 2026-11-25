"""Structural privacy screen for synthetic UI fixtures and generated reports.

This is a deterministic allowlist/pattern check, not a general-purpose name
detector. It rejects emails, long digit runs, phone-like strings, audio/store
URIs and free-form prose. Fixture authors must use the pseudonymous ID grammar
documented in ``ui/README.md``; real data must never be pasted into this tree.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .constants import SYNTHETIC_LABEL, USABILITY_ACCEPTANCE
from .errors import UiValidationError

EMAIL_PATTERN = re.compile(r"[\w.+!#$%&'*/-]+@[\w-]+(?:\.[\w-]+)+")
LONG_DIGITS_PATTERN = re.compile(r"\d{7,}")
PHONE_PATTERN = re.compile(r"\+?\d[\d ().\-]{6,}\d")
URI_PATTERN = re.compile(r"(?:https?|s3|file)://", re.IGNORECASE)
RFC3339_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
UUID_PATTERN = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")
HEX_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
PSEUDONYM_PATTERN = re.compile(
    r"(?:tok-[a-z0-9]+|(?:call|event|item|reviewer|approval|audio|guest)-token-\d+|"
    r"\d{4}-\d{2}-\d{2}|(?:standard|deluxe|suite))\Z"
)

FIXED_LABELS: frozenset[str] = frozenset(
    {
        SYNTHETIC_LABEL,
        USABILITY_ACCEPTANCE,
        "NOT DONE",
        "SYNTHETIC/FIXTURE STAND-IN - NOT A PRODUCTION REVIEW",
        "internal-1.0.0",
        "open",
        "release_blocker",
    }
)


def is_pseudonym(value: str) -> bool:
    """Return whether ``value`` is an allowed pseudonymous fixture token."""
    return PSEUDONYM_PATTERN.fullmatch(value) is not None


def assert_no_sensitive_text(node: Any) -> None:
    """Reject PII-looking strings anywhere in a synthetic payload.

    Args:
        node: JSON-like mapping, sequence or scalar to screen.

    Raises:
        UiValidationError: If an email, long digit run, phone-like string,
            object-store URI or unallowlisted prose is found.

    """
    if isinstance(node, Mapping):
        for key, value in node.items():
            assert_no_sensitive_text(key)
            # Descriptions come only from the hash-pinned, read-only booking
            # schema whose digest is checked before any build.
            if key == "description" and isinstance(value, str):
                continue
            assert_no_sensitive_text(value)
        return
    if isinstance(node, (list, tuple)):
        for value in node:
            assert_no_sensitive_text(value)
        return
    if not isinstance(node, str):
        return
    if (
        node in FIXED_LABELS
        or is_pseudonym(node)
        or RFC3339_PATTERN.fullmatch(node)
        or UUID_PATTERN.fullmatch(node)
        or HEX_DIGEST_PATTERN.fullmatch(node)
    ):
        return
    if (
        EMAIL_PATTERN.search(node)
        or LONG_DIGITS_PATTERN.search(node)
        or PHONE_PATTERN.search(node)
        or URI_PATTERN.search(node)
        or " " in node
        or "@" in node
    ):
        raise UiValidationError(f"non-pseudonymous string in synthetic payload: {node!r}")
