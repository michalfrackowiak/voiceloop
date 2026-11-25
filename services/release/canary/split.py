"""Explicit, bounded, call-level traffic splitting for the canary rehearsal.

A single pseudonymous ``call_id`` is hashed with a configured salt into a stable
bucket in ``[0, 1)``. A bucket below ``canary_fraction`` joins the canary arm;
everything else stays on the control (champion) arm. The unit is always the
call, never the utterance (``context.md`` §2 invariant 7), the fraction is
bounded in :class:`~services.release.canary.settings.CanarySettings`, and the
assignment is reproducible without randomness or wall-clock input.

No audio, transcript or guest detail is read or stored here: only a
pseudonymous identifier and the derived arm/bucket leave this module.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from voiceloop_common import ValidationError

from .settings import CanarySettings

_DIGIT_RUN = re.compile(r"\d{13,19}")


class Arm(StrEnum):
    """The two bounded arms of the simulated traffic split."""

    CHAMPION = "champion"
    CANARY = "canary"


@dataclass(frozen=True)
class SplitAssignment:
    """One deterministic, PII-safe traffic-split decision."""

    call_id: str
    arm: Arm
    bucket: float

    def to_fields(self) -> dict[str, object]:
        """Return allowlisted structured-log fields for this assignment."""
        return {"call_id": self.call_id, "arm": self.arm.value, "bucket": self.bucket}


class TrafficSplit:
    """Deterministic call-level split with no mutable assignment state."""

    def __init__(self, settings: CanarySettings) -> None:
        """Bind the bounded fraction and salt from typed settings."""
        self._fraction = settings.canary_fraction
        self._salt = settings.split_salt

    @property
    def fraction(self) -> float:
        """Return the configured canary fraction (bounded by settings)."""
        return self._fraction

    @property
    def salt(self) -> str:
        """Return the non-secret salt that stabilises bucket assignment."""
        return self._salt

    def assign(self, call_id: str) -> SplitAssignment:
        """Map one pseudonymous call ID to its arm.

        Args:
            call_id: Opaque, pseudonymous call identifier. Long digit runs are
                rejected so an unredacted phone/card number cannot be logged
                as a call ID.

        Returns:
            The call's stable arm and numeric bucket.

        Raises:
            voiceloop_common.ValidationError: If ``call_id`` is blank or looks
                like a raw 13-19 digit identifier.

        """
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValidationError("call_id must be a non-blank pseudonymous identifier")
        if _DIGIT_RUN.fullmatch(call_id):
            raise ValidationError("call_id must be pseudonymous, not a raw digit run")

        payload = f"{self._salt}\x00{call_id}".encode()
        digest = hashlib.sha256(payload).digest()
        bucket = int.from_bytes(digest[:8], "big") / 2**64
        arm = Arm.CANARY if bucket < self._fraction else Arm.CHAMPION
        return SplitAssignment(call_id=call_id, arm=arm, bucket=bucket)

    def counts(self, call_ids: Iterable[str]) -> dict[str, int]:
        """Return canary/control counts for the given pseudonymous calls."""
        result = {arm.value: 0 for arm in Arm}
        for call_id in call_ids:
            result[self.assign(call_id).arm.value] += 1
        return result
