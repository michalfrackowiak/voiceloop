"""Regression-set accumulation for every reviewed failure (P2-05.3).

The regression set is an evaluation set, never a training set. It accumulates
reviewed failures keyed by ``(call_id, turn_index)`` so re-reviewing the same
turn cannot duplicate it. Fixture ``failure_type`` labels are the reviewed
signal used by the synthetic workflow; ``both_fine``/``clean`` turns are the
unbiased random slice (context.md §2 invariant 6) and are not failures.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voiceloop_common import ValidationError, utcnow

from .examples import REVIEW_FAILURE_LABELS, ReviewedItem


def is_reviewed_failure(label: str | None) -> bool:
    """Return whether a review/failure label is a failure to keep for regression."""
    return label in REVIEW_FAILURE_LABELS


def utc_timestamp() -> str:
    """Return the injectable UTC clock as an RFC 3339 string."""
    return utcnow().isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class RegressionRecord:
    """One accumulated reviewed failure with PII-free audit metadata."""

    call_id: str
    turn_index: int
    routing: str
    corrected_transcript: str
    slots: dict[str, Any]
    reviewer_id: str
    created_at: str

    @property
    def source_id(self) -> str:
        """Return the stable ``call_id:turn_index`` key."""
        return f"{self.call_id}:{self.turn_index}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        return {
            "call_id": self.call_id,
            "turn_index": self.turn_index,
            "routing": self.routing,
            "corrected_transcript": self.corrected_transcript,
            "slots": dict(self.slots),
            "reviewer_id": self.reviewer_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegressionRecord:
        """Rebuild a record from one JSONL row."""
        for field in ("call_id", "turn_index", "routing", "corrected_transcript", "slots"):
            if field not in data:
                raise ValidationError(f"regression record missing {field}")
        return cls(
            call_id=str(data["call_id"]),
            turn_index=int(data["turn_index"]),
            routing=str(data["routing"]),
            corrected_transcript=str(data["corrected_transcript"]),
            slots=dict(data["slots"]),
            reviewer_id=str(data.get("reviewer_id", "")),
            created_at=str(data.get("created_at", "")),
        )


class RegressionStore:
    """Append-only, deduplicating in-memory regression set with JSONL I/O."""

    def __init__(self, records: Iterable[RegressionRecord] = ()) -> None:
        """Initialise the store from existing records, deduplicated by source ID."""
        self._records: dict[str, RegressionRecord] = {
            record.source_id: record for record in records
        }

    def accumulate(
        self,
        items: Iterable[ReviewedItem],
        *,
        created_at: str | None = None,
    ) -> tuple[RegressionRecord, ...]:
        """Add every reviewed failure, ignoring repeats of the same turn.

        Args:
            items: Reviewed items whose fixture/routing label is the signal.
            created_at: Optional UTC timestamp for deterministic tests.

        Returns:
            All records currently in the store, sorted by source ID.

        """
        timestamp = created_at or utc_timestamp()
        for item in items:
            label = item.failure_type or item.routing
            if not is_reviewed_failure(label):
                continue
            record = RegressionRecord(
                call_id=item.call_id,
                turn_index=item.turn_index,
                routing=item.routing,
                corrected_transcript=item.corrected_transcript,
                slots=dict(item.slots),
                reviewer_id=item.reviewer_id,
                created_at=timestamp,
            )
            self._records.setdefault(record.source_id, record)
        return self.records()

    def records(self) -> tuple[RegressionRecord, ...]:
        """Return all records sorted by source ID."""
        return tuple(sorted(self._records.values(), key=lambda record: record.source_id))

    def call_ids(self) -> tuple[str, ...]:
        """Return the sorted call IDs held by the regression set."""
        return tuple(sorted({record.call_id for record in self._records.values()}))

    def write(self, path: str | Path) -> Path:
        """Write the canonical JSONL regression set to ``path``."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(record.to_dict(), sort_keys=True) for record in self.records()]
        destination.write_text(
            "".join(f"{line}\n" for line in lines),
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> RegressionStore:
        """Read a JSONL regression set, rejecting malformed rows."""
        source = Path(path)
        records: list[RegressionRecord] = []
        for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"invalid regression JSON on line {number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValidationError(f"regression line {number} must be an object")
            records.append(RegressionRecord.from_dict(row))
        return cls(records)
