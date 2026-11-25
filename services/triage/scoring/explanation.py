"""Score explanation types and the per-turn explanation store contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class FiredSignal:
    """One deterministic signal that contributed to a turn's failure score."""

    name: str
    weight: float
    severity: float
    reason: str

    @property
    def contribution(self) -> float:
        """Return the weighted contribution before score normalisation."""
        return round(self.weight * self.severity, 6)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of this fired signal."""
        return {
            "name": self.name,
            "weight": self.weight,
            "severity": self.severity,
            "contribution": self.contribution,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ScoreExplanation:
    """A per-turn score plus the complete list of signals that fired."""

    event_id: str
    call_id: str
    turn_index: int
    score: float
    signals: tuple[FiredSignal, ...] = ()

    @property
    def signal_names(self) -> tuple[str, ...]:
        """Return fired signal names in the order they were detected."""
        return tuple(signal.name for signal in self.signals)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable explanation record."""
        return {
            "event_id": self.event_id,
            "call_id": self.call_id,
            "turn_index": self.turn_index,
            "score": self.score,
            "signals": [signal.to_dict() for signal in self.signals],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ScoreExplanation:
        """Rebuild an explanation from its JSON view.

        Args:
            payload: A dict previously produced by :meth:`to_dict`.

        Returns:
            The reconstructed immutable explanation.

        """
        signals = tuple(
            FiredSignal(
                name=str(item["name"]),
                weight=float(item["weight"]),
                severity=float(item["severity"]),
                reason=str(item["reason"]),
            )
            for item in payload.get("signals", ())
        )
        return cls(
            event_id=str(payload["event_id"]),
            call_id=str(payload["call_id"]),
            turn_index=int(payload["turn_index"]),
            score=float(payload["score"]),
            signals=signals,
        )


@runtime_checkable
class ExplanationStore(Protocol):
    """Persistence seam for per-turn score explanations.

    The in-memory implementation is the default fake. A real store should upsert
    on ``event_id`` (idempotent re-scoring), index by ``(call_id, turn_index)``,
    and retain explanations under the same policy as turn metadata. No audio or
    transcript text is written by this store beyond the human-readable reasons.
    """

    def put(self, explanation: ScoreExplanation) -> None:
        """Store or replace the explanation for one event."""
        ...

    def get(self, event_id: str) -> ScoreExplanation | None:
        """Return the explanation for ``event_id`` or ``None`` when absent."""
        ...

    def all(self) -> tuple[ScoreExplanation, ...]:
        """Return every stored explanation in insertion order."""
        ...


@dataclass
class InMemoryExplanationStore:
    """Deterministic fake store used by default and in tests."""

    _items: dict[str, ScoreExplanation] = field(default_factory=dict)

    def put(self, explanation: ScoreExplanation) -> None:
        """Store or replace the explanation for one event."""
        self._items[explanation.event_id] = explanation

    def get(self, event_id: str) -> ScoreExplanation | None:
        """Return the explanation for ``event_id`` or ``None`` when absent."""
        return self._items.get(event_id)

    def all(self) -> tuple[ScoreExplanation, ...]:
        """Return every stored explanation in insertion order."""
        return tuple(self._items.values())
