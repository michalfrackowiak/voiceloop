"""Deterministic disagreement scoring between teacher and production transcripts.

The score is a token-level symmetric difference normalised by the longer token
sequence. It uses only the two transcript strings and their content hashes; no
fixture label, model output beyond the injected seam, or audio byte is consulted.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from voiceloop_common import ValidationError

from .batch import TurnRecord, canonical_hash
from .settings import ARTIFACT_SCHEMA_VERSION, SIGNAL_NAME

_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> tuple[str, ...]:
    """Return lowercase alphanumeric tokens in stable order.

    Args:
        text: Production or teacher transcript text.

    Returns:
        Tokens suitable for a multiset diff.

    """
    return tuple(_TOKEN_PATTERN.findall(text.casefold()))


@dataclass(frozen=True)
class ProductionTranscript:
    """Production STT text with a content hash and source label."""

    text: str
    content_hash: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable transcript view."""
        return {"text": self.text, "content_hash": self.content_hash, "source": self.source}


@runtime_checkable
class ProductionTranscriptSource(Protocol):
    """Injectable source of the transcript that actually served production."""

    def transcript_for(self, record: TurnRecord) -> ProductionTranscript:
        """Return the production transcript for one turn."""
        ...


class EventProductionTranscriptSource:
    """Fake source replaying ``stt.text`` from a committed turn event."""

    def transcript_for(self, record: TurnRecord) -> ProductionTranscript:
        """Return the event's own STT text; no other event field is used."""
        stt = record.event.get("stt")
        text = stt.get("text", "") if isinstance(stt, Mapping) else ""
        text = text if isinstance(text, str) else ""
        digest = canonical_hash(
            {
                "kind": "voiceloop.production.transcript",
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "event_id": record.event_id,
                "text": text,
            }
        )
        return ProductionTranscript(
            text=text, content_hash=digest, source="fake://production/event_stt_text"
        )


class InMemoryProductionTranscriptSource:
    """Deterministic in-memory fake keyed by event ID, for unit tests."""

    def __init__(self, texts: Mapping[str, str | ProductionTranscript]) -> None:
        """Index caller-supplied production transcripts.

        Args:
            texts: Event ID to text or ready-made :class:`ProductionTranscript`.

        """
        self._texts = dict(texts)

    def transcript_for(self, record: TurnRecord) -> ProductionTranscript:
        """Return the indexed transcript or raise :class:`ValidationError`."""
        value = self._texts.get(record.event_id)
        if value is None:
            raise ValidationError(f"no production transcript for {record.event_id}")
        if isinstance(value, ProductionTranscript):
            return value
        return ProductionTranscript(
            text=value,
            content_hash=canonical_hash(
                {
                    "kind": "voiceloop.production.transcript",
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "event_id": record.event_id,
                    "text": value,
                }
            ),
            source="fake://production/in_memory",
        )


@dataclass(frozen=True)
class Disagreement:
    """One deterministic teacher-vs-production difference signal."""

    event_id: str
    name: str
    score: float
    changed_tokens: int
    denominator: int
    teacher_text_hash: str
    production_text_hash: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable signal view."""
        return {
            "event_id": self.event_id,
            "name": self.name,
            "score": self.score,
            "changed_tokens": self.changed_tokens,
            "denominator": self.denominator,
            "teacher_text_hash": self.teacher_text_hash,
            "production_text_hash": self.production_text_hash,
            "reason": self.reason,
        }


def score_disagreement(
    event_id: str,
    production_text: str,
    teacher_text: str,
    *,
    production_hash: str,
    teacher_hash: str,
    name: str = SIGNAL_NAME,
) -> Disagreement:
    """Compute a bounded token-level disagreement score.

    The score is ``changed_tokens / max(len(production_tokens), len(teacher_tokens))``
    where ``changed_tokens`` counts both multiset deletions and insertions. Two
    empty transcripts agree; exactly one empty transcript scores 1.0.

    Args:
        event_id: Lineage key for the compared turn.
        production_text: Transcript that served production.
        teacher_text: Offline teacher transcript.
        production_hash: Hash of the production transcript record.
        teacher_hash: Hash of the teacher transcript record.
        name: Signal name; defaults to the task-level constant.

    Returns:
        A disagreement in ``[0, 1]`` rounded to six decimals.

    Raises:
        ValidationError: If hashes are missing or texts are not strings.

    """
    if not isinstance(production_text, str) or not isinstance(teacher_text, str):
        raise ValidationError("transcripts must be strings")
    if not production_hash or not teacher_hash:
        raise ValidationError("both transcript content hashes are required")
    production_tokens = tokenize(production_text)
    teacher_tokens = tokenize(teacher_text)
    denominator = max(len(production_tokens), len(teacher_tokens))
    if denominator == 0:
        changed = 0
        score = 0.0
        reason = "both transcripts are empty; teacher agrees"
    elif not production_tokens or not teacher_tokens:
        changed = denominator
        score = 1.0
        reason = "one transcript is empty; full disagreement"
    else:
        production_counts = Counter(production_tokens)
        teacher_counts = Counter(teacher_tokens)
        changed = int(
            sum((production_counts - teacher_counts).values())
            + sum((teacher_counts - production_counts).values())
        )
        score = round(min(1.0, changed / denominator), 6)
        reason = f"{changed} token changes over {denominator} tokens"
    return Disagreement(
        event_id=event_id,
        name=name,
        score=score,
        changed_tokens=changed,
        denominator=denominator,
        production_text_hash=production_hash,
        teacher_text_hash=teacher_hash,
        reason=reason,
    )
