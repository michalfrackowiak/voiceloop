"""Expansion of reviewed items into STT and parser training examples (P2-05.1).

One human-reviewed item can yield up to three examples: audio → corrected
transcript (STT), corrected transcript → reviewed slots (clean parser), and
original hypothesis → reviewed slots (noisy parser). Every example keeps the
dialogue context from its source turn. No transcript or slot value is invented:
all text comes from the read-only turn event or the reviewed result.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from voiceloop_common import ValidationError

ExampleKind = Literal["stt", "parser_clean", "parser_noisy"]

REVIEW_ROUTINGS: tuple[str, ...] = (
    "stt_wrong",
    "parser_wrong",
    "both_fine",
    "unusable_audio",
)

REVIEW_FAILURE_LABELS: frozenset[str] = frozenset({"stt_wrong", "parser_wrong", "unusable_audio"})
REVIEW_OK_LABELS: frozenset[str] = frozenset({"clean", "both_fine"})

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class ReviewedItem:
    """One human-reviewed turn with its source event and corrected outputs.

    Attributes:
        event: Read-only turn event conforming to the 1.0.0 event contract.
        corrected_transcript: Human-corrected transcript from review stage 1.
        slots: Reviewed booking slots from review stage 2.
        routing: Review adapter routing label (``both_fine`` for a clean turn).
        reviewer_id: Pseudonymous reviewer identifier, if known.
        failure_type: Fixture/triaged failure label used as the reviewed signal.
        synthetic: Whether the source turn is synthetic (must be tagged).

    """

    event: Mapping[str, Any]
    corrected_transcript: str
    slots: dict[str, Any]
    routing: str
    reviewer_id: str = ""
    failure_type: str | None = None
    synthetic: bool = False

    def __post_init__(self) -> None:
        """Reject routing labels outside the review adapter contract."""
        if self.routing not in REVIEW_ROUTINGS:
            raise ValidationError(f"unknown review routing label: {self.routing!r}")

    @property
    def call_id(self) -> str:
        """Return the source call ID used for all split decisions."""
        return str(self.event.get("call_id", ""))

    @property
    def turn_index(self) -> int:
        """Return the zero-based turn index within the source call."""
        raw = self.event.get("turn_index", -1)
        return raw if isinstance(raw, int) and not isinstance(raw, bool) else -1

    @property
    def source_id(self) -> str:
        """Return the stable ``call_id:turn_index`` lineage identifier."""
        return f"{self.call_id}:{self.turn_index}"

    @property
    def hypothesis(self) -> str:
        """Return the original STT hypothesis from the source event."""
        stt = self.event.get("stt")
        text = stt.get("text", "") if isinstance(stt, Mapping) else ""
        return text if isinstance(text, str) else ""

    @property
    def audio_uri(self) -> str:
        """Return the source audio URI, or an empty string when absent."""
        audio = self.event.get("audio")
        uri = audio.get("uri", "") if isinstance(audio, Mapping) else ""
        return str(uri)

    @property
    def duration_ms(self) -> int:
        """Return the source audio duration in milliseconds, when available."""
        audio = self.event.get("audio")
        raw = audio.get("duration_ms", 0) if isinstance(audio, Mapping) else 0
        return raw if isinstance(raw, int) and not isinstance(raw, bool) else 0


@dataclass(frozen=True, slots=True)
class DatasetExample:
    """One training example expanded from a reviewed item."""

    example_id: str
    kind: ExampleKind
    call_id: str
    turn_index: int
    audio_uri: str
    source_text: str
    target_text: str
    slots: dict[str, Any]
    dialogue_context: dict[str, Any]
    routing: str
    corrected_transcript: str
    hypothesis: str
    duration_ms: int
    synthetic: bool = False
    replay: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy for manifests and exports."""
        return {
            "example_id": self.example_id,
            "kind": self.kind,
            "call_id": self.call_id,
            "turn_index": self.turn_index,
            "audio_uri": self.audio_uri,
            "source_text": self.source_text,
            "target_text": self.target_text,
            "slots": dict(self.slots),
            "dialogue_context": dict(self.dialogue_context),
            "routing": self.routing,
            "corrected_transcript": self.corrected_transcript,
            "hypothesis": self.hypothesis,
            "duration_ms": self.duration_ms,
            "synthetic": self.synthetic,
            "replay": self.replay,
        }


def _resolve_path(path: str | Path, root: Path | None = None) -> Path:
    """Resolve a repo-relative fixture URI without mutating it."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (root or REPO_ROOT) / candidate


def build_dialogue_context(
    event: Mapping[str, Any],
    turn_index: int,
    *,
    prior_turns_limit: int = 5,
    root: Path | None = None,
) -> dict[str, Any]:
    """Return the event context plus prior-turn hypotheses from its reference.

    Args:
        event: Read-only source turn event.
        turn_index: Index of the current turn; only earlier turns are included.
        prior_turns_limit: Maximum number of earlier turns to keep (0 disables).
        root: Repository root used to resolve repo-relative context URIs.

    Returns:
        A plain JSON-serialisable context object. Missing or malformed context
        references yield an empty ``prior_turns`` list rather than an error.

    """
    raw = event.get("dialogue_context")
    context = dict(raw) if isinstance(raw, Mapping) else {}
    prior: list[dict[str, Any]] = []
    reference = context.get("prior_turns_ref")
    if prior_turns_limit > 0 and isinstance(reference, str) and reference:
        path = _resolve_path(reference, root)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, Mapping):
                continue
            index = row.get("turn_index")
            if not isinstance(index, int) or isinstance(index, bool) or index >= turn_index:
                continue
            prior.append({"turn_index": index, "hypothesis": str(row.get("hypothesis", ""))})
    context["prior_turns"] = sorted(prior, key=lambda row: row["turn_index"])[-prior_turns_limit:]
    return context


def expand_reviewed_item(
    item: ReviewedItem,
    *,
    max_examples: int = 3,
    prior_turns_limit: int = 5,
    root: Path | None = None,
) -> tuple[DatasetExample, ...]:
    """Expand one reviewed item into up to three lineage-tracked examples.

    STT uses the original hypothesis as the noisy source and the corrected
    transcript as the target; the clean parser example uses the corrected
    transcript; the noisy parser example uses the original hypothesis. Empty
    targets (silence/noise turns) and turns with no reviewed slots yield fewer
    examples, never fabricated ones.

    Args:
        item: Reviewed item built from a source event and human result.
        max_examples: Upper bound between 1 and 3 inclusive.
        prior_turns_limit: Maximum number of earlier turns kept in context.
        root: Repository root for resolving context URIs.

    Returns:
        Examples in STT, clean-parser, noisy-parser order.

    Raises:
        ValidationError: If ``max_examples`` is outside 1..3.

    """
    if not 1 <= max_examples <= 3:
        raise ValidationError("max_examples must be between 1 and 3")
    context = build_dialogue_context(
        item.event,
        item.turn_index,
        prior_turns_limit=prior_turns_limit,
        root=root,
    )
    base: dict[str, Any] = {
        "call_id": item.call_id,
        "turn_index": item.turn_index,
        "audio_uri": item.audio_uri,
        "dialogue_context": context,
        "routing": item.routing,
        "corrected_transcript": item.corrected_transcript,
        "hypothesis": item.hypothesis,
        "duration_ms": item.duration_ms,
        "synthetic": item.synthetic,
    }
    candidates: list[DatasetExample] = []
    if item.audio_uri and item.corrected_transcript.strip():
        candidates.append(
            DatasetExample(
                example_id=f"{item.source_id}:stt",
                kind="stt",
                source_text=item.hypothesis,
                target_text=item.corrected_transcript,
                slots={},
                **base,
            )
        )
    if item.slots and item.corrected_transcript.strip():
        candidates.append(
            DatasetExample(
                example_id=f"{item.source_id}:parser_clean",
                kind="parser_clean",
                source_text=item.corrected_transcript,
                target_text="",
                slots=dict(item.slots),
                **base,
            )
        )
    if item.slots and item.hypothesis and item.hypothesis != item.corrected_transcript:
        candidates.append(
            DatasetExample(
                example_id=f"{item.source_id}:parser_noisy",
                kind="parser_noisy",
                source_text=item.hypothesis,
                target_text="",
                slots=dict(item.slots),
                **base,
            )
        )
    return tuple(candidates[:max_examples])
