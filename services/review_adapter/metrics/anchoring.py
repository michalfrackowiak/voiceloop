"""Descriptive blind-vs-prefilled agreement on explicitly paired fixture items."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .events import CandidatePair, CandidatePairSource, ReviewResultLike

NO_CAUSAL_CLAIM = "descriptive_only_no_causal_claim"
"""Fixed report string: paired synthetic answers cannot establish anchoring effects."""


def answers_agree(first: Mapping[str, object], second: Mapping[str, object]) -> bool:
    """Return whether two scripted slot objects are exactly equal.

    Args:
        first: Slot mapping from one synthetic variant.
        second: Slot mapping from the paired synthetic variant.

    Returns:
        ``True`` when key sets and nested values match exactly.

    """
    return dict(first) == dict(second)


@dataclass(frozen=True)
class AnchoringMetrics:
    """Agreement counts and rates for explicit blind/prefilled fixture pairs."""

    paired_items: int
    unpaired_events: int
    transcript_agreement_count: int
    slots_agreement_count: int
    both_agreement_count: int
    transcript_agreement_rate: float | None
    slots_agreement_rate: float | None
    both_agreement_rate: float | None
    interpretation: str = NO_CAUSAL_CLAIM

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable view containing no answer text."""
        return {
            "paired_items": self.paired_items,
            "unpaired_events": self.unpaired_events,
            "transcript_agreement_count": self.transcript_agreement_count,
            "slots_agreement_count": self.slots_agreement_count,
            "both_agreement_count": self.both_agreement_count,
            "transcript_agreement_rate": self.transcript_agreement_rate,
            "slots_agreement_rate": self.slots_agreement_rate,
            "both_agreement_rate": self.both_agreement_rate,
            "interpretation": self.interpretation,
        }


def _rate(count: int, total: int) -> float | None:
    """Return ``count / total`` or ``None`` when there are no paired items."""
    return count / total if total else None


def compare_prefill_anchoring(
    source: CandidatePairSource,
    results: Sequence[ReviewResultLike],
) -> AnchoringMetrics:
    """Compare paired blind vs prefilled fake answers descriptively.

    Args:
        source: Source of explicit pairs; no pair is inferred from timestamps or scores.
        results: All fixture results, used only to count events without both variants.

    Returns:
        Transcript, slot and combined agreement counts and rates. Agreement means
        identical scripted answers in this synthetic stand-in, not reviewer quality.

    """
    pairs: tuple[CandidatePair, ...] = tuple(source.iter_pairs())
    paired_events = {pair.event_id for pair in pairs}
    unpaired_events = len({result.event_id for result in results} - paired_events)
    transcript_agreements = sum(
        pair.blind.corrected_transcript == pair.prefilled.corrected_transcript for pair in pairs
    )
    slot_agreements = sum(answers_agree(pair.blind.slots, pair.prefilled.slots) for pair in pairs)
    both_agreements = sum(
        pair.blind.corrected_transcript == pair.prefilled.corrected_transcript
        and answers_agree(pair.blind.slots, pair.prefilled.slots)
        for pair in pairs
    )
    total = len(pairs)
    return AnchoringMetrics(
        paired_items=total,
        unpaired_events=unpaired_events,
        transcript_agreement_count=transcript_agreements,
        slots_agreement_count=slot_agreements,
        both_agreement_count=both_agreements,
        transcript_agreement_rate=_rate(transcript_agreements, total),
        slots_agreement_rate=_rate(slot_agreements, total),
        both_agreement_rate=_rate(both_agreements, total),
    )
