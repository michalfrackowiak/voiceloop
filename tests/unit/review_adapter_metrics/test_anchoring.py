"""Blind-versus-prefilled anchoring comparison tests."""

from __future__ import annotations

from collections.abc import Iterator

from services.review_adapter.metrics.anchoring import (
    NO_CAUSAL_CLAIM,
    answers_agree,
    compare_prefill_anchoring,
)
from services.review_adapter.metrics.events import CandidateAnswer, CandidatePair
from tests.unit.review_adapter_metrics.conftest import make_result


class StaticPairSource:
    """In-memory candidate pair source for deterministic anchoring tests."""

    def __init__(self, pairs: tuple[CandidatePair, ...]) -> None:
        """Freeze explicit pairs for one test."""
        self._pairs = pairs

    def iter_pairs(self) -> Iterator[CandidatePair]:
        """Yield the frozen pairs."""
        return iter(self._pairs)


def _pair(
    event_id: str,
    *,
    blind_text: str = "same",
    prefilled_text: str = "same",
    blind_slots: dict[str, object] | None = None,
    prefilled_slots: dict[str, object] | None = None,
) -> CandidatePair:
    """Build one explicit blind/prefilled fake pair."""
    return CandidatePair(
        event_id=event_id,
        call_id="call-1",
        turn_index=0,
        blind=CandidateAnswer("reviewer-blind", blind_text, blind_slots or {}),
        prefilled=CandidateAnswer("reviewer-prefilled", prefilled_text, prefilled_slots or {}),
    )


def test_answers_agree_uses_exact_slot_equality() -> None:
    """Only identical key/value structures count as scripted agreement."""
    assert answers_agree({"guests": 2}, {"guests": 2})
    assert not answers_agree({"guests": 2}, {"guests": 3})
    assert not answers_agree({"guests": 2}, {"guests": 2, "room_type": "deluxe"})


def test_full_agreement_rates_and_no_causal_claim() -> None:
    """Identical scripted pairs produce agreement without an anchoring claim."""
    source = StaticPairSource((_pair("event-1"), _pair("event-2")))

    metrics = compare_prefill_anchoring(source, ())

    assert metrics.paired_items == 2
    assert metrics.unpaired_events == 0
    assert metrics.transcript_agreement_count == 2
    assert metrics.slots_agreement_count == 2
    assert metrics.both_agreement_count == 2
    assert metrics.transcript_agreement_rate == 1.0
    assert metrics.slots_agreement_rate == 1.0
    assert metrics.both_agreement_rate == 1.0
    assert metrics.interpretation == NO_CAUSAL_CLAIM
    assert "text" not in metrics.to_dict()
    assert "slots" not in metrics.to_dict()


def test_mixed_agreement_counts_unpaired_events() -> None:
    """A missing variant is counted as unpaired, never silently paired."""
    pair = _pair("event-1", blind_text="blind", prefilled_text="prefilled")
    source = StaticPairSource((pair,))
    results = (make_result("event-1", blind=True), make_result("event-2", blind=True))

    metrics = compare_prefill_anchoring(source, results)

    assert metrics.paired_items == 1
    assert metrics.unpaired_events == 1
    assert metrics.transcript_agreement_count == 0
    assert metrics.slots_agreement_count == 1
    assert metrics.both_agreement_count == 0
    assert metrics.transcript_agreement_rate == 0.0
    assert metrics.slots_agreement_rate == 1.0
    assert metrics.both_agreement_rate == 0.0


def test_empty_pairs_have_null_rates() -> None:
    """No explicit pairs means no agreement rate can be reported."""
    metrics = compare_prefill_anchoring(StaticPairSource(()), ())

    assert metrics.paired_items == 0
    assert metrics.transcript_agreement_rate is None
    assert metrics.slots_agreement_rate is None
    assert metrics.both_agreement_rate is None
