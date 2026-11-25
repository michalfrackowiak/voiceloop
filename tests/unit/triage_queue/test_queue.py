"""Queue construction, cap, slice, clustering and blind-item tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from services.triage.queue import (
    ExplanationScoreSource,
    InMemoryScoreSource,
    InMemoryTurnSource,
    QueueError,
    QueueReport,
    ScoreFileSource,
    ScoreSnapshot,
    TriageQueueSettings,
    TurnRecord,
    build_queue,
    main,
)


class _FakeExplanation:
    """P2-01 ``ScoreExplanation``-shaped object used to exercise the adapter."""

    def __init__(self, event_id: str, score: float, signal_names: tuple[str, ...] = ()) -> None:
        self.event_id = event_id
        self.score = score
        self.signal_names = signal_names


class _FakeExplanationStore:
    """Fake store exposing only the ``all()`` seam."""

    def __init__(self, explanations: tuple[_FakeExplanation, ...]) -> None:
        self._items = explanations

    def all(self) -> tuple[_FakeExplanation, ...]:
        """Return all fake explanations."""
        return self._items


def _build_fake_queue(
    make_event: Callable[..., dict[str, Any]],
    settings: TriageQueueSettings | None = None,
) -> QueueReport:
    """Build a small queue from fake events with explicit scores."""
    events = (
        make_event("a", "book a room for three nights", conf=0.4),
        make_event("b", "book a room for three nights please", conf=0.4),
        make_event("c", "cancel my reservation", conf=0.4),
        make_event("d", "check in on the twelfth", conf=0.95),
    )
    scores = InMemoryScoreSource({"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.0})
    return build_queue(InMemoryTurnSource(events), scores, settings)


def test_explanation_store_adapter_carries_signal_names(
    make_event: Callable[..., dict[str, Any]],
) -> None:
    store = _FakeExplanationStore(
        (
            _FakeExplanation("a", 0.9, ("low_word_confidence",)),
            _FakeExplanation("b", 0.8, ("decoder_stats",)),
            _FakeExplanation("c", 0.7, ()),
            _FakeExplanation("d", 0.0, ()),
        )
    )
    events = (
        make_event("a", "book a room for three nights", conf=0.4),
        make_event("b", "book a room for three nights please", conf=0.4),
        make_event("c", "cancel my reservation", conf=0.4),
        make_event("d", "check in on the twelfth", conf=0.95),
    )
    report = build_queue(
        InMemoryTurnSource(events),
        ExplanationScoreSource(store),
        TriageQueueSettings(
            random_slice_target=0,
            daily_cap=4,
            blind_share=0.0,
            cluster_similarity_threshold=0.99,
        ),
    )
    by_id = {item.event_id: item for item in report.items}
    assert by_id["a"].signals == ("low_word_confidence",)
    assert by_id["b"].score == 0.8


def test_queue_clusters_near_duplicates_and_keeps_one_representative(
    make_event: Callable[..., dict[str, Any]],
) -> None:
    report = _build_fake_queue(
        make_event,
        TriageQueueSettings(
            random_slice_target=0,
            daily_cap=10,
            blind_share=0.0,
            cluster_similarity_threshold=0.7,
        ),
    )
    assert report.stats.clusters == 2
    assert {item.event_id for item in report.items} == {"a", "c"}
    assert len(report.items) == len({item.cluster_id for item in report.items})


def test_zero_score_turns_only_enter_via_random_slice(
    make_event: Callable[..., dict[str, Any]],
) -> None:
    report = _build_fake_queue(
        make_event,
        TriageQueueSettings(random_slice_target=1, daily_cap=10, blind_share=0.0, random_seed=3),
    )
    random_items = [item for item in report.items if item.slice == "random"]
    assert len(random_items) == 1
    assert all(item.score > 0.0 for item in report.items if item.slice == "failure")


def test_priority_order_is_highest_score_first_with_interleaved_slice(
    fixture_records: tuple[TurnRecord, ...],
    fixture_scores: InMemoryScoreSource,
    queue_settings: TriageQueueSettings,
) -> None:
    report = build_queue(InMemoryTurnSource(tuple(fixture_records)), fixture_scores, queue_settings)
    failure_scores = [item.score for item in report.items if item.slice == "failure"]
    assert failure_scores == sorted(failure_scores, reverse=True)
    random_positions = [index for index, item in enumerate(report.items) if item.slice == "random"]
    assert random_positions
    assert random_positions[0] < len(report.items) - 1


def test_cap_respects_daily_limit_and_reserves_slice(
    fixture_records: tuple[TurnRecord, ...],
    fixture_scores: InMemoryScoreSource,
    queue_settings: TriageQueueSettings,
) -> None:
    report = build_queue(InMemoryTurnSource(tuple(fixture_records)), fixture_scores, queue_settings)
    assert len(report.items) == queue_settings.daily_cap
    assert len([item for item in report.items if item.slice == "random"]) == 2
    assert report.stats.random_slice_size == 2
    assert report.stats.deferred_by_cap > 0


def test_cap_smaller_than_slice_truncates_deterministically(
    fixture_records: tuple[TurnRecord, ...],
    fixture_scores: InMemoryScoreSource,
) -> None:
    settings = TriageQueueSettings(
        random_slice_target=2, daily_cap=1, random_seed=7, blind_share=0.0
    )
    report = build_queue(InMemoryTurnSource(tuple(fixture_records)), fixture_scores, settings)
    assert len(report.items) == 1
    assert report.items[0].slice == "random"
    assert report.stats.random_slice_reserved == 1
    assert report.stats.deferred_by_cap >= 1


def test_blind_share_and_review_item_contract(
    fixture_records: tuple[TurnRecord, ...],
    fixture_scores: InMemoryScoreSource,
    queue_settings: TriageQueueSettings,
) -> None:
    report = build_queue(InMemoryTurnSource(tuple(fixture_records)), fixture_scores, queue_settings)
    blind = [item for item in report.items if item.no_prefill]
    assert len(blind) == 4  # ceil(8 * 0.5)
    expected_keys = {
        "schema_version",
        "call_id",
        "turn_index",
        "audio_uri",
        "hypothesis",
        "low_confidence_spans",
        "blind",
    }
    for item in report.items:
        payload = item.to_review_item()
        assert set(payload) == expected_keys
        assert payload["blind"] is item.no_prefill
        assert payload["schema_version"] == "1.0.0"
    assert report.to_dict()["items"][0]["slice"] in {"random", "failure"}


def test_build_queue_is_repeatable(
    fixture_records: tuple[TurnRecord, ...],
    fixture_scores: InMemoryScoreSource,
    queue_settings: TriageQueueSettings,
) -> None:
    source = InMemoryTurnSource(tuple(fixture_records))
    first = build_queue(source, fixture_scores, queue_settings, day="2026-11-25")
    second = build_queue(source, fixture_scores, queue_settings, day="2026-11-25")
    assert first.to_dict() == second.to_dict()


def test_consent_and_card_number_screening(make_event: Callable[..., dict[str, Any]]) -> None:
    events = (
        make_event("good", recording_consent=True, conf=0.4),
        make_event("no-record", recording_consent=False, conf=0.4),
        make_event("training-no", "check in on the fifth", training_consent=False, conf=0.4),
        make_event(
            "card",
            conf=0.4,
            input_text="my card number is 4111 1111 1111 1111",
        ),
    )
    scores = InMemoryScoreSource({event["event_id"]: 0.5 for event in events})
    report = build_queue(
        InMemoryTurnSource(events),
        scores,
        TriageQueueSettings(random_slice_target=0, daily_cap=10, blind_share=0.0),
    )
    assert report.stats.eligible_turns == 2
    assert report.stats.excluded_turns == 2
    assert {item.event_id for item in report.items} == {"good", "training-no"}
    assert (
        next(item for item in report.items if item.event_id == "training-no").training_consent
        is False
    )


def test_low_confidence_spans_are_derived_from_event_words(
    make_event: Callable[..., dict[str, Any]],
) -> None:
    event = make_event("spans", conf=0.41)
    record = TurnRecord.from_event(event)
    spans = record.low_confidence_spans(0.6)
    assert len(spans) == 2
    assert spans[0].start_ms == 0
    assert spans[0].text == "book"
    assert spans[0].to_dict()["confidence"] == 0.41


def test_missing_score_is_an_explicit_error(make_event: Callable[..., dict[str, Any]]) -> None:
    with pytest.raises(QueueError, match="no failure score"):
        build_queue(
            InMemoryTurnSource((make_event("missing"),)),
            InMemoryScoreSource(),
            TriageQueueSettings(random_slice_target=0),
        )


def test_score_file_source_reads_p2_01_report(tmp_path: Path) -> None:
    report_path = tmp_path / "scores.json"
    report_path.write_text(
        json.dumps(
            {
                "turns": 2,
                "explanations": [
                    {"event_id": "a", "score": 0.5, "signals": [{"name": "decoder_stats"}]},
                    {"event_id": "b", "score": 0.1, "signals": []},
                ],
            }
        ),
        encoding="utf-8",
    )
    source = ScoreFileSource(report_path)
    assert source.snapshot_for("a") == ScoreSnapshot("a", 0.5, ("decoder_stats",))
    assert source.snapshot_for("b").score == 0.1
    with pytest.raises(QueueError, match="no score for event"):
        source.snapshot_for("unknown")


def test_score_file_source_reads_json_lines(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    path.write_text(
        '{"event_id": "a", "score": 0.3}\n{"event_id": "b", "score": 0.7}\n', encoding="utf-8"
    )
    assert ScoreFileSource(path).snapshot_for("b").score == 0.7


def test_cli_writes_queue_report(
    tmp_path: Path,
    turns_path: Path,
    fixture_records: tuple[TurnRecord, ...],
) -> None:
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(
        json.dumps(
            {
                "explanations": [
                    {"event_id": record.event_id, "score": 0.9 if index < 4 else 0.0}
                    for index, record in enumerate(fixture_records)
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "queue.json"
    assert (
        main(
            [
                "--fixtures",
                str(turns_path),
                "--scores",
                str(scores_path),
                "--day",
                "2026-11-25",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["day"] == "2026-11-25"
    assert 0 < len(payload["items"]) <= 30  # production daily_cap default
    assert any(item["slice"] == "random" for item in payload["items"])
