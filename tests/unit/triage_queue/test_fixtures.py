"""Fixture-level P2-02 invariants on the read-only 36-turn manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from services.triage.queue import (
    FixtureTurnSource,
    InMemoryScoreSource,
    InMemoryTurnSource,
    TriageQueueSettings,
    TurnRecord,
    build_queue,
)


def test_fixture_source_loads_all_turns_without_labels(
    fixture_records: tuple[TurnRecord, ...],
) -> None:
    assert len(fixture_records) == 36
    assert len({record.event_id for record in fixture_records}) == 36
    assert len({(record.call_id, record.turn_index) for record in fixture_records}) == 36
    assert all(record.audio_uri for record in fixture_records)


def test_label_mutations_do_not_change_queue(
    tmp_path: Path,
    repo_root: Path,
    turns_path: Path,
    fixture_records: tuple[TurnRecord, ...],
    fixture_scores: InMemoryScoreSource,
    queue_settings: TriageQueueSettings,
) -> None:
    labels: list[dict[str, Any]] = [
        json.loads(line)
        for line in turns_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    changed = []
    for label in labels:
        mutated = dict(label)
        mutated["failure_type"] = "clean"
        mutated["expected_transcript"] = "label leakage must be ignored"
        changed.append(mutated)
    mutated_manifest = tmp_path / "turns.jsonl"
    mutated_manifest.write_text(
        "\n".join(json.dumps(label, sort_keys=True) for label in changed) + "\n",
        encoding="utf-8",
    )
    original = build_queue(
        InMemoryTurnSource(tuple(fixture_records)), fixture_scores, queue_settings, day="fixed"
    )
    from_labels = build_queue(
        FixtureTurnSource(mutated_manifest, root=repo_root),
        fixture_scores,
        queue_settings,
        day="fixed",
    )
    assert from_labels.to_dict() == original.to_dict()


def test_fixture_queue_has_one_representative_per_cluster(
    fixture_records: tuple[TurnRecord, ...],
    fixture_scores: InMemoryScoreSource,
    queue_settings: TriageQueueSettings,
) -> None:
    report = build_queue(InMemoryTurnSource(tuple(fixture_records)), fixture_scores, queue_settings)
    cluster_ids = [item.cluster_id for item in report.items]
    assert len(cluster_ids) == len(set(cluster_ids))
    for item in report.items:
        if item.slice == "random":
            assert item.cluster_id == f"random-{item.event_id}"
            assert item.cluster_size == 1
        else:
            assert item.cluster_id.startswith("cl-")
            assert item.cluster_size >= 1


def test_random_slice_bypasses_failure_filter_even_with_no_failures(
    fixture_records: tuple[TurnRecord, ...],
    queue_settings: TriageQueueSettings,
) -> None:
    all_clean = InMemoryScoreSource({record.event_id: 0.0 for record in fixture_records})
    report = build_queue(InMemoryTurnSource(tuple(fixture_records)), all_clean, queue_settings)
    assert len(report.items) == 2
    assert all(item.slice == "random" for item in report.items)
    assert all(item.score == 0.0 for item in report.items)
    assert report.stats.failure_candidates == 0
