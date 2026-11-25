"""Near-duplicate clustering tests for P2-02."""

from __future__ import annotations

from services.triage.queue import ClusterItem, cluster_turns, normalise_text, text_similarity


def test_normalise_text_removes_punctuation_and_case() -> None:
    assert normalise_text("  BookA room, for  THREE nights! ") == "booka room for three nights"


def test_similarity_of_identical_and_unrelated_text() -> None:
    assert text_similarity("Three nights, please", "three nights please") == 1.0
    assert text_similarity("", "") == 1.0
    assert text_similarity("", "three nights") == 0.0
    assert text_similarity("book a room", "cancel the flight") < 0.4


def test_similarity_is_symmetric_and_bounded() -> None:
    left = "I would like a deluxe room for two guests"
    right = "a deluxe room, two guests please"
    score = text_similarity(left, right)
    assert score == text_similarity(right, left)
    assert 0.0 <= score <= 1.0


def test_near_duplicates_form_one_cluster_with_one_representative() -> None:
    items = (
        ClusterItem("a", "Book a room for three nights", 0.2),
        ClusterItem("b", "book a room for three nights!", 0.4),
        ClusterItem("c", "book a room for three nights please", 0.1),
        ClusterItem("d", "cancel my reservation tomorrow", 0.9),
    )
    clusters = cluster_turns(items, threshold=0.7)
    assert len(clusters) == 2
    assert clusters[0].members == ("a", "b", "c")
    assert clusters[0].representative == "b"  # highest score in that cluster
    assert clusters[1].members == ("d",)


def test_cluster_ids_are_stable_regardless_of_input_order() -> None:
    items = (
        ClusterItem("a", "check in on the twelfth", 0.3),
        ClusterItem("b", "check in on the twelfth please", 0.6),
        ClusterItem("c", "totally different request", 0.1),
    )
    forward = {cluster.cluster_id: cluster.members for cluster in cluster_turns(items, 0.7)}
    backward = {
        cluster.cluster_id: cluster.members
        for cluster in cluster_turns(tuple(reversed(items)), 0.7)
    }
    assert forward == backward


def test_complete_link_prevents_transitive_chaining() -> None:
    items = (
        ClusterItem("a", "the quick brown fox runs", 0.1),
        ClusterItem("b", "the quick brown fox runs fast", 0.2),
        ClusterItem("c", "fast trains cross the icy bridge", 0.9),
    )
    clusters = cluster_turns(items, threshold=0.65)
    assert clusters[0].members == ("a", "b")
    assert clusters[1].members == ("c",)
