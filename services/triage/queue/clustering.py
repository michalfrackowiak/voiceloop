"""Deterministic near-duplicate clustering of STT hypotheses (stdlib only).

Similarity combines token Jaccard overlap with a character-trigram Dice score,
which tolerates punctuation and small wording differences without any ML
dependency. Clustering is complete-link: a candidate joins a cluster only when
it is similar enough to *every* member, so transitive chaining cannot merge
unrelated utterances. All ordering is explicit, making cluster IDs and
representatives stable across runs.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

_TOKEN_SEPARATOR = re.compile(r"[^a-z0-9]+")


def normalise_text(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace in ``text``.

    Args:
        text: Raw STT hypothesis.

    Returns:
        A canonical token string; empty when there are no alphanumeric tokens.

    """
    return " ".join(_TOKEN_SEPARATOR.sub(" ", text.casefold()).split())


def _trigrams(text: str) -> frozenset[str]:
    """Return padded character trigrams for a normalised string."""
    padded = f"  {text} "
    if len(padded) < 3:
        return frozenset({padded})
    return frozenset(padded[index : index + 3] for index in range(len(padded) - 2))


def text_similarity(left: str, right: str) -> float:
    """Score two hypotheses in ``[0, 1]`` using token and character overlap.

    Args:
        left: First raw hypothesis.
        right: Second raw hypothesis.

    Returns:
        ``0.6 * token_jaccard + 0.4 * character_trigram_dice``. Two empty
        hypotheses are identical; one empty and one non-empty are unrelated.

    """
    left_norm = normalise_text(left)
    right_norm = normalise_text(right)
    if not left_norm and not right_norm:
        return 1.0
    if not left_norm or not right_norm:
        return 0.0

    left_tokens = frozenset(left_norm.split())
    right_tokens = frozenset(right_norm.split())
    token_union = left_tokens | right_tokens
    token_score = len(left_tokens & right_tokens) / len(token_union)

    left_grams = _trigrams(left_norm)
    right_grams = _trigrams(right_norm)
    gram_union = left_grams | right_grams
    char_score = len(left_grams & right_grams) / len(gram_union) if gram_union else 0.0
    return round(0.6 * token_score + 0.4 * char_score, 6)


@dataclass(frozen=True)
class ClusterItem:
    """One scored hypothesis considered for near-duplicate grouping."""

    key: str
    text: str
    score: float


@dataclass(frozen=True)
class Cluster:
    """A deterministic near-duplicate group with one chosen representative."""

    cluster_id: str
    members: tuple[str, ...]
    representative: str

    @property
    def size(self) -> int:
        """Return the number of turns in this cluster."""
        return len(self.members)


@dataclass
class _WorkingCluster:
    """Mutable accumulator used while clustering."""

    items: list[ClusterItem] = field(default_factory=list)

    def accepts(self, item: ClusterItem, threshold: float) -> bool:
        """Return whether ``item`` is close enough to every existing member."""
        return all(text_similarity(item.text, member.text) >= threshold for member in self.items)

    def finished(self) -> Cluster:
        """Freeze the group and pick its highest-scoring member."""
        keys = tuple(sorted(item.key for item in self.items))
        digest = hashlib.sha1("\n".join(keys).encode("utf-8")).hexdigest()[:12]
        representative = sorted(self.items, key=lambda item: (-item.score, item.key))[0].key
        return Cluster(cluster_id=f"cl-{digest}", members=keys, representative=representative)


def cluster_turns(items: Sequence[ClusterItem], threshold: float) -> tuple[Cluster, ...]:
    """Group near-duplicate hypotheses with one representative per cluster.

    Args:
        items: Scored hypotheses, normally failure-filtered turns only.
        threshold: Minimum complete-link similarity in ``[0, 1]``.

    Returns:
        Clusters in first-member order, each with a stable ID and highest-score
        representative.

    """
    clusters: list[_WorkingCluster] = []
    for item in sorted(items, key=lambda candidate: candidate.key):
        for cluster in clusters:
            if cluster.accepts(item, threshold):
                cluster.items.append(item)
                break
        else:
            clusters.append(_WorkingCluster(items=[item]))
    return tuple(cluster.finished() for cluster in clusters)
