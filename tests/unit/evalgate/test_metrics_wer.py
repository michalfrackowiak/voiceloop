"""Hand-computed WER and Levenshtein alignment cases."""

from __future__ import annotations

import pytest

from services.evalgate.metrics import AlignmentCounts, levenshtein_counts, tokenize, word_error_rate


def test_tokenize_is_casefolded_and_punctuation_free() -> None:
    """Tokens are lower-cased words, not punctuation."""
    assert tokenize("The Cat, sat!") == ("the", "cat", "sat")
    assert tokenize("") == ()


def test_levenshtein_counts_substitution() -> None:
    """One wrong word is one substitution."""
    assert levenshtein_counts(("the", "cat", "sat"), ("the", "bat", "sat")) == AlignmentCounts(
        substitutions=1, deletions=0, insertions=0
    )


def test_levenshtein_counts_deletion() -> None:
    """A missing word is one deletion."""
    assert levenshtein_counts(("a", "b", "c"), ("a", "c")) == AlignmentCounts(
        substitutions=0, deletions=1, insertions=0
    )


def test_levenshtein_counts_insertion() -> None:
    """Two extra words are two insertions."""
    assert levenshtein_counts(("a", "b"), ("x", "a", "b", "c")) == AlignmentCounts(
        substitutions=0, deletions=0, insertions=2
    )


def test_wer_hand_computed() -> None:
    """WER is edit count divided by the reference token count."""
    assert word_error_rate("the cat sat", "the bat sat") == pytest.approx(1 / 3)
    assert word_error_rate("a b c", "a c") == pytest.approx(1 / 3)
    assert word_error_rate("a b", "x a b c") == pytest.approx(1.0)
    assert word_error_rate("a b", "a b") == 0.0


def test_wer_silence_hallucination_uses_virtual_word() -> None:
    """Empty references use one virtual word, so hallucinations score above zero."""
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "thank you for calling") == 4.0
