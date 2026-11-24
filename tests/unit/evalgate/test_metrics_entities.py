"""Hand-computed entity extraction and error cases."""

from __future__ import annotations

import pytest

from services.evalgate.metrics import (
    EntityPatterns,
    EvaluationError,
    entity_error,
    extract_entities,
)


def test_extract_entities_default_patterns() -> None:
    """Names, dates and numbers are extracted deterministically and normalised."""
    text = "Check in on the twelfth of December for Ada Fixture, two guests."
    extracted = extract_entities(text)
    assert extracted["names"]["ada fixture"] == 1
    assert extracted["names"]["check in"] == 0
    assert extracted["dates"]["twelfth of december"] == 1
    assert extracted["dates"]["december"] == 0  # non-overlapping with the ordinal match
    assert extracted["numbers"]["two"] == 1
    assert extracted["numbers"]["twelfth"] == 0


def test_entity_error_hand_computed_name_substitution() -> None:
    """A wrong surname is a full name error and no other entity error."""
    result = entity_error("Mr. Smith.", "Mr. Smythe.")
    assert result.by_type["names"] == 1.0
    assert result.by_type["dates"] == 0.0
    assert result.by_type["numbers"] == 0.0
    assert result.aggregate == 1.0


def test_entity_error_hand_computed_number_and_aggregate() -> None:
    """One wrong number out of two reference numbers gives aggregate 2/3."""
    result = entity_error(
        "Ada Fixture, two guests and three nights.",
        "Ada Fixture, two guests and four nights.",
    )
    assert result.by_type["names"] == 0.0
    assert result.by_type["numbers"] == 1.0
    assert result.aggregate == pytest.approx(2 / 3)


def test_entity_error_counts_extra_entities_as_errors() -> None:
    """A hallucinated entity on a no-entity reference is a full error."""
    result = entity_error("two guests", "two guests, Ada Fixture")
    assert result.by_type["names"] == 1.0
    assert result.by_type["numbers"] == 0.0
    assert result.aggregate == 1.0


def test_entity_error_no_entities_is_zero() -> None:
    """Silence with no hypothesis entities has no entity error."""
    result = entity_error("", "")
    assert result.by_type == {"names": 0.0, "dates": 0.0, "numbers": 0.0}
    assert result.aggregate == 0.0


def test_entity_pattern_overrides_are_configurable_and_validated() -> None:
    """Overrides replace one type and reject unknown keys or bad regexes."""
    patterns = EntityPatterns().with_overrides({"names": r"\bZed\b"})
    extracted = extract_entities("Zed and Ada Fixture", patterns)
    assert extracted["names"]["zed"] == 1
    assert "ada fixture" not in extracted["names"]
    assert extract_entities("2026-12-12", patterns)["dates"]["2026 12 12"] == 1

    with pytest.raises(EvaluationError, match="unknown entity pattern keys"):
        EntityPatterns().with_overrides({"bogus": "x"})
    with pytest.raises(EvaluationError, match="invalid names entity pattern"):
        EntityPatterns(names="(").compiled()
