"""Inter-reviewer agreement tests on synthetic golden-set candidates only."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from services.review_adapter.metrics.agreement import (
    NOT_HUMAN_RELIABILITY,
    SCRIPTED_GOLDEN_REVIEWERS,
    GoldenCandidate,
    SealedGoldenCandidateSource,
    compute_inter_reviewer_agreement,
)
from services.review_adapter.metrics.events import CandidateAnswer
from services.review_adapter.metrics.settings import ReviewMetricsSettings
from tests.unit.review_adapter_metrics.conftest import GOLDEN_DIR
from voiceloop_common import ValidationError


class StaticGoldenSource:
    """In-memory golden candidate source for deterministic agreement tests."""

    def __init__(self, candidates: tuple[GoldenCandidate, ...]) -> None:
        """Freeze candidates for one test."""
        self._candidates = candidates

    def iter_candidates(self) -> Iterator[GoldenCandidate]:
        """Yield the frozen candidates."""
        return iter(self._candidates)


def _candidate(
    candidate_id: str,
    *,
    first_text: str = "same",
    second_text: str = "same",
    first_slots: dict[str, object] | None = None,
    second_slots: dict[str, object] | None = None,
) -> GoldenCandidate:
    """Build one candidate with two scripted synthetic answers."""
    return GoldenCandidate(
        candidate_id=candidate_id,
        first=CandidateAnswer("reviewer-a-synthetic", first_text, first_slots or {}),
        second=CandidateAnswer("reviewer-b-synthetic", second_text, second_slots or {}),
        sealed_agreed=True,
    )


def test_agreement_counts_and_rates() -> None:
    """Agreement is counted per transcript, slots and both together."""
    source = StaticGoldenSource(
        (
            _candidate("call-1:0"),
            _candidate("call-1:1", first_text="blind", second_text="prefilled"),
        )
    )

    metrics = compute_inter_reviewer_agreement(source)

    assert metrics.candidates == 2
    assert metrics.transcript_agreement_count == 1
    assert metrics.slots_agreement_count == 2
    assert metrics.both_agreement_count == 1
    assert metrics.sealed_agreement_count == 2
    assert metrics.transcript_agreement_rate == 0.5
    assert metrics.slots_agreement_rate == 1.0
    assert metrics.both_agreement_rate == 0.5
    assert metrics.reviewers == SCRIPTED_GOLDEN_REVIEWERS
    assert metrics.interpretation == NOT_HUMAN_RELIABILITY
    assert "corrected_transcript" not in metrics.to_dict()


def test_empty_golden_source_has_null_rates() -> None:
    """No candidates means no agreement rate can be reported."""
    metrics = compute_inter_reviewer_agreement(StaticGoldenSource(()))

    assert metrics.candidates == 0
    assert metrics.transcript_agreement_rate is None
    assert metrics.slots_agreement_rate is None


def test_sealed_golden_source_is_verified_and_never_mutated(
    metrics_settings: ReviewMetricsSettings,
) -> None:
    """The committed synthetic seal verifies and its files are unchanged after reading."""
    before = {
        path.name: path.read_bytes()
        for path in Path(GOLDEN_DIR).resolve().iterdir()
        if path.is_file()
    }

    source = SealedGoldenCandidateSource(metrics_settings)
    candidates = tuple(source.iter_candidates())
    metrics = compute_inter_reviewer_agreement(source)

    after = {
        path.name: path.read_bytes()
        for path in Path(GOLDEN_DIR).resolve().iterdir()
        if path.is_file()
    }
    assert before == after
    assert source.manifest.content_hash == metrics_settings.expected_golden_content_hash
    assert len(candidates) == 36
    assert all(candidate.sealed_agreed for candidate in candidates)
    assert metrics.candidates == 36
    assert metrics.both_agreement_rate == 1.0


def test_sealed_source_rejects_wrong_content_hash(
    metrics_settings: ReviewMetricsSettings,
) -> None:
    """A changed expected hash fails closed instead of measuring unknown labels."""
    changed = metrics_settings.model_copy(update={"expected_golden_content_hash": "wrong"})

    with pytest.raises(ValidationError, match="content hash changed"):
        SealedGoldenCandidateSource(changed)


def test_sealed_source_rejects_tampered_labels(
    metrics_settings: ReviewMetricsSettings, tmp_path: Path
) -> None:
    """Editing a copied labels file breaks the seal; committed artifacts stay untouched."""
    original = Path(GOLDEN_DIR).resolve()
    copied = tmp_path / "golden-artifacts"
    shutil.copytree(original, copied)
    labels_path = copied / "golden.v1.labels.json"
    document = json.loads(labels_path.read_text(encoding="utf-8"))
    document["labels"][0]["corrected_transcript"] = "tampered"
    labels_path.write_text(json.dumps(document), encoding="utf-8")
    changed = metrics_settings.model_copy(update={"golden_artifact_dir": str(copied)})

    with pytest.raises(ValidationError, match="labels were modified after sealing"):
        SealedGoldenCandidateSource(changed)
