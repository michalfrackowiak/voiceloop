"""Inter-reviewer agreement on scripted synthetic golden-set candidates only.

The sealed golden stand-in is read through the committed P1-06 verifier without
audit context. It is never written to, trained on or tuned on. Two pseudonymous
scripted reviewers replay the sealed labels; with ``simulated_synthetic_labels``
as the only supported mode, no human inter-reviewer reliability is claimed.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from services.datasets.golden.labelling import LabelRecord
from services.datasets.golden.seal import load_labels, verify_seal
from voiceloop_common import ValidationError

from .anchoring import answers_agree
from .events import CandidateAnswer, resolve_path
from .settings import ReviewMetricsSettings

SCRIPTED_GOLDEN_REVIEWERS = "scripted_synthetic_reviewers"
"""Fixed report string describing who produced the answers."""

GOLDEN_CANDIDATE_SOURCE = "sealed_synthetic_golden_candidates"
"""Fixed report string describing the candidate provenance."""

NOT_HUMAN_RELIABILITY = "not_human_inter_reviewer_reliability"
"""Fixed report string: these rates are a fixture rehearsal, not human evidence."""


@dataclass(frozen=True)
class GoldenCandidate:
    """One sealed golden candidate with two scripted synthetic reviewer answers."""

    candidate_id: str
    first: CandidateAnswer
    second: CandidateAnswer
    sealed_agreed: bool


class GoldenCandidateSource(Protocol):
    """Injectable source of golden candidates; the protocol is structural."""

    def iter_candidates(self) -> Iterator[GoldenCandidate]:
        """Yield golden candidates in a stable order."""
        ...


class SealedGoldenCandidateSource:
    """Read-only source of scripted answers derived from the sealed v1 stand-in."""

    def __init__(self, settings: ReviewMetricsSettings) -> None:
        """Verify the seal and freeze synthetic candidate answers in memory.

        Args:
            settings: Typed fixture-only settings with the expected seal hash.

        Raises:
            ValidationError: If the seal, hash, synthetic flags or label mode do
                not match the committed synthetic stand-in.

        """
        self._settings = settings
        artifact_dir = resolve_path(settings.golden_artifact_dir)
        verification = verify_seal(artifact_dir, settings.golden_version)
        verification.assert_ok()
        manifest = verification.manifest
        if manifest is None:  # pragma: no cover - guaranteed by a passing seal
            raise ValidationError("verified golden stand-in has no manifest")
        if manifest.content_hash != settings.expected_golden_content_hash:
            raise ValidationError(
                "golden stand-in content hash changed; refusing to measure agreement: "
                f"{manifest.content_hash!r}"
            )
        if not manifest.synthetic or not manifest.workflow_stand_in:
            raise ValidationError(
                "golden stand-in is no longer labelled synthetic/workflow stand-in"
            )
        if manifest.review_mode != "simulated_synthetic_labels":
            raise ValidationError("golden stand-in review mode is not the expected synthetic mode")
        self.manifest = manifest
        labels = load_labels(
            artifact_dir / manifest.labels_path, expected_version=settings.golden_version
        )
        self._candidates = tuple(self._candidate(label) for label in labels)

    def iter_candidates(self) -> Iterator[GoldenCandidate]:
        """Iterate over the frozen scripted golden candidates."""
        return iter(self._candidates)

    @staticmethod
    def _candidate(label: LabelRecord) -> GoldenCandidate:
        """Build two scripted answers from one sealed synthetic label.

        A label whose sealed ``agreed`` flag is false cannot be scripted without
        inventing a disagreement, so it fails closed. No sealed field is edited.
        """
        if not label.agreed:
            raise ValidationError("cannot script a synthetic disagreement from sealed labels")
        answer = CandidateAnswer(
            reviewer_id=label.first_reviewer_id,
            corrected_transcript=label.corrected_transcript,
            slots=dict(label.slots),
        )
        second = CandidateAnswer(
            reviewer_id=label.second_reviewer_id,
            corrected_transcript=label.corrected_transcript,
            slots=dict(label.slots),
        )
        return GoldenCandidate(
            candidate_id=label.source_id,
            first=answer,
            second=second,
            sealed_agreed=label.agreed,
        )


@dataclass(frozen=True)
class AgreementMetrics:
    """Counts and rates for scripted inter-reviewer agreement on golden candidates."""

    candidates: int
    transcript_agreement_count: int
    slots_agreement_count: int
    both_agreement_count: int
    sealed_agreement_count: int
    transcript_agreement_rate: float | None
    slots_agreement_rate: float | None
    both_agreement_rate: float | None
    reviewers: str = SCRIPTED_GOLDEN_REVIEWERS
    source: str = GOLDEN_CANDIDATE_SOURCE
    interpretation: str = NOT_HUMAN_RELIABILITY

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable view containing no label text."""
        return {
            "candidates": self.candidates,
            "transcript_agreement_count": self.transcript_agreement_count,
            "slots_agreement_count": self.slots_agreement_count,
            "both_agreement_count": self.both_agreement_count,
            "sealed_agreement_count": self.sealed_agreement_count,
            "transcript_agreement_rate": self.transcript_agreement_rate,
            "slots_agreement_rate": self.slots_agreement_rate,
            "both_agreement_rate": self.both_agreement_rate,
            "reviewers": self.reviewers,
            "source": self.source,
            "interpretation": self.interpretation,
        }


def _rate(count: int, total: int) -> float | None:
    """Return ``count / total`` or ``None`` when there are no candidates."""
    return count / total if total else None


def compute_inter_reviewer_agreement(source: GoldenCandidateSource) -> AgreementMetrics:
    """Compare two scripted synthetic answers for each sealed golden candidate.

    Args:
        source: Read-only golden candidate source.

    Returns:
        Agreement counts and rates; no text or human-reliability claim.

    """
    candidates: tuple[GoldenCandidate, ...] = tuple(source.iter_candidates())
    transcript_agreements = sum(
        candidate.first.corrected_transcript == candidate.second.corrected_transcript
        for candidate in candidates
    )
    slot_agreements = sum(
        answers_agree(candidate.first.slots, candidate.second.slots) for candidate in candidates
    )
    both_agreements = sum(
        candidate.first.corrected_transcript == candidate.second.corrected_transcript
        and answers_agree(candidate.first.slots, candidate.second.slots)
        for candidate in candidates
    )
    total = len(candidates)
    return AgreementMetrics(
        candidates=total,
        transcript_agreement_count=transcript_agreements,
        slots_agreement_count=slot_agreements,
        both_agreement_count=both_agreements,
        sealed_agreement_count=sum(1 for candidate in candidates if candidate.sealed_agreed),
        transcript_agreement_rate=_rate(transcript_agreements, total),
        slots_agreement_rate=_rate(slot_agreements, total),
        both_agreement_rate=_rate(both_agreements, total),
    )
