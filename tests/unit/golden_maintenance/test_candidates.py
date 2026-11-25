"""P3-09.1: synthetic extension candidates are pseudonymous and fail blind."""

from __future__ import annotations

import pytest

from services.datasets.golden.maintenance import (
    ARTIFACT_LABEL,
    FIXTURE_SHA256,
    GoldenMaintenanceSettings,
    build_candidate_set,
    candidate_document,
    canonical_json,
    contains_pii_pattern,
    load_projected_turns,
    verify_fixture_hash,
)
from services.datasets.golden.maintenance import candidates as candidates_module
from voiceloop_common import ValidationError

_CANDIDATE_KEYS = {
    "candidate_id",
    "call_ref",
    "turn_index",
    "audio_kind",
    "sample_rate_hz",
    "slot_present",
    "source_digest",
}


def test_candidate_set_is_whole_call_pseudonymous_and_reproducible(
    settings: GoldenMaintenanceSettings,
) -> None:
    """Whole calls are selected deterministically and only hashed IDs are stored."""
    candidate_set = build_candidate_set(settings)
    assert candidate_set.version == "candidate.v1"
    assert candidate_set.source_sha256 == FIXTURE_SHA256
    assert candidate_set.requested_calls == settings.candidate_calls
    assert len(candidate_set.records) == 12
    assert len({record.call_ref for record in candidate_set.records}) == settings.candidate_calls
    assert candidate_set.call_refs == tuple(sorted(set(candidate_set.call_refs)))

    for record in candidate_set.records:
        assert set(record.to_dict()) == _CANDIDATE_KEYS
        assert record.candidate_id.startswith("cand-")
        assert record.call_ref.startswith("call-")
        assert len(record.candidate_id.removeprefix("cand-")) == 64
        assert len(record.call_ref.removeprefix("call-")) == 64
        assert len(record.source_digest) == 64
        assert record.audio_kind in ("tone", "silence", "noise")

    assert build_candidate_set(settings) == candidate_set


def test_candidate_document_carries_only_labelled_metadata(
    settings: GoldenMaintenanceSettings,
) -> None:
    """The candidate document has the stand-in label and no raw fixture identity."""
    document = candidate_document(build_candidate_set(settings), ())
    text = canonical_json(document)
    assert document["label"] == ARTIFACT_LABEL
    assert document["synthetic"] is True
    assert document["workflow_stand_in"] is True
    assert "call-0001" not in text
    assert "transcript" not in text
    assert "audio_uri" not in text
    assert not contains_pii_pattern(text)


def test_projection_drops_turns_without_training_consent(
    settings: GoldenMaintenanceSettings,
) -> None:
    """Only consented fixture rows are eligible for candidate selection."""
    rows = load_projected_turns(settings)
    unfiltered = load_projected_turns(settings, require_training_consent=False)
    assert rows
    assert len(rows) <= len(unfiltered)
    assert all(row.source_id for row in rows)


def test_too_many_calls_and_changed_fixture_hash_fail_closed(
    settings: GoldenMaintenanceSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selection and hash checks refuse impossible or changed inputs."""
    oversized = GoldenMaintenanceSettings(output_dir=settings.output_dir, candidate_calls=99)
    with pytest.raises(ValidationError, match="exceeds"):
        build_candidate_set(oversized)

    monkeypatch.setattr(candidates_module, "sha256_file", lambda _: "0" * 64)
    with pytest.raises(ValidationError, match="fixture manifest hash"):
        verify_fixture_hash(settings)


def test_candidate_set_round_trips_without_transcript_fields(
    settings: GoldenMaintenanceSettings,
) -> None:
    """A candidate set rebuilds from its restricted field allowlist."""
    from services.datasets.golden.maintenance import CandidateSet

    candidate_set = build_candidate_set(settings)
    rebuilt = CandidateSet.from_dict(candidate_set.to_dict())
    assert rebuilt == candidate_set
