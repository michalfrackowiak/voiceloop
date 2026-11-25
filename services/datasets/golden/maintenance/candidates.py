"""Read-only projection of synthetic fixtures into extension candidate sets.

Only fixture metadata needed for a failure-blind, pseudonymous candidate set is
read. Transcripts, slot values, notes, audio paths, audio bytes and triage labels
are never copied into a candidate document, audit entry or report. Candidates
use whole calls and SHA-256 pseudonyms, so no speaker can be split and no raw
guest identifier reaches a maintained artifact.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from voiceloop_common import ValidationError

from .settings import (
    ARTIFACT_LABEL,
    GoldenMaintenanceSettings,
    assert_safe_text,
    assert_valid_hash,
    canonical_json,
    pseudonymous_id,
    resolve_path,
    sha256_file,
    sha256_text,
    utc_timestamp,
)

AudioKind = Literal["tone", "silence", "noise"]
_ALLOWED_AUDIO_KINDS: frozenset[str] = frozenset({"tone", "silence", "noise"})
_ALLOWED_SAMPLE_RATES: frozenset[int] = frozenset({8000, 16000})


@dataclass(frozen=True)
class FixtureProjection:
    """In-memory-only metadata for one synthetic fixture turn.

    ``source_id`` and ``call_id`` are pseudonymous fixture labels used only for
    grouping and reproducibility. This object is deliberately never serialised:
    candidate artifacts contain hashed references instead.
    """

    source_id: str
    call_id: str
    turn_index: int
    audio_kind: AudioKind
    sample_rate_hz: int
    slot_present: bool


@dataclass(frozen=True)
class CandidateRecord:
    """One pseudonymous extension candidate; no label or transcript content."""

    candidate_id: str
    call_ref: str
    turn_index: int
    audio_kind: AudioKind
    sample_rate_hz: int
    slot_present: bool
    source_digest: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of the candidate."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CandidateRecord:
        """Rebuild and validate a candidate from a maintained document."""
        try:
            record = cls(
                candidate_id=str(data["candidate_id"]),
                call_ref=str(data["call_ref"]),
                turn_index=int(data["turn_index"]),
                audio_kind=cast(AudioKind, str(data["audio_kind"])),
                sample_rate_hz=int(data["sample_rate_hz"]),
                slot_present=bool(data["slot_present"]),
                source_digest=str(data["source_digest"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"malformed candidate record: {exc}") from exc
        _validate_pseudonym(record.candidate_id, "cand", "candidate_id")
        _validate_pseudonym(record.call_ref, "call", "call_ref")
        assert_valid_hash(record.source_digest, "source_digest")
        if record.audio_kind not in _ALLOWED_AUDIO_KINDS:
            raise ValidationError("candidate audio_kind is outside the synthetic allowlist")
        if record.sample_rate_hz not in _ALLOWED_SAMPLE_RATES:
            raise ValidationError("candidate sample rate is outside the synthetic allowlist")
        if record.turn_index < 0:
            raise ValidationError("candidate turn_index must be non-negative")
        return record


@dataclass(frozen=True)
class ProjectionSelection:
    """Deterministic whole-call selection over projected fixture turns."""

    rows: tuple[FixtureProjection, ...]
    selected_call_ids: tuple[str, ...]
    skipped_call_ids: tuple[str, ...]
    seed: int
    requested_calls: int


@dataclass(frozen=True)
class CandidateSet:
    """A separate synthetic candidate set, never merged into the sealed set."""

    version: str
    records: tuple[CandidateRecord, ...]
    source_sha256: str
    selection_seed: int
    requested_calls: int
    call_refs: tuple[str, ...]
    skipped_call_refs: tuple[str, ...]
    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of the candidate set."""
        return {
            "version": self.version,
            "records": [record.to_dict() for record in self.records],
            "source_sha256": self.source_sha256,
            "selection_seed": self.selection_seed,
            "requested_calls": self.requested_calls,
            "call_refs": list(self.call_refs),
            "skipped_call_refs": list(self.skipped_call_refs),
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CandidateSet:
        """Rebuild and validate a candidate set from a maintained document."""
        try:
            records = tuple(CandidateRecord.from_dict(item) for item in data["records"])
            result = cls(
                version=str(data["version"]),
                records=records,
                source_sha256=str(data["source_sha256"]),
                selection_seed=int(data["selection_seed"]),
                requested_calls=int(data["requested_calls"]),
                call_refs=tuple(str(item) for item in data["call_refs"]),
                skipped_call_refs=tuple(str(item) for item in data["skipped_call_refs"]),
                content_hash=str(data["content_hash"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"malformed candidate set: {exc}") from exc
        assert_valid_hash(result.source_sha256, "source_sha256")
        assert_valid_hash(result.content_hash, "content_hash")
        if not records:
            raise ValidationError("candidate set must contain at least one record")
        if candidate_content_hash(records) != result.content_hash:
            raise ValidationError("candidate content hash does not match records")
        return result


def _validate_pseudonym(value: str, prefix: str, field: str) -> None:
    """Require a pseudonym of the form ``<prefix>-<sha256 hex>``."""
    expected = f"{prefix}-"
    if not value.startswith(expected):
        raise ValidationError(f"{field} has the wrong pseudonym prefix")
    assert_valid_hash(value[len(expected) :], field)


def verify_fixture_hash(settings: GoldenMaintenanceSettings) -> str:
    """Return the fixture SHA-256 and fail closed if it differs from the pin.

    Args:
        settings: Typed maintenance settings with the pinned fixture hash.

    Returns:
        The verified lowercase SHA-256 hex digest of the fixture file.

    Raises:
        ValidationError: If the read-only fixture manifest changed or is unreadable.

    """
    digest = sha256_file(settings.fixtures_path)
    if digest != settings.expected_fixture_sha256:
        raise ValidationError("fixture manifest hash no longer matches the committed pin")
    return digest


def load_projected_turns(
    settings: GoldenMaintenanceSettings,
    *,
    require_training_consent: bool = True,
    path: str | Path | None = None,
) -> tuple[FixtureProjection, ...]:
    """Project the read-only fixture manifest onto failure-blind metadata.

    Args:
        settings: Typed maintenance settings.
        require_training_consent: Drop rows without training consent, as required
            before any candidate can enter a dataset (context.md section 2 #8).
        path: Optional fixture path override for isolated tests.

    Returns:
        Projected rows sorted by ``(call_id, turn_index)``.

    Raises:
        ValidationError: If the fixture hash pin fails or a row is malformed.

    """
    verify_fixture_hash(settings)
    fixture_path = resolve_path(path or settings.fixtures_path)
    try:
        lines = fixture_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValidationError("cannot read the synthetic fixture manifest") from exc

    rows: list[FixtureProjection] = []
    seen: set[tuple[str, int]] = set()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"fixture line {number} is not valid JSON") from exc
        if not isinstance(decoded, dict):
            raise ValidationError(f"fixture line {number} must be an object")
        if decoded.get("synthetic") is not True:
            raise ValidationError(f"fixture line {number} is not tagged synthetic")
        consent = decoded.get("consent")
        if not isinstance(consent, dict):
            raise ValidationError(f"fixture line {number} has no consent object")
        if require_training_consent and consent.get("training") is not True:
            continue
        audio_kind = decoded.get("audio_kind")
        if audio_kind not in _ALLOWED_AUDIO_KINDS:
            raise ValidationError(f"fixture line {number} has an unsupported audio kind")
        sample_rate = decoded.get("sample_rate_hz")
        if sample_rate not in _ALLOWED_SAMPLE_RATES:
            raise ValidationError(f"fixture line {number} has an unsupported sample rate")
        call_id = decoded.get("call_id")
        turn_index = decoded.get("turn_index")
        slots = decoded.get("expected_slots")
        if (
            not isinstance(call_id, str)
            or not call_id.strip()
            or not isinstance(turn_index, int)
            or turn_index < 0
            or not isinstance(slots, dict)
        ):
            raise ValidationError(f"fixture line {number} has invalid identity or slot shape")
        key = (call_id, turn_index)
        if key in seen:
            raise ValidationError(f"fixture line {number} duplicates a source turn")
        seen.add(key)
        source_id = f"{call_id}:{turn_index}"
        assert_safe_text(source_id, "source_id")
        rows.append(
            FixtureProjection(
                source_id=source_id,
                call_id=call_id,
                turn_index=turn_index,
                audio_kind=audio_kind,
                sample_rate_hz=sample_rate,
                slot_present=bool(slots),
            )
        )
    if not rows:
        raise ValidationError("fixture projection is empty")
    return tuple(sorted(rows, key=lambda row: (row.call_id, row.turn_index)))


def select_projected_calls(
    rows: Sequence[FixtureProjection], *, seed: int, max_calls: int
) -> ProjectionSelection:
    """Select whole calls with a seeded RNG; never split a call.

    Args:
        rows: Projected fixture turns.
        seed: Seed for the deterministic call-order shuffle.
        max_calls: Number of whole calls to select.

    Returns:
        Selected rows plus pseudonym-safe audit metadata about the selection.

    Raises:
        ValidationError: If ``max_calls`` exceeds the available whole calls.

    """
    if max_calls < 1:
        raise ValidationError("max_calls must be positive")
    grouped: dict[str, list[FixtureProjection]] = {}
    for row in rows:
        grouped.setdefault(row.call_id, []).append(row)
    order = sorted(grouped)
    random.Random(seed).shuffle(order)
    if max_calls > len(order):
        raise ValidationError("max_calls exceeds the number of whole fixture calls")
    chosen = tuple(sorted(order[:max_calls]))
    skipped = tuple(sorted(order[max_calls:]))
    selected = tuple(
        sorted(
            (row for row in rows if row.call_id in chosen),
            key=lambda row: (row.call_id, row.turn_index),
        )
    )
    return ProjectionSelection(
        rows=selected,
        selected_call_ids=chosen,
        skipped_call_ids=skipped,
        seed=seed,
        requested_calls=max_calls,
    )


def candidate_content_hash(records: Sequence[CandidateRecord]) -> str:
    """Hash every field of every candidate in canonical sorted order."""
    ordered = sorted(records, key=lambda record: record.candidate_id)
    return sha256_text(canonical_json({"candidates": [record.to_dict() for record in ordered]}))


def build_candidate_set(
    settings: GoldenMaintenanceSettings, *, rows: Sequence[FixtureProjection] | None = None
) -> CandidateSet:
    """Build a fresh, separately namespaced synthetic extension candidate set.

    Args:
        settings: Typed maintenance settings.
        rows: Optional projected pool; defaults to consented fixture turns.

    Returns:
        A versioned candidate set with a content hash and SHA-256 pseudonyms.

    """
    pool = tuple(rows) if rows is not None else load_projected_turns(settings)
    selection = select_projected_calls(
        pool, seed=settings.candidate_seed, max_calls=settings.candidate_calls
    )
    records = tuple(
        CandidateRecord(
            candidate_id=pseudonymous_id("cand", row.source_id),
            call_ref=pseudonymous_id("call", row.call_id),
            turn_index=row.turn_index,
            audio_kind=row.audio_kind,
            sample_rate_hz=row.sample_rate_hz,
            slot_present=row.slot_present,
            source_digest=sha256_text(row.source_id),
        )
        for row in selection.rows
    )
    return CandidateSet(
        version=settings.candidate_version,
        records=records,
        source_sha256=verify_fixture_hash(settings),
        selection_seed=settings.candidate_seed,
        requested_calls=settings.candidate_calls,
        call_refs=tuple(sorted({record.call_ref for record in records})),
        skipped_call_refs=tuple(
            sorted({pseudonymous_id("call", call_id) for call_id in selection.skipped_call_ids})
        ),
        content_hash=candidate_content_hash(records),
    )


def candidate_document(
    candidate_set: CandidateSet, reviews: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Assemble the internal, PII-free candidate document for one version."""
    return {
        "version": candidate_set.version,
        "label": ARTIFACT_LABEL,
        "synthetic": True,
        "workflow_stand_in": True,
        "candidates": candidate_set.to_dict(),
        "reviews": [dict(review) for review in reviews],
        "written_at": utc_timestamp(),
    }


def load_candidate_document(
    path: str | Path,
) -> tuple[CandidateSet, tuple[dict[str, Any], ...]]:
    """Read a maintained candidate document and validate its set and reviews."""
    try:
        document = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("cannot read the candidate document") from exc
    if not isinstance(document, dict) or document.get("label") != ARTIFACT_LABEL:
        raise ValidationError("candidate document has the wrong synthetic stand-in label")
    reviews = document.get("reviews")
    if not isinstance(reviews, list) or not all(isinstance(item, dict) for item in reviews):
        raise ValidationError("candidate document has malformed reviews")
    return CandidateSet.from_dict(document["candidates"]), tuple(reviews)
