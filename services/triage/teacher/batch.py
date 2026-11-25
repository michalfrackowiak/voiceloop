"""Offline batch teacher transcription over read-only committed fixtures.

This module contains no model, GPU, network or audio-decoding code. A
:class:`TeacherTranscriber` is injected; the built-in
:class:`DeterministicFixtureTeacher` is a clearly labelled fixture stand-in that
derives a pseudo-transcript from recorded turn-event metadata, never from a
fixture label or the sealed golden set. Only text references (audio URIs) enter
artifacts; audio bytes are never read or written here.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

from voiceloop_common import ValidationError

from .settings import ARTIFACT_SCHEMA_VERSION, HASH_ALGORITHM, RUN_LABEL, TeacherSettings

REPO_ROOT = Path(__file__).resolve().parents[3]
_TEACHER_RECORD_KIND = "voiceloop.teacher.transcript"
_WORD_PATTERN = re.compile(r"\S+")
_PUNCTUATION_PATTERN = re.compile(r"[^a-z0-9']")


def resolve_repo_path(path: str | Path, *, root: Path | None = None) -> Path:
    """Resolve a repo-relative or absolute path against the repository root.

    Args:
        path: Path declared in settings or a fixture manifest.
        root: Base directory; defaults to the repository root.

    Returns:
        The resolved path (not required to exist).

    """
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (root or REPO_ROOT) / candidate


def canonical_hash(payload: Mapping[str, Any], *, algorithm: str = HASH_ALGORITHM) -> str:
    """Hash a JSON-like payload in stable canonical form.

    Args:
        payload: String-keyed, JSON-serialisable mapping. Never audio bytes.
        algorithm: Hashlib algorithm name; only ``sha256`` is used by default.

    Returns:
        Lower-case hexadecimal digest.

    Raises:
        ValidationError: If the algorithm is unavailable or payload is not JSON.

    """
    try:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"cannot hash non-JSON payload: {exc}") from exc
    try:
        digest = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValidationError(f"unsupported hash algorithm: {algorithm}") from exc
    digest.update(encoded.encode("utf-8"))
    return digest.hexdigest()


def fixture_inputs_digest(
    turns_path: str | Path = "tests/fixtures/turns.jsonl", *, root: Path | None = None
) -> str:
    """Hash the committed labels manifest and every referenced event JSON file.

    The digest covers file paths and raw UTF-8 bytes only. WAV audio files are
    deliberately excluded: this package must never read or copy audio bytes.

    Args:
        turns_path: Read-only fixture labels manifest.
        root: Base directory for relative paths; defaults to the repository root.

    Returns:
        Hexadecimal SHA-256 digest over the ordered fixture text inputs.

    Raises:
        ValidationError: If the manifest or an event file is unreadable.

    """
    manifest = resolve_repo_path(turns_path, root=root)
    base = root or REPO_ROOT
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValidationError(f"cannot read fixtures: {manifest}") from exc
    digest = hashlib.sha256()
    digest.update(manifest.relative_to(base).as_posix().encode("utf-8"))
    digest.update(b"\0")
    digest.update(manifest.read_bytes())
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"invalid fixture JSON at {manifest}") from exc
        if not isinstance(row, Mapping) or not isinstance(row.get("event_path"), str):
            raise ValidationError(f"fixture row needs an event_path at {manifest}")
        event_path = resolve_repo_path(str(row["event_path"]), root=root)
        try:
            payload = event_path.read_bytes()
        except OSError as exc:
            raise ValidationError(f"cannot read fixture event: {event_path}") from exc
        digest.update(event_path.relative_to(base).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class TurnRecord:
    """An immutable, label-free view of one committed turn event."""

    event_id: str
    call_id: str
    turn_index: int
    audio_uri: str
    event: Mapping[str, Any]

    @classmethod
    def from_event(cls, event: Mapping[str, Any]) -> TurnRecord:
        """Build a record and validate lineage identifiers.

        Args:
            event: A dict conforming to ``turn_event`` 1.0.0.

        Returns:
            The immutable view.

        Raises:
            ValidationError: If identifiers or the audio reference are missing.

        """
        event_id = event.get("event_id")
        call_id = event.get("call_id")
        turn_index = event.get("turn_index")
        audio = event.get("audio")
        audio_uri = audio.get("uri") if isinstance(audio, Mapping) else None
        if not isinstance(event_id, str) or not event_id:
            raise ValidationError("turn event needs a non-empty event_id")
        if not isinstance(call_id, str) or not call_id:
            raise ValidationError("turn event needs a non-empty call_id")
        if isinstance(turn_index, bool) or not isinstance(turn_index, int) or turn_index < 0:
            raise ValidationError("turn event needs a non-negative integer turn_index")
        if not isinstance(audio_uri, str) or not audio_uri:
            raise ValidationError("turn event needs a non-empty audio.uri")
        return cls(
            event_id=event_id,
            call_id=call_id,
            turn_index=turn_index,
            audio_uri=audio_uri,
            event=event,
        )

    @property
    def source_id(self) -> str:
        """Return the stable ``call_id:turn_index`` lineage key."""
        return f"{self.call_id}:{self.turn_index}"


@runtime_checkable
class TurnSource(Protocol):
    """Injectable source of eligible turn events; never reads labels."""

    def iter_turns(self) -> Iterator[TurnRecord]:
        """Yield turn records in a stable order."""
        ...


class InMemoryTurnSource:
    """Deterministic in-memory fake :class:`TurnSource` for tests."""

    def __init__(self, records: Sequence[TurnRecord | Mapping[str, Any]]) -> None:
        """Validate and index caller-supplied events or records.

        Args:
            records: Turn-event mappings or already validated records.

        """
        self._records = tuple(
            record if isinstance(record, TurnRecord) else TurnRecord.from_event(record)
            for record in records
        )

    def iter_turns(self) -> Iterator[TurnRecord]:
        """Yield the indexed records in input order."""
        return iter(self._records)


class FixtureTurnSource:
    """Read-only source for the committed synthetic fixture manifest.

    The manifest is used only to locate event files and to check call/turn
    agreement. No ``expected_transcript``, ``expected_slots``, ``failure_type``
    or audio payload is read into batch processing.
    """

    def __init__(
        self, turns_path: str | Path = "tests/fixtures/turns.jsonl", *, root: Path | None = None
    ) -> None:
        """Set the manifest path.

        Args:
            turns_path: Repo-relative or absolute labels manifest path.
            root: Base directory for relative paths; defaults to the repository root.

        """
        self._root = root or REPO_ROOT
        self._path = resolve_repo_path(turns_path, root=root)

    def iter_turns(self) -> Iterator[TurnRecord]:
        """Load and yield the referenced fixture events.

        Yields:
            One :class:`TurnRecord` per manifest row.

        Raises:
            ValidationError: If the manifest or an event is invalid or misaligned.

        """
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValidationError(f"cannot read fixtures: {self._path}") from exc
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"invalid fixture JSON at line {number}") from exc
            if not isinstance(row, Mapping) or not isinstance(row.get("event_path"), str):
                raise ValidationError(f"fixture line {number} needs an event_path")
            if row.get("synthetic") is not True:
                raise ValidationError(f"fixture line {number} is not labelled synthetic")
            event_path = resolve_repo_path(str(row["event_path"]), root=self._root)
            try:
                event = json.loads(event_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValidationError(f"cannot read fixture event: {event_path}") from exc
            if not isinstance(event, Mapping):
                raise ValidationError(f"fixture event is not an object: {event_path}")
            record = TurnRecord.from_event(cast(Mapping[str, Any], event))
            if row.get("call_id") not in (None, record.call_id):
                raise ValidationError(f"call_id mismatch at fixture line {number}")
            if row.get("turn_index") not in (None, record.turn_index):
                raise ValidationError(f"turn_index mismatch at fixture line {number}")
            yield record


@dataclass(frozen=True)
class TeacherTranscript:
    """One fake or real teacher transcript plus its content hash."""

    model_id: str
    text: str
    content_hash: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable transcript view (no audio bytes)."""
        return {
            "model_id": self.model_id,
            "text": self.text,
            "content_hash": self.content_hash,
            "reason": self.reason,
        }


@runtime_checkable
class TeacherTranscriber(Protocol):
    """Injectable offline teacher adapter; never used in the call path."""

    @property
    def model_id(self) -> str:
        """Stable identifier recorded with every transcript."""
        ...

    def transcribe(self, record: TurnRecord) -> TeacherTranscript:
        """Return a teacher transcript for one turn's audio reference.

        Real adapters should resolve ``record.audio_uri`` at its native sample
        rate (including 8 kHz), use the exact deployed decoding settings and
        return the deployed artifact's output. This package imports no such
        runtime; see the README's real-adapter seam.
        """
        ...


def _number(value: Any) -> float | None:
    """Return a finite numeric value, or ``None`` for missing/non-numeric input."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _replace_low_confidence_token(text: str, word: str) -> str:
    """Replace the first punctuation-insensitive token match with ``[unclear]``."""
    target = _PUNCTUATION_PATTERN.sub("", word.casefold())
    tokens = _WORD_PATTERN.findall(text)
    for index, token in enumerate(tokens):
        if _PUNCTUATION_PATTERN.sub("", token.casefold()) == target:
            tokens[index] = "[unclear]"
            break
    else:
        tokens.append("[unclear]")
    return " ".join(tokens)


class DeterministicFixtureTeacher:
    """Fake teacher that derives stand-in transcripts from event metadata.

    The WAV fixtures are tone/noise/silence carriers with no intelligible
    speech, so this fake cannot decode them. Instead it deterministically:

    * returns an empty transcript when ``no_speech_prob`` is at or above the
      configured ceiling (a stand-in for a teacher hearing no speech);
    * replaces the lowest-confidence word with ``[unclear]`` when that word is
      below the configured ceiling (a stand-in for a teacher hearing it
      differently);
    * otherwise replays the production text verbatim.

    No fixture label, expected transcript, golden label or audio byte is used.
    """

    def __init__(
        self,
        model_id: str = "fake://teacher/fixture-standin-v1",
        *,
        low_confidence_ceiling: float = 0.60,
        no_speech_prob_ceiling: float = 0.50,
        algorithm: str = HASH_ALGORITHM,
    ) -> None:
        """Configure the deterministic stand-in.

        Args:
            model_id: Explicitly fake model identifier recorded in artifacts.
            low_confidence_ceiling: Word confidence below which a token is masked.
            no_speech_prob_ceiling: No-speech probability at or above which the
                teacher returns an empty transcript.
            algorithm: Hash algorithm for transcript content hashes.

        """
        self._model_id = model_id
        self._low_confidence_ceiling = low_confidence_ceiling
        self._no_speech_prob_ceiling = no_speech_prob_ceiling
        self._algorithm = algorithm

    @property
    def model_id(self) -> str:
        """Return the explicitly fake model identifier."""
        return self._model_id

    def transcribe(self, record: TurnRecord) -> TeacherTranscript:
        """Derive one deterministic stand-in transcript from event metadata."""
        stt = record.event.get("stt")
        stt = stt if isinstance(stt, Mapping) else {}
        text = stt.get("text", "")
        text = text if isinstance(text, str) else ""
        stats = stt.get("decoder_stats")
        stats = stats if isinstance(stats, Mapping) else {}
        no_speech = _number(stats.get("no_speech_prob"))
        if no_speech is not None and no_speech >= self._no_speech_prob_ceiling:
            result = ""
            reason = (
                "fixture stand-in: no_speech_prob "
                f"{no_speech:.2f} >= {self._no_speech_prob_ceiling:.2f}; empty teacher transcript"
            )
        else:
            words = stt.get("words")
            lowest: Mapping[str, Any] | None = None
            lowest_conf = float("inf")
            if isinstance(words, (list, tuple)):
                for word in words:
                    if not isinstance(word, Mapping):
                        continue
                    confidence = _number(word.get("conf"))
                    if confidence is not None and confidence < lowest_conf:
                        lowest_conf = confidence
                        lowest = word
            if lowest is not None and lowest_conf < self._low_confidence_ceiling:
                masked = str(lowest.get("w", ""))
                result = _replace_low_confidence_token(text, masked)
                reason = (
                    "fixture stand-in: lowest-confidence word "
                    f"{masked!r} ({lowest_conf:.2f}) replaced with [unclear]"
                )
            else:
                result = " ".join(text.split())
                reason = "fixture stand-in: production text replayed; no audio decoding"

        payload = {
            "kind": _TEACHER_RECORD_KIND,
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "event_id": record.event_id,
            "call_id": record.call_id,
            "turn_index": record.turn_index,
            "audio_uri": record.audio_uri,
            "model_id": self._model_id,
            "text": result,
        }
        return TeacherTranscript(
            model_id=self._model_id,
            text=result,
            content_hash=canonical_hash(payload, algorithm=self._algorithm),
            reason=reason,
        )


@dataclass(frozen=True)
class TeacherBatchRecord:
    """One teacher transcript joined to its production text reference."""

    event_id: str
    call_id: str
    turn_index: int
    audio_uri: str
    production_text: str
    teacher_text: str
    production_text_hash: str
    teacher_text_hash: str
    teacher_model_id: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON view: text references and hashes, never audio bytes."""
        return {
            "event_id": self.event_id,
            "call_id": self.call_id,
            "turn_index": self.turn_index,
            "audio_uri": self.audio_uri,
            "production_text": self.production_text,
            "teacher_text": self.teacher_text,
            "production_text_hash": self.production_text_hash,
            "teacher_text_hash": self.teacher_text_hash,
            "teacher_model_id": self.teacher_model_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class BatchResult:
    """Deterministic output of one offline batch teacher run."""

    run_label: str
    teacher_model_id: str
    hash_algorithm: str
    fixtures_sha256: str
    records: tuple[TeacherBatchRecord, ...]

    @property
    def turns(self) -> int:
        """Return the number of transcribed turns."""
        return len(self.records)

    @property
    def calls(self) -> int:
        """Return the number of distinct source calls."""
        return len({record.call_id for record in self.records})


def run_batch(
    turns: TurnSource,
    teacher: TeacherTranscriber,
    settings: TeacherSettings | None = None,
    *,
    fixtures_sha256: str | None = None,
) -> BatchResult:
    """Transcribe every injected turn with the injected teacher.

    Args:
        turns: Read-only turn source (fixtures by default).
        teacher: Offline transcriber seam; the default fake is metadata-only.
        settings: Typed settings; defaults to environment-backed values.
        fixtures_sha256: Precomputed digest of read-only fixture text inputs.

    Returns:
        A deterministic batch result with per-turn content hashes.

    Raises:
        ValidationError: If duplicate event IDs or a malformed transcript appear.

    """
    config = settings or TeacherSettings()
    records: list[TeacherBatchRecord] = []
    seen: set[str] = set()
    for record in turns.iter_turns():
        if record.event_id in seen:
            raise ValidationError(f"duplicate event_id in teacher batch: {record.event_id}")
        seen.add(record.event_id)
        transcript = teacher.transcribe(record)
        if transcript.model_id != teacher.model_id:
            raise ValidationError(f"teacher model_id mismatch for {record.event_id}")
        stt = record.event.get("stt")
        production_text = stt.get("text", "") if isinstance(stt, Mapping) else ""
        production_text = production_text if isinstance(production_text, str) else ""
        production_hash = canonical_hash(
            {
                "kind": "voiceloop.production.transcript",
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "event_id": record.event_id,
                "text": production_text,
            },
            algorithm=config.hash_algorithm,
        )
        records.append(
            TeacherBatchRecord(
                event_id=record.event_id,
                call_id=record.call_id,
                turn_index=record.turn_index,
                audio_uri=record.audio_uri,
                production_text=production_text,
                teacher_text=transcript.text,
                production_text_hash=production_hash,
                teacher_text_hash=transcript.content_hash,
                teacher_model_id=transcript.model_id,
                reason=transcript.reason,
            )
        )
    return BatchResult(
        run_label=RUN_LABEL,
        teacher_model_id=teacher.model_id,
        hash_algorithm=config.hash_algorithm,
        fixtures_sha256=fixtures_sha256 or fixture_inputs_digest(config.fixtures_path),
        records=tuple(records),
    )
