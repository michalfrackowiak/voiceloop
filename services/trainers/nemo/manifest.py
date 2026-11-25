"""Read-only loader for the P2-05 NeMo ASR manifest JSONL export.

The export is consumed as text; no WAV bytes, sealed golden set or regression
set is read here. Rows are validated before they reach the fake fine-tune, and
the source file is never written back.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from voiceloop_common import ValidationError


def canonical_json(value: object) -> str:
    """Return deterministic compact JSON text for hashing and stand-in artifacts."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    """Return the lowercase SHA-256 hex digest of UTF-8 text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class NemoManifestRow:
    """One validated row of the P2-05 NeMo ASR export.

    The export shape is exactly ``audio_filepath``, ``duration`` and ``text``.
    Text and paths are hashed before entering the stand-in artifact so no
    transcript or audio path is copied into a persisted output.
    """

    audio_filepath: str
    duration: float
    text: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> NemoManifestRow:
        """Validate one decoded JSONL row.

        Args:
            raw: Decoded object from the NeMo manifest export.

        Returns:
            A validated :class:`NemoManifestRow`.

        Raises:
            ValidationError: If a required field is missing, empty or malformed.

        """
        audio_filepath = raw.get("audio_filepath")
        if not isinstance(audio_filepath, str) or not audio_filepath.strip():
            raise ValidationError("NeMo row requires a non-empty audio_filepath string")
        duration = raw.get("duration")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise ValidationError("NeMo row duration must be a positive finite number")
        duration_value = float(duration)
        if not math.isfinite(duration_value) or duration_value <= 0.0:
            raise ValidationError("NeMo row duration must be a positive finite number")
        text = raw.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValidationError("NeMo row requires a non-empty text string")
        return cls(audio_filepath=audio_filepath, duration=duration_value, text=text)

    def to_dict(self) -> dict[str, Any]:
        """Return the exact P2-05 export shape as a JSON-serialisable dict."""
        return {"audio_filepath": self.audio_filepath, "duration": self.duration, "text": self.text}

    def fingerprint(self) -> str:
        """Return a stable SHA-256 fingerprint of this row, independent of key order."""
        return sha256_text(canonical_json(self.to_dict()))


class NemoManifestSource(Protocol):
    """Injectable bridge from a P2-05 dataset manifest to NeMo training rows."""

    def load_rows(self, manifest: Mapping[str, Any]) -> tuple[NemoManifestRow, ...]:
        """Return validated read-only rows for one dataset manifest."""
        ...


class JsonlNemoManifestSource:
    """Reader for the plain NeMo JSONL written by P2-05's ``export_nemo``.

    The path may be injected in the constructor or supplied as a non-schema
    ``nemo_path`` sidecar on the manifest passed to the trainer. Files are read
    only; audio references are never opened.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        """Configure an optional export path.

        Args:
            path: Explicit JSONL path; when omitted, ``manifest['nemo_path']``
                is used. An empty value fails closed at load time.

        """
        self._path = Path(path) if path is not None else None

    def load_rows(self, manifest: Mapping[str, Any]) -> tuple[NemoManifestRow, ...]:
        """Parse and validate all rows of the NeMo export.

        Args:
            manifest: Dataset manifest, optionally carrying a ``nemo_path`` sidecar.

        Returns:
            Validated rows in file order.

        Raises:
            ValidationError: If the path is missing/unreadable, the file is
                empty, or any row is malformed.

        """
        raw_path: str | Path | None = self._path
        if raw_path is None:
            raw_path = manifest.get("nemo_path")
        if not isinstance(raw_path, (str, Path)) or not str(raw_path):
            raise ValidationError("NeMo manifest source needs a path or manifest['nemo_path']")
        path = Path(raw_path)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValidationError(f"cannot read NeMo manifest at {path}: {exc}") from exc
        rows: list[NemoManifestRow] = []
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(
                    f"invalid NeMo manifest JSON on line {number}: {exc}"
                ) from exc
            if not isinstance(decoded, Mapping):
                raise ValidationError(f"NeMo manifest line {number} must be a JSON object")
            try:
                rows.append(NemoManifestRow.from_mapping(decoded))
            except ValidationError as exc:
                raise ValidationError(f"NeMo manifest line {number}: {exc}") from exc
        if not rows:
            raise ValidationError("NeMo manifest export has no usable rows")
        return tuple(rows)


class InMemoryNemoManifestSource:
    """Injectable in-memory source for synthetic rows; ignores manifest paths."""

    def __init__(self, rows: Sequence[NemoManifestRow]) -> None:
        """Hold an immutable snapshot of pre-validated synthetic rows."""
        self._rows = tuple(rows)

    def load_rows(self, manifest: Mapping[str, Any]) -> tuple[NemoManifestRow, ...]:
        """Return the injected rows (the manifest is accepted but unused)."""
        return self._rows
