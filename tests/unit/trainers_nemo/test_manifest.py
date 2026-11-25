"""Tests for the read-only NeMo manifest export loader."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from services.trainers.nemo import (
    InMemoryNemoManifestSource,
    JsonlNemoManifestSource,
    NemoManifestRow,
)
from voiceloop_common import ValidationError

from .conftest import DATASET_HASH


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_loads_p2_style_export_without_mutating_it(nemo_export: Path) -> None:
    before = _file_hash(nemo_export)
    rows = JsonlNemoManifestSource(str(nemo_export)).load_rows({"content_hash": DATASET_HASH})
    assert len(rows) == 3
    assert set(rows[0].to_dict()) == {"audio_filepath", "duration", "text"}
    assert rows[0].text.startswith("Check in")
    assert rows[0].audio_filepath.startswith("synthetic://audio/")
    assert _file_hash(nemo_export) == before


def test_manifest_sidecar_path(nemo_export: Path) -> None:
    rows = JsonlNemoManifestSource().load_rows(
        {"content_hash": DATASET_HASH, "nemo_path": str(nemo_export)}
    )
    assert len(rows) == 3
    fingerprints = [row.fingerprint() for row in rows]
    assert len(set(fingerprints)) == 3
    assert all(len(fingerprint) == 64 for fingerprint in fingerprints)


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"audio_filepath": "synthetic://a.wav", "duration": 0.2},
        {"audio_filepath": "synthetic://a.wav", "text": "hello"},
        {"audio_filepath": "", "duration": 0.2, "text": "hello"},
        {"audio_filepath": "synthetic://a.wav", "duration": 0.0, "text": "hello"},
        {"audio_filepath": "synthetic://a.wav", "duration": float("nan"), "text": "hello"},
        {"audio_filepath": "synthetic://a.wav", "duration": 0.2, "text": "   "},
        {"audio_filepath": "synthetic://a.wav", "duration": "0.2", "text": "hello"},
    ],
)
def test_rejects_malformed_rows(row: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        NemoManifestRow.from_mapping(row)


def test_missing_path_fails_closed() -> None:
    with pytest.raises(ValidationError, match="nemo_path"):
        JsonlNemoManifestSource().load_rows({"content_hash": DATASET_HASH})


def test_missing_file_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="cannot read"):
        JsonlNemoManifestSource(tmp_path / "absent.jsonl").load_rows({})


def test_empty_export_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n  \n", encoding="utf-8")
    with pytest.raises(ValidationError, match="no usable rows"):
        JsonlNemoManifestSource(path).load_rows({})


def test_malformed_line_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text(
        json.dumps({"audio_filepath": "synthetic://a.wav", "duration": 0.2, "text": "ok"})
        + "\n{bad json\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="line 2"):
        JsonlNemoManifestSource(path).load_rows({})


def test_in_memory_source_is_a_valid_injection(rows: tuple[NemoManifestRow, ...]) -> None:
    source = InMemoryNemoManifestSource(rows)
    assert source.load_rows({"content_hash": DATASET_HASH}) == rows
