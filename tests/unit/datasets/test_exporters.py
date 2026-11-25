"""Tests that plain-text exports parse back with the expected counts and fields."""

from __future__ import annotations

import json
from pathlib import Path

from services.datasets import (
    DatasetExample,
    build_dataset_from_fixtures,
    export_all,
    export_hf,
    export_nemo,
    export_sft,
)


def test_exports_parse_back_with_counts_and_fields(tmp_path: Path) -> None:
    """NeMo, HF and SFT files round-trip through json with correct records."""
    build = build_dataset_from_fixtures()
    result = export_all(build.examples, tmp_path)
    expected_stt = [example for example in build.examples if example.kind == "stt"]
    expected_parser = [example for example in build.examples if example.kind.startswith("parser")]

    nemo = [
        json.loads(line)
        for line in result.nemo_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(nemo) == len(expected_stt) == result.counts["nemo"]
    assert set(nemo[0]) == {"audio_filepath", "duration", "text"}
    assert nemo[0]["text"] == expected_stt[0].target_text
    assert nemo[0]["audio_filepath"] == expected_stt[0].audio_uri
    assert nemo[0]["duration"] == round(expected_stt[0].duration_ms / 1000, 3)

    hf = json.loads(result.hf_path.read_text(encoding="utf-8"))
    assert len(hf) == len(build.examples) == result.counts["hf"]
    assert hf[0]["example_id"] == build.examples[0].example_id
    assert "dialogue_context" in hf[0]
    assert "prior_turns" in hf[0]["dialogue_context"]

    sft = [
        json.loads(line)
        for line in result.sft_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(sft) == len(expected_parser) == result.counts["sft"]
    assert [message["role"] for message in sft[0]["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert json.loads(sft[0]["messages"][2]["content"]) == expected_parser[0].slots
    assert expected_parser[0].source_text in sft[0]["messages"][1]["content"]


def test_empty_exports_are_valid_empty_files(tmp_path: Path) -> None:
    """Exporting no examples writes empty JSONL and an empty JSON array."""
    empty: tuple[DatasetExample, ...] = ()
    assert export_nemo(empty, tmp_path / "empty.nemo.jsonl").read_text(encoding="utf-8") == ""
    assert (
        json.loads(export_hf(empty, tmp_path / "empty.hf.json").read_text(encoding="utf-8")) == []
    )
    assert export_sft(empty, tmp_path / "empty.sft.jsonl").read_text(encoding="utf-8") == ""
