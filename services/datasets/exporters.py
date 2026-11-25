"""Plain-text dataset exporters (P2-05.6).

Writers emit NeMo ASR manifest JSONL, a Hugging Face ``datasets`` JSON array
and TRL-style SFT JSONL using only the standard library. Real target-library
loading requires optional dependencies (NeMo, ``datasets``, TRL) and remains a
follow-up; tests parse the files back with ``json`` to check counts and fields.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .examples import DatasetExample

SFT_SYSTEM_PROMPT = "Extract the booking slots from the guest utterance as JSON."


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> Path:
    """Write one compact JSON object per line, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n" for record in records
        ),
        encoding="utf-8",
    )
    return path


def export_nemo(examples: Sequence[DatasetExample], path: str | Path) -> Path:
    """Write a NeMo-compatible ASR manifest from STT examples.

    Each line has ``audio_filepath``, ``duration`` (seconds) and ``text``.
    Parser-only examples are excluded; audio URIs are kept verbatim.

    Args:
        examples: All dataset examples; STT rows are selected.
        path: Destination ``.jsonl`` file.

    Returns:
        The written path.

    """
    records = [
        {
            "audio_filepath": example.audio_uri,
            "duration": round(example.duration_ms / 1000, 3),
            "text": example.target_text,
        }
        for example in examples
        if example.kind == "stt"
    ]
    return _write_jsonl(Path(path), records)


def export_hf(examples: Sequence[DatasetExample], path: str | Path) -> Path:
    """Write all examples as a JSON array loadable by ``datasets.load_dataset``.

    Args:
        examples: All dataset examples.
        path: Destination ``.json`` file.

    Returns:
        The written path.

    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps([example.to_dict() for example in examples], indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return destination


def export_sft(examples: Sequence[DatasetExample], path: str | Path) -> Path:
    """Write parser examples as TRL-style chat messages in JSONL.

    The user message includes dialogue context and the source transcript
    (corrected for clean parser examples, noisy hypothesis otherwise); the
    assistant message is the reviewed slot object as compact JSON.

    Args:
        examples: All dataset examples; parser rows are selected.
        path: Destination ``.jsonl`` file.

    Returns:
        The written path.

    """
    records: list[dict[str, Any]] = []
    for example in examples:
        if example.kind not in ("parser_clean", "parser_noisy"):
            continue
        context = json.dumps(example.dialogue_context, sort_keys=True, ensure_ascii=False)
        user = f"Context: {context}\nUtterance: {example.source_text}"
        records.append(
            {
                "messages": [
                    {"role": "system", "content": SFT_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                    {
                        "role": "assistant",
                        "content": json.dumps(example.slots, sort_keys=True, ensure_ascii=False),
                    },
                ],
                "example_id": example.example_id,
                "call_id": example.call_id,
                "turn_index": example.turn_index,
                "kind": example.kind,
            }
        )
    return _write_jsonl(Path(path), records)


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Paths and record counts for one set of exports."""

    nemo_path: Path
    hf_path: Path
    sft_path: Path
    counts: dict[str, int]


def export_all(
    examples: Sequence[DatasetExample],
    output_dir: str | Path,
    *,
    prefix: str = "dataset",
) -> ExportResult:
    """Write all three exports into one directory.

    Args:
        examples: All dataset examples.
        output_dir: Destination directory, created if absent.
        prefix: File-name prefix for the three outputs.

    Returns:
        Paths and parsed-back record counts for verification.

    """
    base = Path(output_dir)
    nemo = export_nemo(examples, base / f"{prefix}.nemo.jsonl")
    hf = export_hf(examples, base / f"{prefix}.hf.json")
    sft = export_sft(examples, base / f"{prefix}.sft.jsonl")
    return ExportResult(
        nemo_path=nemo,
        hf_path=hf,
        sft_path=sft,
        counts={
            "nemo": len(
                [line for line in nemo.read_text(encoding="utf-8").splitlines() if line.strip()]
            ),
            "hf": len(json.loads(hf.read_text(encoding="utf-8"))),
            "sft": len(
                [line for line in sft.read_text(encoding="utf-8").splitlines() if line.strip()]
            ),
        },
    )
