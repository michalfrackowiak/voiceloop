"""Deterministic, call-level sampler for golden-set candidates.

The sampling unit is the **call**, never the utterance and never the failure
label (context.md section 2 invariants 6 and 7). Every turn of a selected call
is kept, which prevents speaker leakage and keeps the sample unbiased with
respect to triage outcomes. The sampler cannot see ``failure_type`` at all:
:class:`SourceTurn` has no such field.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voiceloop_common import ValidationError

from .audit import resolve_path
from .settings import GoldenSettings


class SamplingError(ValidationError):
    """Raised when a target band cannot be met without splitting a call."""


@dataclass(frozen=True)
class SourceTurn:
    """One candidate turn, deliberately without any triage/failure metadata."""

    call_id: str
    turn_index: int
    audio_uri: str
    hypothesis: str
    corrected_transcript: str
    slots: dict[str, Any]

    @property
    def source_id(self) -> str:
        """Return the stable ``call_id:turn_index`` identifier."""
        return f"{self.call_id}:{self.turn_index}"


@dataclass(frozen=True)
class SampleResult:
    """A deterministic call-level sample and its audit metadata."""

    turns: tuple[SourceTurn, ...]
    call_ids: tuple[str, ...]
    selection_order: tuple[str, ...]
    seed: int
    target_min: int
    target_max: int
    skipped_calls: tuple[str, ...]


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from ``path``."""
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read JSON at {path}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ValidationError(f"expected a JSON object at {path}")
    return decoded


def load_source_turns(path: str | Path | None = None) -> tuple[SourceTurn, ...]:
    """Load the read-only synthetic label manifest into failure-blind turns.

    Extra manifest fields (including ``failure_type`` and ``case_id``) are
    ignored. The STT hypothesis is read from each referenced turn event.

    Args:
        path: JSONL manifest; defaults to the setting's fixture path.

    Returns:
        Turns sorted by ``(call_id, turn_index)``.

    """
    settings = GoldenSettings()
    fixture_path = resolve_path(path or settings.fixtures_path)
    try:
        lines = fixture_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValidationError(f"cannot read fixture turns: {fixture_path}") from exc

    turns: list[SourceTurn] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"invalid fixture JSON on line {number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValidationError(f"fixture line {number} must be an object")
        for field in ("call_id", "turn_index", "audio_path", "event_path"):
            if field not in row:
                raise ValidationError(f"fixture line {number} missing {field}")
        event = _read_json(resolve_path(str(row["event_path"])))
        stt = event.get("stt")
        hypothesis = stt.get("text", "") if isinstance(stt, dict) else ""
        if not isinstance(hypothesis, str):
            raise ValidationError(f"fixture line {number}: stt.text must be text")
        slots = row.get("expected_slots", {})
        transcript = row.get("expected_transcript", "")
        if not isinstance(slots, dict) or not isinstance(transcript, str):
            raise ValidationError(f"fixture line {number}: invalid label fields")
        turns.append(
            SourceTurn(
                call_id=str(row["call_id"]),
                turn_index=int(row["turn_index"]),
                audio_uri=str(row["audio_path"]),
                hypothesis=hypothesis,
                corrected_transcript=transcript,
                slots=dict(slots),
            )
        )
    if not turns:
        raise ValidationError("fixture manifest contains no turns")
    return tuple(sorted(turns, key=lambda turn: (turn.call_id, turn.turn_index)))


def sample_calls(
    turns: Sequence[SourceTurn],
    *,
    seed: int,
    target_min: int,
    target_max: int,
) -> SampleResult:
    """Select whole calls with a seeded RNG until the utterance band is met.

    Calls are shuffled from a sorted call-ID list, so the result is stable for
    a given seed and independent of input row order. A call is skipped rather
    than split when it would push the count above ``target_max``.

    Args:
        turns: Failure-blind candidate turns.
        seed: Seed for the deterministic RNG.
        target_min: Minimum number of utterances required.
        target_max: Maximum number of utterances permitted.

    Returns:
        The selected turns, sorted by call and turn index.

    Raises:
        SamplingError: If parameters are invalid or the band is unreachable
            without splitting a call.

    """
    if target_min < 1 or target_max < target_min:
        raise SamplingError("require 1 <= target_min <= target_max")
    grouped: dict[str, list[SourceTurn]] = {}
    for turn in turns:
        grouped.setdefault(turn.call_id, []).append(turn)
    order = sorted(grouped)
    random.Random(seed).shuffle(order)

    chosen: list[str] = []
    skipped: list[str] = []
    count = 0
    for call_id in order:
        if count >= target_min:
            break
        size = len(grouped[call_id])
        if count + size <= target_max:
            chosen.append(call_id)
            count += size
        else:
            skipped.append(call_id)
    if count < target_min:
        raise SamplingError(
            f"cannot reach {target_min} utterances from whole calls (got {count}); "
            "lower the target or add calls"
        )
    selected = tuple(
        sorted(
            (turn for call_id in chosen for turn in grouped[call_id]),
            key=lambda turn: (turn.call_id, turn.turn_index),
        )
    )
    return SampleResult(
        turns=selected,
        call_ids=tuple(sorted(chosen)),
        selection_order=tuple(chosen),
        seed=seed,
        target_min=target_min,
        target_max=target_max,
        skipped_calls=tuple(sorted(skipped)),
    )


def select_entire_pool(turns: Sequence[SourceTurn]) -> SampleResult:
    """Select every call: the v1 synthetic workflow stand-in, not a real sample.

    The fixture pool contains 36 tone/noise/silence turns across six calls; it
    is far smaller than the production 300-500 band and is never represented
    as a production golden set.
    """
    if not turns:
        raise SamplingError("cannot select an empty pool")
    call_ids = tuple(sorted({turn.call_id for turn in turns}))
    return SampleResult(
        turns=tuple(sorted(turns, key=lambda turn: (turn.call_id, turn.turn_index))),
        call_ids=call_ids,
        selection_order=call_ids,
        seed=0,
        target_min=len(turns),
        target_max=len(turns),
        skipped_calls=(),
    )
