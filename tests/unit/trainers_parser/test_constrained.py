"""Strict booking-schema constrained-decoding checks (reject, never repair)."""

from __future__ import annotations

from typing import Any

import pytest

from services.trainers.parser import (
    BookingSchemaDecoder,
    ConstrainedParserRuntime,
    FakeAdapterStore,
    FakeVllmRuntime,
    ParserLoraTrainer,
    canonical_json,
    decode_booking_output,
)
from tests.unit.trainers_parser.conftest import make_slots
from voiceloop_common import ValidationError


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[]",
        "null",
        '"just text"',
        '{"check_in": "2026-12-12"} trailing',
    ],
)
def test_malformed_or_non_object_output_is_rejected(raw: str) -> None:
    with pytest.raises(ValidationError):
        decode_booking_output(raw)


@pytest.mark.parametrize(
    "changes",
    [
        {"check_out": "2026-12-01"},
        {"guests": 0},
        {"room_type": "penthouse"},
        {"guest_name": ""},
        {"contact": {"email": "not-an-email"}},
    ],
)
def test_schema_and_cross_field_violations_are_rejected(changes: dict[str, Any]) -> None:
    slots = make_slots()
    if "contact" in changes:
        slots["contact"] = changes["contact"]
    else:
        slots.update(changes)
    with pytest.raises(ValidationError):
        BookingSchemaDecoder().decode(canonical_json(slots))


def test_missing_field_is_rejected_without_repair() -> None:
    slots = make_slots()
    del slots["room_type"]
    decoder = BookingSchemaDecoder()
    with pytest.raises(ValidationError, match="room_type"):
        decoder.decode(canonical_json(slots))


def test_valid_output_roundtrips() -> None:
    slots = make_slots()
    assert decode_booking_output(canonical_json(slots)) == slots


def test_constrained_runtime_accepts_valid_and_rejects_injected_bad_output(
    manifest: dict[str, Any], core_settings: Any, examples: tuple[Any, ...]
) -> None:
    store = FakeAdapterStore()
    result = ParserLoraTrainer(examples=examples, store=store).train(manifest, core_settings)
    uri = result["artifact_uri"]
    runtime = FakeVllmRuntime(store)
    constrained = ConstrainedParserRuntime(runtime)
    handle = constrained.load_adapter(uri)
    prompt = f"Context: {{}}\nUtterance: {examples[0].source_text}"
    assert constrained.generate(handle, prompt) == examples[0].slots

    bad_runtime = FakeVllmRuntime(
        store, generation_overrides={handle: canonical_json({**make_slots(), "guests": 0})}
    )
    constrained_bad = ConstrainedParserRuntime(bad_runtime)
    bad_handle = constrained_bad.load_adapter(uri)
    with pytest.raises(ValidationError):
        constrained_bad.generate(bad_handle, prompt)
    constrained.unload_adapter(handle)
    constrained.unload_adapter(handle)
    assert runtime.stats.unloads == 1
