"""Tests for the committed synthetic fixture set (P0-05)."""

from __future__ import annotations

import json
import re
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from schemas import REPO_ROOT, schema_path, validate
from schemas.booking import BookingValidationError, validate_booking
from tests.fixtures.generate_fixtures import (
    DeterministicToneBackend,
    generate_fixture_set,
    load_tts_backend,
)

FIXTURES_DIR: Path = Path(__file__).resolve().parent
MANIFEST: Path = FIXTURES_DIR / "turns.jsonl"
ALLOWED_FAILURE_TYPES = {
    "stt_wrong",
    "parser_wrong",
    "both_fine",
    "unusable_audio",
    "clean",
}
SYNTHETIC_NAMES = {
    "Ada Fixture",
    "Sam Sample",
    "Jordan Example",
    "Morgan Mock",
    "Casey Placeholder",
    "Riley Synthetic",
    "Mr. Smith",
}
ALLOWED_EMAIL_SUFFIX = "@example.com"
ALLOWED_PHONE_PREFIX = "+1 555 010"
BOOKING_PATH = schema_path("booking.schema.json")


def _read_labels(path: Path = MANIFEST) -> list[dict[str, Any]]:
    """Read a JSONL labels manifest."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _resolve(relative: str) -> Path:
    """Resolve a repo-relative fixture URI without allowing escapes."""
    candidate = Path(relative)
    assert not candidate.is_absolute(), f"fixture URI must be relative: {relative}"
    assert ".." not in candidate.parts, f"fixture URI must stay in repo: {relative}"
    return REPO_ROOT / candidate


def _event(label: dict[str, Any]) -> dict[str, Any]:
    """Load the event payload referenced by a label row."""
    return json.loads(_resolve(label["event_path"]).read_text(encoding="utf-8"))


def _wav(label: dict[str, Any]) -> tuple[int, bytes]:
    """Read a labelled WAV's sample rate and raw PCM frames."""
    with wave.open(str(_resolve(label["audio_path"])), "rb") as wav_file:
        return wav_file.getframerate(), wav_file.readframes(wav_file.getnframes())


def _case(case_id: str) -> dict[str, Any]:
    """Return the single label row for a named failure case."""
    matches = [label for label in LABELS if label["case_id"] == case_id]
    assert len(matches) == 1, f"expected one {case_id} fixture, found {len(matches)}"
    return matches[0]


def _strings(value: Any) -> Iterator[str]:
    """Yield every string nested inside a JSON-like value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


LABELS: list[dict[str, Any]] = _read_labels()


def test_at_least_30_turns_across_six_calls() -> None:
    assert len(LABELS) >= 30
    assert len({label["call_id"] for label in LABELS}) >= 6
    assert all(label["synthetic"] is True for label in LABELS)


def test_no_duplicate_call_and_turn_index() -> None:
    keys = [(label["call_id"], label["turn_index"]) for label in LABELS]
    assert len(keys) == len(set(keys))


def test_every_event_validates_against_turn_event_schema() -> None:
    for label in LABELS:
        validate(_event(label), "turn_event.schema.json")


def test_events_match_labels() -> None:
    for label in LABELS:
        event = _event(label)
        assert event["call_id"] == label["call_id"]
        assert event["turn_index"] == label["turn_index"]
        assert event["audio"]["uri"] == label["audio_path"]
        assert event["audio"]["sample_rate_hz"] == label["sample_rate_hz"]
        assert event["consent"] == label["consent"]
        assert "stt" in event and "parser" in event and "dialogue_context" in event
        if label["expected_outcome"] is None:
            assert "outcome" not in event
        else:
            assert event["outcome"]["status"] == label["expected_outcome"]
            assert event["outcome"]["final_record"] == label["expected_slots"]


def test_audio_files_exist_with_declared_sample_rate() -> None:
    for label in LABELS:
        event = _event(label)
        rate, pcm = _wav(label)
        assert rate == label["sample_rate_hz"] == event["audio"]["sample_rate_hz"]
        assert len(pcm) > 0
        assert len(pcm) % 2 == 0
        samples = len(pcm) // 2
        duration_ms = round(1000 * samples / rate)
        assert abs(duration_ms - event["audio"]["duration_ms"]) <= 1
        assert _resolve(label["audio_path"]).stat().st_size < 40_000


def test_exactly_silent_turns_are_truly_silent() -> None:
    silent = [label for label in LABELS if label["audio_kind"] == "silence"]
    assert len(silent) >= 2
    for label in silent:
        _, pcm = _wav(label)
        assert all(byte == 0 for byte in pcm)
    assert any(_event(label)["stt"]["text"] == "" for label in silent)


def test_line_noise_turns_have_nonzero_samples() -> None:
    noisy = [label for label in LABELS if label["audio_kind"] == "noise"]
    assert len(noisy) >= 2
    for label in noisy:
        _, pcm = _wav(label)
        assert any(byte != 0 for byte in pcm)
        assert label["failure_type"] == "unusable_audio"


def test_8000_hz_call_is_native_and_preserved() -> None:
    telephony_calls = {label["call_id"] for label in LABELS if label["sample_rate_hz"] == 8000}
    assert telephony_calls
    for call_id in telephony_calls:
        call_labels = [label for label in LABELS if label["call_id"] == call_id]
        assert {label["sample_rate_hz"] for label in call_labels} == {8000}
        for label in call_labels:
            rate, _ = _wav(label)
            assert rate == 8000


def test_failure_type_vocabulary_is_covered() -> None:
    observed = {label["failure_type"] for label in LABELS}
    assert observed == ALLOWED_FAILURE_TYPES
    assert all(label["failure_type"] in ALLOWED_FAILURE_TYPES for label in LABELS)


def test_expected_slots_and_schema_valid_parsers_match_booking_shape() -> None:
    for label in LABELS:
        slots = label["expected_slots"]
        assert isinstance(slots, dict)
        if slots:
            validate_booking(slots, BOOKING_PATH)
        parser = _event(label)["parser"]
        if parser["schema_valid"]:
            validate_booking(parser["output"], BOOKING_PATH)
        else:
            with pytest.raises(BookingValidationError):
                validate_booking(parser["output"], BOOKING_PATH)


def test_named_failure_misheard_surname() -> None:
    label = _case("misheard_surname")
    event = _event(label)
    assert label["failure_type"] == "stt_wrong"
    assert "Mr. Smith" in label["expected_transcript"]
    assert label["expected_slots"]["guest_name"] == "Mr. Smith"
    assert "Smythe" in event["stt"]["text"]
    assert event["parser"]["output"]["guest_name"] == "Mr. Smythe"
    low_confidence = [word for word in event["stt"]["words"] if word["conf"] < 0.5]
    assert any("smythe" in word["w"].lower() for word in low_confidence)


def test_named_failure_date_confusion() -> None:
    label = _case("date_confusion")
    event = _event(label)
    assert label["failure_type"] == "parser_wrong"
    assert "three nights" in label["expected_transcript"]
    assert event["stt"]["text"] == label["expected_transcript"]
    parser_output = event["parser"]["output"]
    assert parser_output["guests"] != label["expected_slots"]["guests"]
    assert parser_output["check_out"] != label["expected_slots"]["check_out"]


def test_named_failure_schema_invalid_parse() -> None:
    label = _case("schema_invalid_parse")
    event = _event(label)
    assert label["failure_type"] == "parser_wrong"
    parser_output = event["parser"]["output"]
    assert parser_output["check_out"] < parser_output["check_in"]
    assert event["parser"]["schema_valid"] is False
    validate_booking(label["expected_slots"], BOOKING_PATH)
    with pytest.raises(BookingValidationError, match="check_out must be after check_in"):
        validate_booking(parser_output, BOOKING_PATH)


def test_no_training_consent_fixture_exists() -> None:
    label = _case("no_training_consent")
    assert label["consent"] == {"recording": True, "training": False}
    assert _event(label)["consent"] == label["consent"]


def test_no_real_looking_pii() -> None:
    documents: list[Any] = [*LABELS, *(_event(label) for label in LABELS)]
    for document in documents:
        for text in _strings(document):
            if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", text):
                continue  # event UUID, not a phone number
            for email in re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]*[A-Za-z0-9]", text):
                assert email.endswith(ALLOWED_EMAIL_SUFFIX), email
            for phone in re.findall(r"\+?\d[\d ().-]{7,}\d", text):
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", phone):
                    continue  # calendar date, not a phone number
                assert phone.startswith(ALLOWED_PHONE_PREFIX), phone
            assert not re.search(r"(?<!\d)\d{13,19}(?!\d)", text), text
    for label in LABELS:
        if label["expected_slots"]:
            assert label["expected_slots"]["guest_name"] in SYNTHETIC_NAMES


def test_regeneration_is_deterministic(tmp_path: Path) -> None:
    destination = tmp_path / "regenerated"
    generated = generate_fixture_set(output_dir=destination)
    assert generated == LABELS

    relative_paths = {Path(label["audio_path"]).relative_to("tests/fixtures") for label in LABELS}
    relative_paths.update(
        Path(label["event_path"]).relative_to("tests/fixtures") for label in LABELS
    )
    relative_paths.update(
        Path("contexts") / f"{call_id}.jsonl" for call_id in {row["call_id"] for row in LABELS}
    )
    relative_paths.update({Path("turns.jsonl"), Path("summary.json")})
    for relative in relative_paths:
        assert (destination / relative).read_bytes() == (FIXTURES_DIR / relative).read_bytes()


def test_tts_backend_hook_is_used_but_silence_stays_exact(tmp_path: Path) -> None:
    class ConstantBackend:
        """Fake backend that stands in for a future real TTS engine."""

        def synthesize(
            self, *, text: str, sample_rate_hz: int, duration_ms: int, seed: int
        ) -> list[int]:
            del text, seed
            return [1234] * (sample_rate_hz * duration_ms // 1000)

    destination = tmp_path / "hooked"
    generated = generate_fixture_set(output_dir=destination, tts_backend=ConstantBackend())
    assert len(generated) == len(LABELS)
    tone = next(
        label
        for label in generated
        if label["audio_kind"] == "tone" and label["sample_rate_hz"] == 16000
    )
    silent = next(label for label in generated if label["audio_kind"] == "silence")
    tone_path = destination / Path(tone["audio_path"]).relative_to("tests/fixtures")
    with wave.open(str(tone_path), "rb") as wav_file:
        assert wav_file.readframes(wav_file.getnframes()) == b"\xd2\x04" * wav_file.getnframes()
    silent_path = destination / Path(silent["audio_path"]).relative_to("tests/fixtures")
    with wave.open(str(silent_path), "rb") as wav_file:
        assert all(byte == 0 for byte in wav_file.readframes(wav_file.getnframes()))


def test_tts_backend_default_and_loader() -> None:
    backend = DeterministicToneBackend()
    samples = backend.synthesize(text="hello", sample_rate_hz=8000, duration_ms=200, seed=7)
    assert len(samples) == 1600
    loaded = load_tts_backend("tests.fixtures.generate_fixtures:DeterministicToneBackend")
    assert isinstance(loaded, DeterministicToneBackend)
    with pytest.raises(ValueError, match="module:attribute"):
        load_tts_backend("no-separator")
