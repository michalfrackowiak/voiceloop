"""Deterministic synthetic fixture generator for VoiceLoop (P0-05).

Everything produced here is synthetic. The committed fixtures use a stdlib-only
tone-carrier backend: the WAVs contain tones, noise or exact silence, NEVER real
speech or real guest audio. The ``tts_backend`` hook exists so a real TTS engine
can be plugged in later without changing the event/label pipeline.

Regenerate with: .\\.venv\\Scripts\\python.exe tests/fixtures/generate_fixtures.py
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import struct
import sys
import uuid
import wave
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # support direct `python tests/fixtures/generate_fixtures.py`
    sys.path.insert(0, str(REPO_ROOT))

from schemas import (  # noqa: E402 - repo root must be on sys.path for direct invocation
    schema_path,
    validate,
)
from schemas.booking import validate_booking  # noqa: E402

FIXTURES_DIR: Path = Path(__file__).resolve().parent
TURN_EVENT_SCHEMA: str = "turn_event.schema.json"
BOOKING_SCHEMA: str = "booking.schema.json"
REFERENCE_DATE: str = "2026-11-24"
TTS_MODEL_ID: str = "synthetic-stt-stub"
PARSER_MODEL_ID: str = "synthetic-parser-stub"
MODEL_VERSION: str = "1.0.0"
EVENT_NAMESPACE: uuid.UUID = uuid.UUID("2f1a6c34-8d5e-4b7a-9c21-0e3f4a5b6c7d")
GENERATOR_VERSION: str = "1.0.0"
SYNTHETIC_BACKEND_NAME: str = "deterministic-tone-v1"

AudioKind = Literal["tone", "silence", "noise"]
FailureType = Literal["stt_wrong", "parser_wrong", "both_fine", "unusable_audio", "clean"]


class TtsBackend(Protocol):
    """Interface for pluggable speech-carrier synthesis (no real guest data)."""

    def synthesize(
        self, *, text: str, sample_rate_hz: int, duration_ms: int, seed: int
    ) -> list[int]:
        """Return exactly ``sample_rate_hz * duration_ms // 1000`` 16-bit PCM samples.

        Args:
            text: Label transcript the carrier stands in for (not spoken aloud).
            sample_rate_hz: Native output rate; 8000 Hz must not be resampled.
            duration_ms: Requested duration.
            seed: Deterministic per-turn seed.

        Returns:
            Mono 16-bit signed PCM samples.

        """


class DeterministicToneBackend:
    """Stdlib-only backend emitting a deterministic synthetic tone carrier.

    This is NOT speech synthesis: the samples are harmonic tones used as
    stand-ins so audio plumbing can be tested without a TTS engine or GPU.
    """

    def synthesize(
        self, *, text: str, sample_rate_hz: int, duration_ms: int, seed: int
    ) -> list[int]:
        """Return a deterministic, low-amplitude tone carrier as 16-bit PCM."""
        rng = random.Random(seed)
        word_count = max(1, len(text.split()))
        sample_count = sample_rate_hz * duration_ms // 1000
        f0 = 130.0 + (seed % 7) * 17.0
        f1 = f0 * 2.0 + (seed % 5) * 11.0
        f2 = f0 * 3.0 + (seed % 3) * 7.0
        samples: list[int] = []
        for index in range(sample_count):
            time_s = index / sample_rate_hz
            envelope = _fade_envelope(index, sample_count, sample_rate_hz, word_count)
            tone = (
                0.55 * math.sin(2.0 * math.pi * f0 * time_s)
                + 0.28 * math.sin(2.0 * math.pi * f1 * time_s)
                + 0.12 * math.sin(2.0 * math.pi * f2 * time_s)
            )
            samples.append(_to_pcm16(tone * envelope * (0.17 + 0.01 * rng.random())))
        return samples


@dataclass(frozen=True)
class TurnSpec:
    """One synthetic turn: label, hypothesis, parser output and audio kind."""

    call_id: str
    turn_index: int
    audio_kind: AudioKind
    expected_transcript: str
    hypothesis: str
    expected_slots: dict[str, Any]
    parser_output: dict[str, Any]
    parser_schema_valid: bool
    failure_type: FailureType
    expected_slot: str | None
    notes: str
    sample_rate_hz: int = 16000
    training_consent: bool = True
    case_id: str | None = None
    low_conf_words: tuple[str, ...] = ()
    outcome_status: str | None = None


def _turn(
    *,
    call_id: str,
    turn_index: int,
    expected_transcript: str,
    hypothesis: str,
    expected_slots: dict[str, Any],
    parser_output: dict[str, Any],
    parser_schema_valid: bool,
    failure_type: FailureType,
    expected_slot: str | None,
    notes: str,
    audio_kind: AudioKind = "tone",
    sample_rate_hz: int = 16000,
    training_consent: bool = True,
    case_id: str | None = None,
    low_conf_words: tuple[str, ...] = (),
    outcome_status: str | None = None,
) -> TurnSpec:
    """Build a TurnSpec with the common defaults made explicit."""
    return TurnSpec(
        call_id=call_id,
        turn_index=turn_index,
        audio_kind=audio_kind,
        expected_transcript=expected_transcript,
        hypothesis=hypothesis,
        expected_slots=expected_slots,
        parser_output=parser_output,
        parser_schema_valid=parser_schema_valid,
        failure_type=failure_type,
        expected_slot=expected_slot,
        notes=notes,
        sample_rate_hz=sample_rate_hz,
        training_consent=training_consent,
        case_id=case_id,
        low_conf_words=low_conf_words,
        outcome_status=outcome_status,
    )


def _booking(
    *,
    guest_name: str,
    check_in: str,
    check_out: str,
    guests: int,
    room_type: str,
    email: str,
    phone: str,
) -> dict[str, Any]:
    """Build a complete booking slot object in the reference booking shape."""
    return {
        "check_in": check_in,
        "check_out": check_out,
        "guests": guests,
        "room_type": room_type,
        "guest_name": guest_name,
        "contact": {"email": email, "phone": phone},
    }


def _with(booking: dict[str, Any], **changes: Any) -> dict[str, Any]:
    """Return a shallow copy of a booking with selected slot overrides."""
    updated = dict(booking)
    updated.update(changes)
    return updated


def build_turn_specs() -> list[TurnSpec]:
    """Return the fixed set of 36 synthetic turns across six synthetic calls."""
    specs: list[TurnSpec] = []

    # call-0001: 16 kHz clean baseline booking flow.
    ada = _booking(
        guest_name="Ada Fixture",
        check_in="2026-12-12",
        check_out="2026-12-14",
        guests=2,
        room_type="standard",
        email="ada.fixture@example.com",
        phone="+1 555 010 0001",
    )
    specs.append(
        _turn(
            call_id="call-0001",
            turn_index=0,
            expected_transcript="Hi, I'd like to book a room for December.",
            hypothesis="Hi, I'd like to book a room for December.",
            expected_slots={},
            parser_output={},
            parser_schema_valid=False,
            failure_type="clean",
            expected_slot="intent",
            notes="Greeting; no booking slots established yet.",
        )
    )
    specs.append(
        _turn(
            call_id="call-0001",
            turn_index=1,
            expected_transcript=(
                "Check in on the twelfth of December for Ada Fixture, two guests, please."
            ),
            hypothesis="Check in on the twelfth of December for Ada Fixture, two guests, please.",
            expected_slots=ada,
            parser_output=ada,
            parser_schema_valid=True,
            failure_type="clean",
            expected_slot="check_in",
            notes="Date, name and guest count parsed correctly.",
        )
    )
    deluxe_ada = _with(ada, room_type="deluxe")
    specs.append(
        _turn(
            call_id="call-0001",
            turn_index=2,
            expected_transcript="Make that a deluxe room.",
            hypothesis="Make that a deluxe room.",
            expected_slots=deluxe_ada,
            parser_output=deluxe_ada,
            parser_schema_valid=True,
            failure_type="both_fine",
            expected_slot="room_type",
            notes="Room-type correction verified correct by the simulated reviewer.",
        )
    )
    three_nights_ada = _with(deluxe_ada, check_out="2026-12-15")
    specs.append(
        _turn(
            call_id="call-0001",
            turn_index=3,
            expected_transcript="And three nights, not two.",
            hypothesis="And three nights, not two.",
            expected_slots=three_nights_ada,
            parser_output=three_nights_ada,
            parser_schema_valid=True,
            failure_type="both_fine",
            expected_slot="check_out",
            notes="Number correction with low-ish confidence but correct output.",
        )
    )
    specs.append(
        _turn(
            call_id="call-0001",
            turn_index=4,
            expected_transcript="Yes, Ada Fixture is the name on the booking.",
            hypothesis="Yes, Ada Fixture is the name on the booking.",
            expected_slots=three_nights_ada,
            parser_output=three_nights_ada,
            parser_schema_valid=True,
            failure_type="clean",
            expected_slot="guest_name",
            notes="Name confirmation; slots unchanged.",
        )
    )
    confirmed_ada = _with(
        three_nights_ada,
        contact={
            "email": "ada.fixture@example.com",
            "phone": ada["contact"]["phone"],
        },
    )
    specs.append(
        _turn(
            call_id="call-0001",
            turn_index=5,
            expected_transcript="Yes, confirm it. My email is ada.fixture@example.com.",
            hypothesis="Yes, confirm it. My email is ada.fixture@example.com.",
            expected_slots=confirmed_ada,
            parser_output=confirmed_ada,
            parser_schema_valid=True,
            failure_type="clean",
            expected_slot="contact.email",
            notes="Final confirmation with synthetic email address.",
            outcome_status="confirmed",
        )
    )

    # call-0002: 16 kHz named failure cases (misheard surname, date confusion,
    # and a schema-invalid parse with check_out before check_in).
    sam = _booking(
        guest_name="Sam Sample",
        check_in="2026-12-05",
        check_out="2026-12-07",
        guests=1,
        room_type="standard",
        email="sam.sample@example.com",
        phone="+1 555 010 0002",
    )
    specs.append(
        _turn(
            call_id="call-0002",
            turn_index=0,
            expected_transcript="Hello, I need to change a reservation.",
            hypothesis="Hello, I need to change a reservation.",
            expected_slots={},
            parser_output={},
            parser_schema_valid=False,
            failure_type="clean",
            expected_slot="intent",
            notes="Greeting for the failure-case call.",
        )
    )
    specs.append(
        _turn(
            call_id="call-0002",
            turn_index=1,
            expected_transcript=(
                "Check in December fifth for two nights, one guest, a standard room, "
                "for Sam Sample."
            ),
            hypothesis=(
                "Check in December fifth for two nights, one guest, a standard room, "
                "for Sam Sample."
            ),
            expected_slots=sam,
            parser_output=sam,
            parser_schema_valid=True,
            failure_type="clean",
            expected_slot="check_in",
            notes="Initial slot fill for the failure-case call.",
        )
    )
    specs.append(
        _turn(
            call_id="call-0002",
            turn_index=2,
            expected_transcript="The name is Mr. Smith.",
            hypothesis="Mr. Smythe.",
            expected_slots=_with(sam, guest_name="Mr. Smith"),
            parser_output=_with(sam, guest_name="Mr. Smythe"),
            parser_schema_valid=True,
            failure_type="stt_wrong",
            expected_slot="guest_name",
            case_id="misheard_surname",
            low_conf_words=("smythe",),
            notes="Named failure: STT misheard 'Smith' as 'Smythe'; parser copied the error.",
        )
    )
    two_guests_three_nights = _with(sam, guests=2, check_out="2026-12-08")
    specs.append(
        _turn(
            call_id="call-0002",
            turn_index=3,
            expected_transcript="I'd like three nights and two guests, please.",
            hypothesis="I'd like three nights and two guests, please.",
            expected_slots=two_guests_three_nights,
            parser_output=_with(sam, guests=3, check_out="2026-12-07"),
            parser_schema_valid=True,
            failure_type="parser_wrong",
            expected_slot="guests",
            case_id="date_confusion",
            notes="Named failure: 'three' attached to guests instead of nights; STT was correct.",
        )
    )
    tenth_to_fourteenth = _with(sam, check_in="2026-12-10", check_out="2026-12-14")
    specs.append(
        _turn(
            call_id="call-0002",
            turn_index=4,
            expected_transcript="Arrive on the tenth and leave on the fourteenth.",
            hypothesis="Arrive on the tenth and leave on the fourteenth.",
            expected_slots=tenth_to_fourteenth,
            parser_output=_with(sam, check_in="2026-12-14", check_out="2026-12-10"),
            parser_schema_valid=False,
            failure_type="parser_wrong",
            expected_slot="check_out",
            case_id="schema_invalid_parse",
            notes="Named failure: schema-invalid parse with check_out before check_in.",
        )
    )
    corrected_sam = _with(tenth_to_fourteenth, guests=2)
    specs.append(
        _turn(
            call_id="call-0002",
            turn_index=5,
            expected_transcript="Yes, two guests from the tenth to the fourteenth. Confirm.",
            hypothesis="Yes, two guests from the tenth to the fourteenth. Confirm.",
            expected_slots=corrected_sam,
            parser_output=corrected_sam,
            parser_schema_valid=True,
            failure_type="both_fine",
            expected_slot="check_out",
            notes="Staff-corrected slots verified before the outcome was recorded.",
            outcome_status="edited_by_staff",
        )
    )

    # call-0003: 16 kHz unusable audio (exact silence, hallucinated silence,
    # line noise) followed by a recovered booking.
    jordan = _booking(
        guest_name="Jordan Example",
        check_in="2026-12-20",
        check_out="2026-12-22",
        guests=1,
        room_type="standard",
        email="jordan.example@example.com",
        phone="+1 555 010 0003",
    )
    specs.append(
        _turn(
            call_id="call-0003",
            turn_index=0,
            expected_transcript="Can you hear me?",
            hypothesis="Can you hear me?",
            expected_slots={},
            parser_output={},
            parser_schema_valid=False,
            failure_type="clean",
            expected_slot="intent",
            notes="Greeting; no booking slots established yet.",
        )
    )
    specs.append(
        _turn(
            call_id="call-0003",
            turn_index=1,
            expected_transcript="",
            hypothesis="",
            expected_slots={},
            parser_output={},
            parser_schema_valid=False,
            failure_type="unusable_audio",
            expected_slot=None,
            audio_kind="silence",
            case_id="silence_exact",
            notes="Exactly-zero PCM; STT and parser both returned no output.",
        )
    )
    specs.append(
        _turn(
            call_id="call-0003",
            turn_index=2,
            expected_transcript="",
            hypothesis="Thank you for calling.",
            expected_slots={},
            parser_output={},
            parser_schema_valid=False,
            failure_type="unusable_audio",
            expected_slot=None,
            audio_kind="silence",
            case_id="silence_hallucination",
            low_conf_words=("thank", "you", "for", "calling"),
            notes="Exactly-zero PCM with hallucinated STT text; no booking slot.",
        )
    )
    specs.append(
        _turn(
            call_id="call-0003",
            turn_index=3,
            expected_transcript="",
            hypothesis="Hello?",
            expected_slots={},
            parser_output={},
            parser_schema_valid=False,
            failure_type="unusable_audio",
            expected_slot=None,
            audio_kind="noise",
            case_id="line_noise",
            low_conf_words=("hello",),
            notes="Deterministic line hiss and crackle; no usable speech.",
        )
    )
    specs.append(
        _turn(
            call_id="call-0003",
            turn_index=4,
            expected_transcript=(
                "Sorry, the line dropped. Check in December twentieth for Jordan Example, "
                "one guest."
            ),
            hypothesis=(
                "Sorry, the line dropped. Check in December twentieth for Jordan Example, "
                "one guest."
            ),
            expected_slots=jordan,
            parser_output=jordan,
            parser_schema_valid=True,
            failure_type="clean",
            expected_slot="check_in",
            notes="Recovery after unusable audio; slots parsed correctly.",
        )
    )
    suite_jordan = _with(jordan, room_type="suite")
    specs.append(
        _turn(
            call_id="call-0003",
            turn_index=5,
            expected_transcript="Make it a suite, please.",
            hypothesis="Make it a suite, please.",
            expected_slots=suite_jordan,
            parser_output=suite_jordan,
            parser_schema_valid=True,
            failure_type="both_fine",
            expected_slot="room_type",
            notes="Room upgrade verified in the final record.",
            outcome_status="confirmed",
        )
    )

    # call-0004: native 8 kHz telephony variant, preserved as-is (context §16).
    rate_8k = 8000
    morgan = _booking(
        guest_name="Morgan Mock",
        check_in="2027-01-08",
        check_out="2027-01-10",
        guests=2,
        room_type="standard",
        email="morgan.mock@example.com",
        phone="+1 555 010 0004",
    )
    specs.append(
        _turn(
            call_id="call-0004",
            turn_index=0,
            expected_transcript="Hello?",
            hypothesis="Hello?",
            expected_slots={},
            parser_output={},
            parser_schema_valid=False,
            failure_type="clean",
            expected_slot="intent",
            sample_rate_hz=rate_8k,
            notes="8 kHz telephony call; audio is stored at its original rate.",
        )
    )
    specs.append(
        _turn(
            call_id="call-0004",
            turn_index=1,
            expected_transcript="Check in the eighth of January for Morgan Mock, two guests.",
            hypothesis="Check in the eighth of January for Morgan Mock, two guests.",
            expected_slots=morgan,
            parser_output=morgan,
            parser_schema_valid=True,
            failure_type="clean",
            expected_slot="check_in",
            sample_rate_hz=rate_8k,
            notes="8 kHz booking details parsed correctly.",
        )
    )
    deluxe_morgan = _with(morgan, room_type="deluxe")
    specs.append(
        _turn(
            call_id="call-0004",
            turn_index=2,
            expected_transcript="A deluxe room, please.",
            hypothesis="A deluxe room, please.",
            expected_slots=deluxe_morgan,
            parser_output=deluxe_morgan,
            parser_schema_valid=True,
            failure_type="both_fine",
            expected_slot="room_type",
            sample_rate_hz=rate_8k,
            notes="8 kHz room-type slot verified correct.",
        )
    )
    four_guests_morgan = _with(deluxe_morgan, guests=4)
    specs.append(
        _turn(
            call_id="call-0004",
            turn_index=3,
            expected_transcript="Make it four guests.",
            hypothesis="Make it two guests.",
            expected_slots=four_guests_morgan,
            parser_output=deluxe_morgan,
            parser_schema_valid=True,
            failure_type="stt_wrong",
            expected_slot="guests",
            sample_rate_hz=rate_8k,
            case_id="misheard_guests",
            low_conf_words=("two",),
            notes="8 kHz telephony: 'four' misheard as 'two'; slot error traces to STT.",
        )
    )
    three_nights_morgan = _with(four_guests_morgan, check_out="2027-01-11")
    specs.append(
        _turn(
            call_id="call-0004",
            turn_index=4,
            expected_transcript="And three nights.",
            hypothesis="And three nights.",
            expected_slots=three_nights_morgan,
            parser_output=three_nights_morgan,
            parser_schema_valid=True,
            failure_type="clean",
            expected_slot="check_out",
            sample_rate_hz=rate_8k,
            notes="Duration parsed correctly at 8 kHz.",
        )
    )
    specs.append(
        _turn(
            call_id="call-0004",
            turn_index=5,
            expected_transcript="Yes, confirm.",
            hypothesis="Yes, confirm.",
            expected_slots=three_nights_morgan,
            parser_output=three_nights_morgan,
            parser_schema_valid=True,
            failure_type="clean",
            expected_slot="intent",
            sample_rate_hz=rate_8k,
            notes="8 kHz call confirmed after the corrected guest count.",
            outcome_status="confirmed",
        )
    )

    # call-0005: consent edge case plus a parser date error.
    casey = _booking(
        guest_name="Casey Placeholder",
        check_in="2027-02-02",
        check_out="2027-02-04",
        guests=2,
        room_type="suite",
        email="casey.placeholder@example.com",
        phone="+1 555 010 0005",
    )
    specs.append(
        _turn(
            call_id="call-0005",
            turn_index=0,
            expected_transcript="Hi, I'd like to book a suite in February.",
            hypothesis="Hi, I'd like to book a suite in February.",
            expected_slots={},
            parser_output={},
            parser_schema_valid=False,
            failure_type="clean",
            expected_slot="intent",
            notes="Greeting; no booking slots established yet.",
        )
    )
    specs.append(
        _turn(
            call_id="call-0005",
            turn_index=1,
            expected_transcript=(
                "Check in February second for Casey Placeholder, two guests, a suite."
            ),
            hypothesis="Check in February second for Casey Placeholder, two guests, a suite.",
            expected_slots=casey,
            parser_output=casey,
            parser_schema_valid=True,
            failure_type="clean",
            expected_slot="check_in",
            notes="Initial slot fill for the consent case.",
        )
    )
    specs.append(
        _turn(
            call_id="call-0005",
            turn_index=2,
            expected_transcript="My email is casey.placeholder@example.com.",
            hypothesis="My email is casey.placeholder@example.com.",
            expected_slots=casey,
            parser_output=casey,
            parser_schema_valid=True,
            failure_type="clean",
            expected_slot="contact.email",
            training_consent=False,
            case_id="no_training_consent",
            notes="Recording consented, training consent withheld: exclude from training sets.",
        )
    )
    four_nights_casey = _with(casey, check_out="2027-02-06")
    specs.append(
        _turn(
            call_id="call-0005",
            turn_index=3,
            expected_transcript="I'll stay four nights.",
            hypothesis="I'll stay four nights.",
            expected_slots=four_nights_casey,
            parser_output=_with(casey, check_out="2027-02-05"),
            parser_schema_valid=True,
            failure_type="parser_wrong",
            expected_slot="check_out",
            case_id="nights_misparsed",
            notes="Parser calculated three nights instead of the stated four.",
        )
    )
    specs.append(
        _turn(
            call_id="call-0005",
            turn_index=4,
            expected_transcript="You can reach me at plus one five five five, zero one zero, "
            "zero zero zero five.",
            hypothesis="You can reach me at plus one five five five, zero one zero, "
            "zero zero zero five.",
            expected_slots=four_nights_casey,
            parser_output=four_nights_casey,
            parser_schema_valid=True,
            failure_type="both_fine",
            expected_slot="contact.phone",
            notes="Synthetic reserved-range phone number parsed correctly.",
        )
    )
    specs.append(
        _turn(
            call_id="call-0005",
            turn_index=5,
            expected_transcript="Confirm the booking, please.",
            hypothesis="Confirm the booking, please.",
            expected_slots=four_nights_casey,
            parser_output=four_nights_casey,
            parser_schema_valid=True,
            failure_type="clean",
            expected_slot="intent",
            notes="Final record reflects the staff-corrected check-out date.",
            outcome_status="edited_by_staff",
        )
    )

    # call-0006: general-capability mixture of noise and parser errors.
    riley = _booking(
        guest_name="Riley Synthetic",
        check_in="2027-03-03",
        check_out="2027-03-05",
        guests=2,
        room_type="standard",
        email="riley.synthetic@example.com",
        phone="+1 555 010 0006",
    )
    specs.append(
        _turn(
            call_id="call-0006",
            turn_index=0,
            expected_transcript="Good morning, I have a booking query.",
            hypothesis="Good morning, I have a booking query.",
            expected_slots={},
            parser_output={},
            parser_schema_valid=False,
            failure_type="clean",
            expected_slot="intent",
            notes="Greeting; no booking slots established yet.",
        )
    )
    specs.append(
        _turn(
            call_id="call-0006",
            turn_index=1,
            expected_transcript="Check in March third for Riley Synthetic, two guests, standard.",
            hypothesis="Check in March third for Riley Synthetic, two guests, standard.",
            expected_slots=riley,
            parser_output=riley,
            parser_schema_valid=True,
            failure_type="clean",
            expected_slot="check_in",
            notes="Initial slot fill parsed correctly.",
        )
    )
    specs.append(
        _turn(
            call_id="call-0006",
            turn_index=2,
            expected_transcript="",
            hypothesis="Mm-hmm?",
            expected_slots={},
            parser_output={},
            parser_schema_valid=False,
            failure_type="unusable_audio",
            expected_slot=None,
            audio_kind="noise",
            case_id="line_noise_burst",
            low_conf_words=("mm-hmm",),
            notes="Short deterministic noise burst; backchannel is not a booking slot.",
        )
    )
    deluxe_riley = _with(riley, room_type="deluxe")
    missing_guests_riley = {key: value for key, value in deluxe_riley.items() if key != "guests"}
    specs.append(
        _turn(
            call_id="call-0006",
            turn_index=3,
            expected_transcript="Two guests and a deluxe room, please.",
            hypothesis="Two guests and a deluxe room, please.",
            expected_slots=deluxe_riley,
            parser_output=missing_guests_riley,
            parser_schema_valid=False,
            failure_type="parser_wrong",
            expected_slot="guests",
            case_id="schema_invalid_missing_field",
            notes="Parser dropped the required guests slot; schema_valid=false.",
        )
    )
    three_nights_riley = _with(deluxe_riley, check_out="2027-03-06")
    specs.append(
        _turn(
            call_id="call-0006",
            turn_index=4,
            expected_transcript="And three nights.",
            hypothesis="And three nights.",
            expected_slots=three_nights_riley,
            parser_output=three_nights_riley,
            parser_schema_valid=True,
            failure_type="both_fine",
            expected_slot="check_out",
            notes="Correct duration after an earlier parser error.",
        )
    )
    specs.append(
        _turn(
            call_id="call-0006",
            turn_index=5,
            expected_transcript="That's right, confirm.",
            hypothesis="That's right, confirm.",
            expected_slots=three_nights_riley,
            parser_output=three_nights_riley,
            parser_schema_valid=True,
            failure_type="clean",
            expected_slot="intent",
            notes="Final confirmation for the mixed-signal call.",
            outcome_status="confirmed",
        )
    )
    return specs


def _fade_envelope(index: int, sample_count: int, rate: int, pulses: int) -> float:
    """Return a bounded amplitude envelope with fades and slow modulation."""
    fade_in = min(1.0, index / max(1, int(0.02 * rate)))
    fade_out = min(1.0, (sample_count - 1 - index) / max(1, int(0.03 * rate)))
    modulation = 0.75 + 0.25 * math.sin(2.0 * math.pi * pulses * index / sample_count)
    return max(0.0, min(fade_in, fade_out) * modulation)


def _noise_samples(sample_rate_hz: int, duration_ms: int, seed: int) -> list[int]:
    """Return deterministic band-limited hiss with occasional crackles."""
    rng = random.Random(seed + 101)
    sample_count = sample_rate_hz * duration_ms // 1000
    filtered = 0.0
    samples: list[int] = []
    for index in range(sample_count):
        white = rng.uniform(-1.0, 1.0)
        filtered = 0.92 * filtered + 0.08 * white
        crackle = 0.55 if rng.random() < 0.004 else 0.0
        envelope = _fade_envelope(index, sample_count, sample_rate_hz, 2)
        samples.append(_to_pcm16((filtered * 0.4 + crackle) * envelope * 0.4))
    return samples


def _telephony_bandpass(samples: list[int], sample_rate_hz: int) -> list[int]:
    """Apply a light 300-3400 Hz one-pole bandpass to emulate a telephony handset."""
    step = 1.0 / sample_rate_hz
    alpha_hp = (2.0 * math.pi * 300.0 * step) / (1.0 + 2.0 * math.pi * 300.0 * step)
    alpha_lp = (2.0 * math.pi * 3400.0 * step) / (1.0 + 2.0 * math.pi * 3400.0 * step)
    hp_previous_input = 0.0
    hp_previous_output = 0.0
    lp_previous = 0.0
    output: list[int] = []
    for sample in samples:
        value = sample / 32768.0
        high_passed = alpha_hp * (hp_previous_output + value - hp_previous_input)
        hp_previous_input, hp_previous_output = value, high_passed
        lp_previous += alpha_lp * (high_passed - lp_previous)
        output.append(_to_pcm16(lp_previous))
    return output


def _to_pcm16(value: float) -> int:
    """Clamp a float in [-1, 1] to a signed 16-bit PCM sample."""
    return int(max(-32768, min(32767, round(value * 32767))))


def _render_audio(
    spec: TurnSpec, duration_ms: int, seed: int, tts_backend: TtsBackend
) -> list[int]:
    """Render PCM for one turn, using the backend only for speech-like carriers."""
    sample_count = spec.sample_rate_hz * duration_ms // 1000
    if spec.audio_kind == "silence":
        samples = [0] * sample_count
    elif spec.audio_kind == "noise":
        samples = _noise_samples(spec.sample_rate_hz, duration_ms, seed)
    else:
        samples = tts_backend.synthesize(
            text=spec.expected_transcript,
            sample_rate_hz=spec.sample_rate_hz,
            duration_ms=duration_ms,
            seed=seed,
        )
    if len(samples) != sample_count:
        raise ValueError(
            f"TTS backend returned {len(samples)} samples for {spec.call_id} "
            f"turn {spec.turn_index}; expected {sample_count}"
        )
    if spec.sample_rate_hz == 8000:
        samples = _telephony_bandpass(samples, spec.sample_rate_hz)
    return samples


def _write_wav(path: Path, samples: list[int], sample_rate_hz: int) -> None:
    """Write mono 16-bit PCM WAV using only the stdlib wave and struct modules."""
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def _seed(call_id: str, turn_index: int) -> int:
    """Derive a stable per-turn seed from the call id and turn index."""
    call_number = int(call_id.rsplit("-", 1)[1])
    return call_number * 1000 + turn_index


def _timestamp(call_id: str, turn_index: int) -> str:
    """Return a deterministic RFC 3339 timestamp for a synthetic turn."""
    call_number = int(call_id.rsplit("-", 1)[1])
    hour = 10 + (call_number - 1) // 6
    minute = ((call_number - 1) % 6) * 8 + turn_index
    return f"2026-11-24T{hour:02d}:{minute:02d}:00Z"


def _build_words(
    hypothesis: str, duration_ms: int, low_conf_words: tuple[str, ...], seed: int
) -> list[dict[str, Any]]:
    """Build deterministic word-level timings and confidences for an STT hypothesis."""
    tokens = hypothesis.split()
    if not tokens:
        return []
    rng = random.Random(seed + 17)
    span = max(duration_ms - 40, 1)
    step = max(20, span // len(tokens))
    lowered = {word.lower() for word in low_conf_words}
    words: list[dict[str, Any]] = []
    for index, token in enumerate(tokens):
        start_ms = 20 + index * step
        end_ms = min(duration_ms, start_ms + max(20, int(step * 0.85)))
        confidence = 0.96 - (index % 4) * 0.01 + rng.uniform(-0.02, 0.02)
        if token.strip(".,!?").lower() in lowered:
            confidence = 0.41
        words.append(
            {
                "w": token,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "conf": round(max(0.05, min(0.99, confidence)), 2),
            }
        )
    return words


def _decoder_stats(spec: TurnSpec, seed: int) -> dict[str, float]:
    """Return deterministic decoder signals that match the labelled failure type."""
    rng = random.Random(seed + 31)
    average_by_type = {
        "clean": -0.18,
        "both_fine": -0.31,
        "stt_wrong": -0.68,
        "parser_wrong": -0.24,
        "unusable_audio": -1.40,
    }
    if spec.audio_kind == "silence":
        no_speech_prob = 0.82 if spec.hypothesis else 0.97
    elif spec.audio_kind == "noise":
        no_speech_prob = 0.71
    elif spec.failure_type == "stt_wrong":
        no_speech_prob = 0.03
    else:
        no_speech_prob = 0.002
    return {
        "avg_logprob": round(average_by_type[spec.failure_type] + rng.uniform(-0.03, 0.03), 3),
        "compression_ratio": round(1.35 + rng.uniform(-0.05, 0.15), 3),
        "no_speech_prob": no_speech_prob,
    }


def _build_event(spec: TurnSpec, audio_uri: str, duration_ms: int, seed: int) -> dict[str, Any]:
    """Build one turn event payload conforming to turn_event.schema.json 1.0.0."""
    event: dict[str, Any] = {
        "event_id": str(
            uuid.uuid5(EVENT_NAMESPACE, f"voiceloop:fixture:{spec.call_id}:{spec.turn_index}")
        ),
        "call_id": spec.call_id,
        "turn_index": spec.turn_index,
        "ts": _timestamp(spec.call_id, spec.turn_index),
        "consent": {"recording": True, "training": spec.training_consent},
        "audio": {
            "uri": audio_uri,
            "sample_rate_hz": spec.sample_rate_hz,
            "channels": 1,
            "duration_ms": duration_ms,
        },
        "stt": {
            "model_id": TTS_MODEL_ID,
            "model_version": MODEL_VERSION,
            "text": spec.hypothesis,
            "words": _build_words(spec.hypothesis, duration_ms, spec.low_conf_words, seed),
            "decoder_stats": _decoder_stats(spec, seed),
        },
        "parser": {
            "model_id": PARSER_MODEL_ID,
            "model_version": MODEL_VERSION,
            "input_text": spec.hypothesis,
            "output": spec.parser_output,
            "schema_valid": spec.parser_schema_valid,
            "logprob": -0.28 if spec.parser_schema_valid else -1.85,
        },
        "dialogue_context": {
            "expected_slot": spec.expected_slot,
            "reference_date": REFERENCE_DATE,
            "timezone": "Europe/London",
            "prior_turns_ref": (
                f"tests/fixtures/contexts/{spec.call_id}.jsonl" if spec.turn_index > 0 else None
            ),
        },
    }
    if spec.outcome_status is not None:
        event["outcome"] = {
            "status": spec.outcome_status,
            "final_record": spec.expected_slots,
            "updated_at": _timestamp(spec.call_id, spec.turn_index + 1),
        }
    return event


def _write_contexts(output_dir: Path, specs: list[TurnSpec]) -> None:
    """Write one synthetic prior-turn transcript history per call."""
    by_call: dict[str, list[TurnSpec]] = defaultdict(list)
    for spec in specs:
        by_call[spec.call_id].append(spec)
    for call_id, call_specs in by_call.items():
        lines = [
            json.dumps(
                {"turn_index": spec.turn_index, "hypothesis": spec.hypothesis},
                sort_keys=True,
            )
            for spec in sorted(call_specs, key=lambda item: item.turn_index)
        ]
        path = output_dir / "contexts" / f"{call_id}.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _label_record(
    spec: TurnSpec, event_uri: str, audio_uri: str, duration_ms: int
) -> dict[str, Any]:
    """Build one labelled manifest row for a fixture turn."""
    return {
        "call_id": spec.call_id,
        "turn_index": spec.turn_index,
        "event_path": event_uri,
        "audio_path": audio_uri,
        "expected_transcript": spec.expected_transcript,
        "expected_slots": spec.expected_slots,
        "expected_outcome": spec.outcome_status,
        "failure_type": spec.failure_type,
        "consent": {"recording": True, "training": spec.training_consent},
        "sample_rate_hz": spec.sample_rate_hz,
        "audio_kind": spec.audio_kind,
        "case_id": spec.case_id,
        "synthetic": True,
        "notes": spec.notes,
    }


def generate_fixture_set(
    output_dir: Path = FIXTURES_DIR,
    tts_backend: TtsBackend | None = None,
) -> list[dict[str, Any]]:
    """Regenerate every fixture file under ``output_dir`` and return label rows.

    Args:
        output_dir: Destination directory; defaults to tests/fixtures.
        tts_backend: Optional speech-carrier backend; defaults to the committed
            deterministic tone backend. All outputs are labelled synthetic.

    Returns:
        The label rows written to ``turns.jsonl``.

    Raises:
        ValueError: If a backend returns the wrong number of samples.

    """
    backend: TtsBackend = tts_backend if tts_backend is not None else DeterministicToneBackend()
    specs = build_turn_specs()
    for subdir in ("audio", "events", "contexts"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    _write_contexts(output_dir, specs)

    records: list[dict[str, Any]] = []
    for spec in specs:
        seed = _seed(spec.call_id, spec.turn_index)
        duration_ms = 240 + (seed % 5) * 40
        samples = _render_audio(spec, duration_ms, seed, backend)
        stem = f"turn-{spec.turn_index:03d}"
        audio_rel = f"tests/fixtures/audio/{spec.call_id}/{stem}.wav"
        event_rel = f"tests/fixtures/events/{spec.call_id}/{stem}.json"

        audio_path = output_dir / "audio" / spec.call_id / f"{stem}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        _write_wav(audio_path, samples, spec.sample_rate_hz)

        event = _build_event(spec, audio_rel, duration_ms, seed)
        validate(event, TURN_EVENT_SCHEMA)
        if spec.expected_slots:
            validate_booking(spec.expected_slots, schema_path(BOOKING_SCHEMA))
        if spec.parser_schema_valid:
            validate_booking(spec.parser_output, schema_path(BOOKING_SCHEMA))

        event_path = output_dir / "events" / spec.call_id / f"{stem}.json"
        event_path.parent.mkdir(parents=True, exist_ok=True)
        event_path.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        records.append(_label_record(spec, event_rel, audio_rel, duration_ms))

    (output_dir / "turns.jsonl").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
    summary = {
        "generator_version": GENERATOR_VERSION,
        "synthetic": True,
        "tts_backend": SYNTHETIC_BACKEND_NAME if tts_backend is None else "custom-hook",
        "turn_count": len(records),
        "call_ids": sorted({spec.call_id for spec in specs}),
        "sample_rates_hz": sorted({spec.sample_rate_hz for spec in specs}),
        "failure_types": sorted({spec.failure_type for spec in specs}),
        "case_ids": sorted(spec.case_id for spec in specs if spec.case_id),
        "note": "WAVs are tones, noise or exact silence; they contain no real speech.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return records


def load_tts_backend(dotted_path: str) -> TtsBackend:
    """Resolve a ``module:attribute`` TTS backend for optional real-TTS use.

    Args:
        dotted_path: Import path such as ``my_package.tts:MyBackend``.

    Returns:
        An object implementing :class:`TtsBackend`.

    Raises:
        ValueError: If the path is malformed.
        TypeError: If the resolved object has no callable ``synthesize`` method.

    """
    module_name, separator, attribute = dotted_path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("tts_backend must use 'module:attribute' form")
    candidate: Any = getattr(importlib.import_module(module_name), attribute)
    if isinstance(candidate, type):  # allow module:Class as well as module:instance
        candidate = candidate()
    if not callable(getattr(candidate, "synthesize", None)):
        raise TypeError(f"{dotted_path} does not implement TtsBackend.synthesize")
    return cast("TtsBackend", candidate)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for regenerating the synthetic fixture set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIXTURES_DIR,
        help="Directory to write fixtures into (default: tests/fixtures).",
    )
    parser.add_argument(
        "--tts-backend",
        default=None,
        help="Optional module:attribute implementing TtsBackend (default: deterministic tones).",
    )
    args = parser.parse_args(argv)
    backend = load_tts_backend(args.tts_backend) if args.tts_backend else None
    records = generate_fixture_set(output_dir=args.output_dir, tts_backend=backend)
    print(f"Generated {len(records)} synthetic turns in {args.output_dir}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
