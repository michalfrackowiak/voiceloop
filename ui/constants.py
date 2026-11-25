"""Fixed labels, pinned hashes and read-only contract locations for P4-03.

The pinned hashes are verification anchors only. All referenced source files are
read-only: this task owns ``ui/`` and never edits schemas, fixtures or the sealed
golden stand-in.
"""

from __future__ import annotations

from pathlib import Path

UI_DIR: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = UI_DIR.parent

SCHEMA_PATH: Path = REPO_ROOT / "schemas" / "booking.schema.json"
OPENAPI_PATH: Path = REPO_ROOT / "schemas" / "review_adapter.openapi.yaml"
FIXTURES_PATH: Path = REPO_ROOT / "tests" / "fixtures" / "turns.jsonl"
GOLDEN_MANIFEST_PATH: Path = (
    REPO_ROOT / "services" / "datasets" / "golden" / "artifacts" / "golden.v1.json"
)
GENERATED_DIR: Path = UI_DIR / "data"
FIXTURE_ITEM_PATH: Path = UI_DIR / "fixtures" / "synthetic_items.json"
INDEX_TEMPLATE_PATH: Path = UI_DIR / "index.template.html"
GO_NO_GO_PATH: Path = GENERATED_DIR / "go_no_go_report.json"

SYNTHETIC_LABEL: str = "SYNTHETIC/FIXTURE STAND-IN - NOT A HUMAN USABILITY RESULT"
USABILITY_STATUS: str = "NOT DONE"
USABILITY_ACCEPTANCE: str = (
    "OPEN - RELEASE BLOCKER: an independent usability test must show time per item "
    "no worse than the adapter-based tool"
)
INTERNAL_STATE_VERSION: str = "internal-1.0.0"

EXPECTED_FIXTURE_SHA256: str = "aca23a303954d035245a97f78319df3694d63611b717a2ff2131bc2e0a4453fd"
EXPECTED_GOLDEN_CONTENT_HASH: str = (
    "662cdaf43ee2d5fde03faf5e69fde196114d4f312406d94c00bf53714d03da8e"
)
EXPECTED_BOOKING_SCHEMA_SHA256: str = (
    "27e9553ce1d70ba94230738a26fedd8bcf7703afe6e41d1cc8d87f55b9d382aa"
)

# Routing enum from schemas/review_adapter.openapi.yaml 1.0.0 (read-only).
ROUTING_LABELS: tuple[str, ...] = (
    "stt_wrong",
    "parser_wrong",
    "both_fine",
    "unusable_audio",
)

ALLOWED_SPEEDS: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
LOW_CONFIDENCE_CEILING: float = 0.60
WAVEFORM_BARS: int = 48

# Measured actions only. Unknown action names fail closed so subjective or
# unverifiable labels cannot be smuggled into the go/no-go comparison.
CLICK_ACTIONS: tuple[str, ...] = (
    "play",
    "speed_change",
    "loop_toggle",
    "seek",
    "diff_toggle",
    "field_focus",
    "validation_retry",
    "routing_change",
    "second_opinion",
)

# A go/no-go decision must be derived from measurements, never from preference.
SUBJECTIVE_KEYS: frozenset[str] = frozenset(
    {
        "preference",
        "subjective",
        "subjective_score",
        "opinion",
        "gut_feel",
        "preferred_tool",
        "reviewer_liked",
        "designer_choice",
    }
)
