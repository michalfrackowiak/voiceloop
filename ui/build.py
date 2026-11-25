"""Regenerate the static review UI artifacts from read-only contracts.

Everything written here is deterministic and stays inside ``ui/``. The browser
loads only inline JSON, never a network or object-store URI.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .constants import (
    GENERATED_DIR,
    GO_NO_GO_PATH,
    INDEX_TEMPLATE_PATH,
    SCHEMA_PATH,
    SYNTHETIC_LABEL,
    UI_DIR,
    USABILITY_ACCEPTANCE,
    USABILITY_STATUS,
)
from .errors import UiValidationError
from .form import build_form_descriptor
from .hashes import verify_read_only_inputs
from .privacy import assert_no_sensitive_text
from .review import build_review_state, load_review_items


def _json_block(payload: Mapping[str, Any] | list[Any]) -> str:
    """Serialize payload for inline embedding, escaping script-closing markup."""
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False).replace("<", "\\u003c")


def render_index(states: list[dict[str, Any]], decision: Mapping[str, Any]) -> str:
    """Render the static HTML shell with embedded, deterministic state.

    Args:
        states: One review state per synthetic item.
        decision: Go/no-go report object from the P2-04 metrics input.

    Returns:
        A complete HTML document string.

    Raises:
        UiValidationError: If a required template placeholder is missing.

    """
    template = INDEX_TEMPLATE_PATH.read_text(encoding="utf-8")
    replacements = {
        "{{SYNTHETIC_LABEL}}": SYNTHETIC_LABEL,
        "{{USABILITY_STATUS}}": USABILITY_STATUS,
        "{{USABILITY_ACCEPTANCE}}": USABILITY_ACCEPTANCE,
        "{{STATE_JSON}}": _json_block(states),
        "{{GONOGO_JSON}}": _json_block(decision),
    }
    for placeholder, value in replacements.items():
        if placeholder not in template:
            raise UiValidationError(f"missing template placeholder: {placeholder}")
        template = template.replace(placeholder, value)
    return template


def build_artifacts(
    output_dir: Path = GENERATED_DIR,
    decision: Mapping[str, Any] | None = None,
    index_path: Path | None = None,
) -> dict[str, str]:
    """Regenerate the form descriptor, review state, decision report and HTML.

    Args:
        output_dir: Directory for generated JSON artifacts.
        decision: Go/no-go report; defaults to the committed fail-closed report.
        index_path: HTML destination; defaults to ``ui/index.html``.

    Returns:
        Mapping from artifact name to written path.

    Raises:
        UiValidationError: If hashes or embedded payloads fail the privacy screen.

    """
    verification = verify_read_only_inputs()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    descriptor = build_form_descriptor(schema, verification.booking_schema_sha256)
    items = load_review_items()
    states = [build_review_state(item, descriptor) for item in items]
    assert_no_sensitive_text(states)
    if decision is None:
        if not GO_NO_GO_PATH.exists():
            raise UiValidationError("go/no-go report missing; generate it before --build")
        decision = json.loads(GO_NO_GO_PATH.read_text(encoding="utf-8"))
    assert_no_sensitive_text(decision)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for name, payload in (
        ("booking.form.json", descriptor.to_dict()),
        ("review_state.json", states),
        ("go_no_go_report.json", dict(decision)),
    ):
        path = output_dir / name
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written[name] = str(path)
    index = index_path if index_path is not None else UI_DIR / "index.html"
    index.write_text(render_index(states, decision), encoding="utf-8")
    written["index.html"] = str(index)
    return written
