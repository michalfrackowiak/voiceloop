"""Synthetic review items, two-stage state, routing and second-opinion flow.

All identifiers and slot values are pseudonymous tokens. No hypothesis prose,
real transcript, contact detail, audio byte or external service is involved.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import (
    FIXTURE_ITEM_PATH,
    INTERNAL_STATE_VERSION,
    ROUTING_LABELS,
    SYNTHETIC_LABEL,
    USABILITY_ACCEPTANCE,
    USABILITY_STATUS,
)
from .errors import UiValidationError
from .form import FormDescriptor, ValidationIssue, validate_slots
from .player import DiffOp, LowConfidenceSpan, PlayerState, diff_tokens, new_player
from .privacy import assert_no_sensitive_text

ITEM_KEYS = frozenset(
    {
        "item_id",
        "call_id",
        "turn_index",
        "audio_ref",
        "duration_ms",
        "blind",
        "reviewer_id",
        "hypothesis_tokens",
        "corrected_tokens",
        "low_confidence_spans",
        "prefill_slots",
        "final_record",
    }
)
REASON_CODES = frozenset(
    {
        "review_not_completed",
        "unknown_item_id",
        "missing_item_id",
        "missing_primary_reviewer_id",
        "missing_second_reviewer_id",
        "same_reviewer",
        "missing_approval_id",
        "unrecognised_approval",
        "slot_validation_failed",
        "routing_missing",
    }
)


@dataclass(frozen=True)
class ReviewItem:
    """One pseudonymous, fixture-only review item."""

    item_id: str
    call_id: str
    turn_index: int
    audio_ref: str
    duration_ms: int
    blind: bool
    reviewer_id: str
    hypothesis_tokens: tuple[str, ...]
    corrected_tokens: tuple[str, ...]
    low_confidence_spans: tuple[LowConfidenceSpan, ...]
    prefill_slots: dict[str, Any]
    final_record: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable item with no audio bytes or PII."""
        return {
            "item_id": self.item_id,
            "call_id": self.call_id,
            "turn_index": self.turn_index,
            "audio_ref": self.audio_ref,
            "duration_ms": self.duration_ms,
            "blind": self.blind,
            "reviewer_id": self.reviewer_id,
            "hypothesis_tokens": list(self.hypothesis_tokens),
            "corrected_tokens": list(self.corrected_tokens),
            "low_confidence_spans": [span.to_dict() for span in self.low_confidence_spans],
            "prefill_slots": {} if self.blind else self.prefill_slots,
            "final_record": self.final_record,
        }


@dataclass(frozen=True)
class ReviewSubmission:
    """A completed fake submission keyed by pseudonymous IDs only."""

    item_id: str
    reviewer_id: str
    routing: str
    slots: dict[str, Any]
    completed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable submission without free text."""
        return {
            "item_id": self.item_id,
            "reviewer_id": self.reviewer_id,
            "routing": self.routing,
            "slots": self.slots,
            "completed": self.completed,
        }


@dataclass(frozen=True)
class SecondOpinionRequest:
    """Request inputs that must all be present before a second opinion opens."""

    item_id: str
    primary_reviewer_id: str
    second_reviewer_id: str
    approval_id: str | None


@dataclass(frozen=True)
class SecondOpinionState:
    """Fail-closed result of a second-opinion eligibility check."""

    allowed: bool
    reasons: tuple[str, ...]
    audit_event: dict[str, str] | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable eligibility state."""
        return {
            "allowed": self.allowed,
            "reasons": list(self.reasons),
            "audit_event": self.audit_event,
        }


def load_review_items(path: Path = FIXTURE_ITEM_PATH) -> tuple[ReviewItem, ...]:
    """Load and screen the authored synthetic item fixtures.

    Args:
        path: Optional alternative fixture path for tests.

    Returns:
        Tuple of screened review items in file order.

    Raises:
        UiValidationError: On unknown keys, invalid IDs, malformed spans or
            any PII-looking string.

    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert_no_sensitive_text(payload)
    if not isinstance(payload, list):
        raise UiValidationError("synthetic items must be a JSON array")
    items: list[ReviewItem] = []
    seen: set[str] = set()
    for raw in payload:
        if not isinstance(raw, Mapping) or set(raw) != ITEM_KEYS:
            raise UiValidationError("synthetic item has unknown or missing keys")
        item_id = str(raw["item_id"])
        if not item_id or item_id in seen:
            raise UiValidationError("synthetic item IDs must be unique and non-empty")
        seen.add(item_id)
        span_records: list[LowConfidenceSpan] = []
        raw_spans = raw["low_confidence_spans"]
        if not isinstance(raw_spans, list):
            raise UiValidationError("low_confidence_spans must be a list")
        for span in raw_spans:
            if not isinstance(span, Mapping):
                raise UiValidationError("span must be an object")
            span_records.append(
                LowConfidenceSpan(
                    start_ms=int(span["start_ms"]),
                    end_ms=int(span["end_ms"]),
                    token_id=str(span["token_id"]),
                    confidence=float(span["confidence"]),
                )
            )
        hypothesis = raw["hypothesis_tokens"]
        corrected = raw["corrected_tokens"]
        prefill = raw["prefill_slots"]
        final_record = raw["final_record"]
        if not isinstance(hypothesis, list) or not isinstance(corrected, list):
            raise UiValidationError("token lists must be arrays")
        if not isinstance(prefill, Mapping) or not isinstance(final_record, Mapping):
            raise UiValidationError("slot objects must be mappings")
        items.append(
            ReviewItem(
                item_id=item_id,
                call_id=str(raw["call_id"]),
                turn_index=int(raw["turn_index"]),
                audio_ref=str(raw["audio_ref"]),
                duration_ms=int(raw["duration_ms"]),
                blind=bool(raw["blind"]),
                reviewer_id=str(raw["reviewer_id"]),
                hypothesis_tokens=tuple(str(token) for token in hypothesis),
                corrected_tokens=tuple(str(token) for token in corrected),
                low_confidence_spans=tuple(span_records),
                prefill_slots=dict(prefill),
                final_record=dict(final_record),
            )
        )
    return tuple(items)


def _flatten_slots(values: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one nesting level of slot values for side-panel comparison."""
    flat: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, Mapping):
            for child, child_value in value.items():
                flat[f"{key}.{child}"] = child_value
        else:
            flat[str(key)] = value
    return flat


def record_diff(values: Mapping[str, Any], final_record: Mapping[str, Any]) -> list[dict[str, str]]:
    """Compare slot values to the booking side panel field by field.

    Returns:
        Sorted records with ``same``, ``different`` or ``missing`` status.
        Values themselves are never included.

    """
    left = _flatten_slots(values)
    right = _flatten_slots(final_record)
    rows: list[dict[str, str]] = []
    for path in sorted(set(left) | set(right)):
        if path not in left or path not in right:
            status = "missing"
        elif left[path] == right[path]:
            status = "same"
        else:
            status = "different"
        rows.append({"field": path, "status": status})
    return rows


def build_review_state(item: ReviewItem, descriptor: FormDescriptor) -> dict[str, Any]:
    """Assemble one deterministic UI state document from an item and form.

    Blind items receive no prefill; routing starts unselected and second
    opinion starts blocked until a completed submission and approval exist.
    """
    slots = {} if item.blind else item.prefill_slots
    issues: tuple[ValidationIssue, ...] = validate_slots(descriptor, slots)
    player: PlayerState = new_player(item.item_id, item.duration_ms, item.low_confidence_spans)
    diff: tuple[DiffOp, ...] = diff_tokens(item.hypothesis_tokens, item.corrected_tokens)
    second = evaluate_second_opinion(
        SecondOpinionRequest(item.item_id, item.reviewer_id, "", None),
        {},
        frozenset(),
    )
    return {
        "contract_version": INTERNAL_STATE_VERSION,
        "label": SYNTHETIC_LABEL,
        "usability_status": USABILITY_STATUS,
        "usability_acceptance": USABILITY_ACCEPTANCE,
        "item": item.to_dict(),
        "player": player.to_dict(),
        "diff": [operation.to_dict() for operation in diff],
        "form": descriptor.to_dict(),
        "slot_issues": [issue.to_dict() for issue in issues],
        "final_record": item.final_record,
        "record_diff": record_diff(slots, item.final_record),
        "routing": {"labels": list(ROUTING_LABELS), "selected": None},
        "blind_mode": {
            "active": item.blind,
            "prefill_withheld": item.blind,
            "prefill_slots": slots,
        },
        "second_opinion": second.to_dict(),
    }


def evaluate_second_opinion(
    request: SecondOpinionRequest,
    submissions: Mapping[str, ReviewSubmission],
    approvals: frozenset[str],
    descriptor: FormDescriptor | None = None,
) -> SecondOpinionState:
    """Decide whether a second-opinion request may proceed; fail closed.

    Args:
        request: IDs supplied by the reviewer and the approval source.
        submissions: Completed primary submissions keyed by item ID.
        approvals: Allowlisted pseudonymous approval IDs.
        descriptor: Optional generated form used to re-check submitted slots.

    Returns:
        An allowed state with a pseudonymous audit event, or a blocked state
        with deterministic reason codes and no audit event.

    """
    reasons: list[str] = []
    if not request.item_id:
        reasons.append("missing_item_id")
    if not request.primary_reviewer_id:
        reasons.append("missing_primary_reviewer_id")
    if not request.second_reviewer_id:
        reasons.append("missing_second_reviewer_id")
    if request.primary_reviewer_id and request.primary_reviewer_id == request.second_reviewer_id:
        reasons.append("same_reviewer")
    if not request.approval_id:
        reasons.append("missing_approval_id")
    elif request.approval_id not in approvals:
        reasons.append("unrecognised_approval")
    submission = submissions.get(request.item_id)
    if submission is None:
        reasons.append("unknown_item_id")
    else:
        if not submission.completed:
            reasons.append("review_not_completed")
        if submission.routing not in ROUTING_LABELS:
            reasons.append("routing_missing")
        if submission.reviewer_id != request.primary_reviewer_id:
            reasons.append("unknown_item_id")
        if descriptor is not None and validate_slots(descriptor, submission.slots):
            reasons.append("slot_validation_failed")
    sorted_reasons = tuple(sorted(set(reasons)))
    if sorted_reasons or submission is None:
        return SecondOpinionState(allowed=False, reasons=sorted_reasons, audit_event=None)
    return SecondOpinionState(
        allowed=True,
        reasons=(),
        audit_event={
            "status": "second_opinion_requested",
            "item_id": request.item_id,
            "primary_reviewer_id": request.primary_reviewer_id,
            "second_reviewer_id": request.second_reviewer_id,
            "approval_id": str(request.approval_id),
            "label": SYNTHETIC_LABEL,
        },
    )
