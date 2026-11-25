"""Dataset builder orchestration (P2-05).

Loads human-reviewed results (synthetic fixture labels for the v1 workflow),
runs the P1-09 consent/PII filters, applies golden and regression exclusion,
expands eligible items, mixes seeded replay turns, computes call-level splits
and returns everything needed for the manifest and exporters.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from voiceloop_common import ValidationError

from .examples import (
    REPO_ROOT,
    REVIEW_ROUTINGS,
    DatasetExample,
    ReviewedItem,
    expand_reviewed_item,
)
from .filters import (
    DroppedEvent,
    DropReason,
    PiiCategory,
    PiiFinding,
    detect_pii,
    filter_eligible_events,
)
from .golden import GoldenManifest, assert_sealed
from .manifest import DatasetBuild, FilterCounts
from .replay import sample_replay
from .settings import DatasetSettings
from .splits import assert_no_call_in_multiple_splits, assign_calls

CONSENT_DROP_REASONS: frozenset[DropReason] = frozenset(
    {
        DropReason.CONSENT_MISSING,
        DropReason.RECORDING_CONSENT_DENIED,
        DropReason.TRAINING_CONSENT_DENIED,
    }
)


def _resolve_path(path: str | Path, root: Path | None = None) -> Path:
    """Resolve a repo-relative path without mutating the fixture URI."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (root or REPO_ROOT) / candidate


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from disk."""
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read JSON at {path}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ValidationError(f"expected a JSON object at {path}")
    return decoded


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSON Lines into objects, skipping blank lines."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValidationError(f"cannot read JSONL at {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"invalid JSON on line {number} of {path}: {exc}") from exc
        if not isinstance(decoded, dict):
            raise ValidationError(f"line {number} of {path} must be an object")
        rows.append(decoded)
    return rows


def load_reviewed_items_from_fixtures(
    path: str | Path | None = None,
    *,
    settings: DatasetSettings | None = None,
    root: Path | None = None,
) -> tuple[ReviewedItem, ...]:
    """Build reviewed items from the read-only synthetic label manifest.

    ``failure_type`` is the reviewed signal: known failure labels are retained,
    ``clean`` and ``both_fine`` are mapped to the ``both_fine`` routing label so
    the unbiased random slice stays eligible for training (context.md §2.6).

    Args:
        path: Optional override for the label JSONL path.
        settings: Dataset settings; defaults to a fresh :class:`DatasetSettings`.
        root: Repository root for resolving repo-relative paths.

    Returns:
        Reviewed items in fixture manifest order.

    """
    config = settings or DatasetSettings()
    manifest_path = _resolve_path(path or config.fixtures_path, root)
    items: list[ReviewedItem] = []
    for row in _read_jsonl(manifest_path):
        for field in ("call_id", "turn_index", "event_path", "expected_transcript"):
            if field not in row:
                raise ValidationError(f"fixture row missing {field}: {row!r}"[:200])
        event = _read_json(_resolve_path(str(row["event_path"]), root))
        failure_type = row.get("failure_type")
        routing = failure_type if isinstance(failure_type, str) else "both_fine"
        if routing not in REVIEW_ROUTINGS:
            routing = "both_fine"
        slots = row.get("expected_slots", {})
        if not isinstance(slots, dict):
            raise ValidationError("fixture expected_slots must be an object")
        items.append(
            ReviewedItem(
                event=event,
                corrected_transcript=str(row["expected_transcript"]),
                slots=dict(slots),
                routing=routing,
                reviewer_id="synthetic-reviewer",
                failure_type=(failure_type if isinstance(failure_type, str) else "both_fine"),
                synthetic=bool(row.get("synthetic", False)),
            )
        )
    return tuple(items)


def load_golden_call_ids(
    path: str | Path | None = None,
    *,
    settings: DatasetSettings | None = None,
    root: Path | None = None,
    verify: bool = True,
) -> tuple[str, ...]:
    """Read sealed golden call IDs through the P1-06 public API.

    Args:
        path: Optional override for the sealed golden manifest path.
        settings: Dataset settings; defaults to a fresh :class:`DatasetSettings`.
        root: Repository root for resolving repo-relative paths.
        verify: When true, verify the seal before trusting its call IDs.

    Returns:
        Sorted golden call IDs. These must be excluded from every data split.

    """
    config = settings or DatasetSettings()
    manifest_path = _resolve_path(path or config.golden_manifest_path, root)
    if verify:
        manifest = assert_sealed(str(manifest_path.parent), manifest_path.stem)
    else:
        data = _read_json(manifest_path)
        try:
            manifest = GoldenManifest.from_dict(data)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"malformed golden manifest: {exc}") from exc
    return tuple(sorted(manifest.call_ids))


def _drop_record(
    event: Mapping[str, Any],
    reasons: tuple[DropReason, ...],
    findings: tuple[Any, ...] = (),
) -> DroppedEvent:
    """Build a PII-free drop record from a source event using the filter type."""
    index = event.get("turn_index", -1)
    audio = event.get("audio")
    return DroppedEvent(
        event_id=str(event.get("event_id", "")),
        call_id=str(event.get("call_id", "")),
        turn_index=index if isinstance(index, int) and not isinstance(index, bool) else -1,
        audio_uri=str(audio.get("uri", "")) if isinstance(audio, Mapping) else "",
        reasons=reasons,
        pii_findings=tuple(findings),
    )


def _card_findings(text: str) -> tuple[PiiFinding, ...]:
    """Return only card-number findings reported by the P1-09 detector.

    Args:
        text: Free-text value to screen read-only.

    Returns:
        Card findings carrying offsets only, never matched text.

    """
    return tuple(
        finding for finding in detect_pii(text) if finding.category is PiiCategory.CARD_NUMBER
    )


def _iter_json_strings(value: Any) -> Iterator[str]:
    """Yield every string nested in a JSON-like value, including mappings.

    Args:
        value: Slot object, context object or other JSON-like value.

    Yields:
        Each string value found recursively, in traversal order.

    """
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _iter_json_strings(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _iter_json_strings(nested)


def _deduplicate_findings(findings: Iterable[PiiFinding]) -> tuple[PiiFinding, ...]:
    """Return findings in first-seen order without repeats."""
    return tuple(dict.fromkeys(findings))


def _screen_reviewed_items(
    items: Sequence[ReviewedItem],
    drops: list[DroppedEvent],
) -> list[ReviewedItem]:
    """Drop items whose reviewed text or slots contain a card number.

    The event filter only sees ``stt.text``; this screens the corrected
    transcript, the original hypothesis and every string nested in reviewed
    slots using the same P1-09 ``detect_pii`` public API.

    Args:
        items: Reviewed items surviving the consent/PII event filter.
        drops: Drop-record accumulator updated for every hard-dropped item.

    Returns:
        Items safe to expand into examples.

    """
    kept: list[ReviewedItem] = []
    for item in items:
        findings: list[PiiFinding] = []
        for text in (item.corrected_transcript, item.hypothesis):
            findings.extend(_card_findings(text))
        for text in _iter_json_strings(item.slots):
            findings.extend(_card_findings(text))
        collected = _deduplicate_findings(findings)
        if collected:
            drops.append(_drop_record(item.event, (DropReason.CARD_NUMBER_PII,), collected))
        else:
            kept.append(item)
    return kept


def _example_card_findings(example: DatasetExample) -> tuple[PiiFinding, ...]:
    """Screen every free-text field that can enter an export from an example.

    This post-expansion check covers dialogue context, including
    ``prior_turns[*].hypothesis`` loaded from a context reference, as well as
    every string nested in slots and all transcript-like fields.

    Args:
        example: Expanded example to screen read-only.

    Returns:
        Card findings from all screened fields, deduplicated.

    """
    payload: dict[str, Any] = {
        "corrected_transcript": example.corrected_transcript,
        "hypothesis": example.hypothesis,
        "source_text": example.source_text,
        "target_text": example.target_text,
        "slots": example.slots,
        "dialogue_context": example.dialogue_context,
    }
    return _deduplicate_findings(
        finding for text in _iter_json_strings(payload) for finding in _card_findings(text)
    )


def _expand_screened_items(
    items: Iterable[ReviewedItem],
    *,
    settings: DatasetSettings,
    drops: list[DroppedEvent],
    replay: bool = False,
    existing_ids: set[str] | None = None,
) -> list[DatasetExample]:
    """Expand items and hard-drop any whose examples contain card PII.

    Args:
        items: Items already screened at review level.
        settings: Typed dataset settings controlling context length.
        drops: Drop-record accumulator updated for every hard-dropped item.
        replay: Whether to tag the resulting examples as replay.
        existing_ids: Example IDs already used, to prevent duplicate replay.

    Returns:
        Examples from items that passed both expansion and PII screening.

    """
    examples: list[DatasetExample] = []
    seen = existing_ids if existing_ids is not None else set()
    for item in items:
        expanded = expand_reviewed_item(item, prior_turns_limit=settings.prior_turns_limit)
        findings = _deduplicate_findings(
            finding for example in expanded for finding in _example_card_findings(example)
        )
        if findings:
            drops.append(_drop_record(item.event, (DropReason.CARD_NUMBER_PII,), findings))
            continue
        for example in expanded:
            if example.example_id in seen:
                continue
            candidate = replace(example, replay=True) if replay else example
            examples.append(candidate)
            seen.add(candidate.example_id)
    return examples


def _kept_by_filter(
    items: Sequence[ReviewedItem],
    kept_events: Sequence[Mapping[str, Any]],
) -> list[ReviewedItem]:
    """Match filter-surviving events back to reviewed items by object identity.

    ``filter_eligible_events`` returns the same read-only mappings it received,
    so identity is exact even when two candidate items share a source key
    (e.g. a re-review in the same batch).
    """
    kept_ids = {id(event) for event in kept_events}
    return [item for item in items if id(item.event) in kept_ids]


def build_dataset(
    reviewed_items: Iterable[ReviewedItem],
    *,
    settings: DatasetSettings | None = None,
    golden_call_ids: Iterable[str] = (),
    regression_call_ids: Iterable[str] = (),
    replay_pool: Iterable[ReviewedItem] = (),
) -> DatasetBuild:
    """Build a filtered, split and lineage-tracked dataset from reviewed items.

    Consent and card-number PII are enforced by calling the P1-09 filter public
    API. Because the event filter only sees ``stt.text``, reviewed transcripts,
    slot values and dialogue-context text are additionally screened with its
    ``detect_pii`` function; any card finding hard-drops the whole reviewed item
    before examples or exports can contain it. Golden and regression calls never
    reach a training split. Replay items pass the same filters.

    Args:
        reviewed_items: Human-reviewed items (fixture labels for v1).
        settings: Typed ``VOICELOOP_DATASETS_`` settings.
        golden_call_ids: Sealed golden call IDs to exclude.
        regression_call_ids: Cumulative reviewed-failure call IDs to exclude.
        replay_pool: Earlier verified turns available for seeded replay mixing.

    Returns:
        Examples, split plan, filter counts and PII-free drop records.

    """
    config = settings or DatasetSettings()
    items = tuple(reviewed_items)
    replay_items = tuple(replay_pool)
    drops: list[DroppedEvent] = []

    kept_events, event_drops = filter_eligible_events([item.event for item in items])
    drops.extend(event_drops)
    eligible = _screen_reviewed_items(_kept_by_filter(items, kept_events), drops)

    replay_events, replay_drops = filter_eligible_events([item.event for item in replay_items])
    drops.extend(replay_drops)
    replay_eligible = _screen_reviewed_items(_kept_by_filter(replay_items, replay_events), drops)

    golden = set(golden_call_ids)
    regression = set(regression_call_ids) - golden
    excluded = golden | regression
    golden_excluded = sum(1 for item in eligible if item.call_id in golden)
    regression_excluded = sum(1 for item in eligible if item.call_id in regression)
    qualified = [item for item in eligible if item.call_id not in excluded]

    examples = _expand_screened_items(qualified, settings=config, drops=drops)
    existing_ids = {example.example_id for example in examples}
    replay_selection = sample_replay(
        replay_eligible,
        size=config.replay_size,
        seed=config.replay_seed,
    )
    replay_candidates = [item for item in replay_selection.items if item.call_id not in excluded]
    replay_examples = _expand_screened_items(
        replay_candidates,
        settings=config,
        drops=drops,
        replay=True,
        existing_ids=existing_ids,
    )
    examples.extend(replay_examples)

    plan = assign_calls(
        {item.call_id for item in eligible} | {item.call_id for item in replay_eligible},
        golden_call_ids=golden,
        regression_call_ids=regression,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
        seed=config.split_seed,
    )
    assert_no_call_in_multiple_splits(plan)
    allowed_calls = plan.assigned_calls()
    final_examples = tuple(example for example in examples if example.call_id in allowed_calls)

    counts = FilterCounts(
        consent_dropped=sum(
            1 for drop in drops if any(reason in CONSENT_DROP_REASONS for reason in drop.reasons)
        ),
        pii_dropped=sum(1 for drop in drops if DropReason.CARD_NUMBER_PII in drop.reasons),
        replay_added=len(replay_examples),
        golden_excluded=golden_excluded,
        regression_excluded=regression_excluded,
        synthetic_tagged=sum(1 for example in final_examples if example.synthetic),
    )
    return DatasetBuild(
        examples=final_examples,
        split_plan=plan,
        filter_counts=counts,
        dropped=tuple(drops),
    )


def build_dataset_from_fixtures(
    *,
    settings: DatasetSettings | None = None,
    golden_call_ids: Iterable[str] = (),
    regression_call_ids: Iterable[str] = (),
    replay_pool: Iterable[ReviewedItem] = (),
    path: str | Path | None = None,
    root: Path | None = None,
) -> DatasetBuild:
    """Convenience wrapper: load reviewed fixture items and build a dataset."""
    config = settings or DatasetSettings()
    items = load_reviewed_items_from_fixtures(path, settings=config, root=root)
    return build_dataset(
        items,
        settings=config,
        golden_call_ids=golden_call_ids,
        regression_call_ids=regression_call_ids,
        replay_pool=replay_pool,
    )
