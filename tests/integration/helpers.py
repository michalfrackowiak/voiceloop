"""Typed helpers and injectable fakes for the P2-13 loop rehearsal.

All helpers use public APIs and in-memory stand-ins. No helper starts Docker,
opens a network connection, decodes real audio, imports a training framework or
writes to a committed fixture. Derived artefacts live under pytest ``tmp_path``
and are explicitly labelled SYNTHETIC / FIXTURE STAND-IN.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sdk import EnqueueStatus, SdkClient
from sdk.transport import InMemoryTransport
from services.datasets import (
    DatasetBuild,
    DatasetSettings,
    ExportResult,
    ReviewedItem,
    build_dataset,
    build_manifest,
    export_all,
)
from services.datasets.golden import VerificationReport
from services.evalgate import EvalgateSettings
from services.evalgate.harness import evaluate
from services.review_adapter import (
    InMemoryReviewTool,
    QueueReportSource,
    ReviewCoordinator,
    ScriptedReview,
    StoredReviewResult,
)
from services.trainers.core import (
    FakeMlflowLogger,
    TrainerRegistry,
    TrainerResult,
    TrainerRunner,
    TrainerSettings,
)
from services.trainers.parser import (
    ParserLoraTrainer,
    ParserSettings,
    SftJsonlExampleLoader,
)
from services.triage.queue import (
    ExplanationScoreSource,
    InMemoryTurnSource,
    QueueReport,
    ScoreSource,
    TriageQueueSettings,
    build_queue,
)
from services.triage.scoring import InMemoryExplanationStore, ScoringEngine

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_TURNS_PATH = REPO_ROOT / "tests" / "fixtures" / "turns.jsonl"
GOLDEN_ARTIFACTS_DIR = REPO_ROOT / "services" / "datasets" / "golden" / "artifacts"

EXPECTED_FIXTURE_SHA256 = "aca23a303954d035245a97f78319df3694d63611b717a2ff2131bc2e0a4453fd"
EXPECTED_GOLDEN_CONTENT_HASH = "662cdaf43ee2d5fde03faf5e69fde196114d4f312406d94c00bf53714d03da8e"

SYNTHETIC_REVIEWER_ID = "fixture-reviewer-synthetic"
FIXTURE_STANDIN_PREFIX = "fixture-standin-"
TRAIN_GIT_SHA = "1234567890abcdef"
TRAIN_IMAGE_DIGEST = "ghcr.io/voiceloop/trainer@sha256:" + "1" * 64
CARD_NUMBER = "4111 1111 1111 1111"
DERIVED_SLICE_MARKER = " (SYNTHETIC-\u00d8)"

FIXED_REVIEW_TIME = datetime(2026, 11, 25, tzinfo=UTC)
EVAL_SETS: tuple[str, ...] = ("general_capability", "regression")
EVAL_BOOTSTRAP_ITERATIONS = 200
EVAL_BOOTSTRAP_SEED = 7


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    """A built dataset, its schema-valid manifest, exports and trainer input.

    ``manifest`` remains schema-valid for lineage and guard APIs. The trainer
    needs an ``sft_path`` override that dataset_manifest 1.0.0 forbids, so that
    extra field is isolated in ``trainer_manifest``.
    """

    build: DatasetBuild
    manifest: dict[str, Any]
    exports: ExportResult
    trainer_manifest: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TrainingBundle:
    """A fake parser training result plus its in-memory lineage log."""

    result: TrainerResult
    logger: FakeMlflowLogger
    converted_uri: str


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def golden_call_ids(report: VerificationReport) -> tuple[str, ...]:
    """Return sorted sealed call IDs from a verified golden report."""
    manifest = report.manifest
    if manifest is None:
        raise ValueError("golden verification report has no manifest")
    return tuple(sorted(str(call_id) for call_id in manifest.call_ids))


def read_fixture_rows(path: Path = FIXTURE_TURNS_PATH) -> list[dict[str, Any]]:
    """Read the committed synthetic label manifest without modifying it."""
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"fixture row must be an object: {path}")
            rows.append(row)
    return rows


def fixture_events(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Load the turn events referenced by synthetic fixture label rows."""
    events: list[dict[str, Any]] = []
    for row in rows:
        event_path = REPO_ROOT / str(row["event_path"])
        decoded = json.loads(event_path.read_text(encoding="utf-8"))
        if not isinstance(decoded, dict):
            raise TypeError(f"fixture event must be an object: {event_path}")
        events.append(decoded)
    return events


def deliver_via_fake_sdk(
    events: Sequence[Mapping[str, Any]],
) -> tuple[InMemoryTransport, tuple[dict[str, Any], ...]]:
    """Deliver events through the public SDK using an injected in-memory transport."""
    transport = InMemoryTransport()
    with SdkClient(transport=transport) as client:
        statuses = [client.enqueue(event) for event in events]
        if not client.wait_until_idle(timeout_seconds=30.0):
            raise RuntimeError("fake SDK worker did not become idle")
    if any(status != EnqueueStatus.ACCEPTED for status in statuses):
        raise RuntimeError("fake SDK rejected a synthetic fixture event")
    return transport, transport.events


def score_delivered_events(
    events: Sequence[Mapping[str, Any]],
) -> InMemoryExplanationStore:
    """Score delivered turn events with P2-01 and persist their explanations."""
    store = InMemoryExplanationStore()
    engine = ScoringEngine(store=store)
    for event in events:
        engine.score(event)
    return store


def build_review_queue(
    events: Sequence[Mapping[str, Any]],
    store: InMemoryExplanationStore,
    *,
    settings: TriageQueueSettings | None = None,
    day: str = "2026-11-25",
) -> QueueReport:
    """Build a P2-02 queue from delivered events and P2-01 explanations."""
    config = settings or TriageQueueSettings()
    return build_queue(
        InMemoryTurnSource(events),
        ExplanationScoreSource(store),
        config,
        day=day,
    )


def build_queue_with_score_source(
    events: Sequence[Mapping[str, Any]],
    scores: ScoreSource,
    *,
    settings: TriageQueueSettings,
    day: str = "2026-11-25",
) -> QueueReport:
    """Build a comparison queue with an explicitly injected score source."""
    return build_queue(InMemoryTurnSource(events), scores, settings, day=day)


def scripted_reviews(
    events: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, ScriptedReview]:
    """Derive fake stage-1/2 answers from synthetic fixture labels only."""
    by_key = {(row["call_id"], row["turn_index"]): row for row in rows}
    reviews: dict[str, ScriptedReview] = {}
    for event in events:
        row = by_key[(event["call_id"], event["turn_index"])]
        failure_type = str(row.get("failure_type", "both_fine"))
        routing = "both_fine" if failure_type == "clean" else failure_type
        reviews[str(event["event_id"])] = ScriptedReview(
            corrected_transcript=str(row["expected_transcript"]),
            slots=dict(row["expected_slots"]),
            routing=routing,
            reviewer_id=SYNTHETIC_REVIEWER_ID,
        )
    return reviews


def run_scripted_review(
    queue_report: QueueReport,
    reviews: Mapping[str, ScriptedReview],
) -> tuple[StoredReviewResult, ...]:
    """Run P2-03's two stages through the in-memory fake review tool."""
    tool = InMemoryReviewTool(reviews, clock=lambda: FIXED_REVIEW_TIME)
    coordinator = ReviewCoordinator(tool)
    return coordinator.run_all(QueueReportSource(queue_report))


def reviewed_items_from_results(
    results: Sequence[StoredReviewResult],
    events: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[ReviewedItem, ...]:
    """Join P2-03 lineage-rich results to source events and synthetic labels."""
    events_by_id = {str(event["event_id"]): event for event in events}
    labels_by_key = {(row["call_id"], row["turn_index"]): row for row in rows}
    items: list[ReviewedItem] = []
    for result in results:
        event = events_by_id[result.event_id]
        label = labels_by_key[(event["call_id"], event["turn_index"])]
        view = result.to_dataset_view()
        items.append(
            ReviewedItem(
                event=event,
                corrected_transcript=str(view["corrected_transcript"]),
                slots=dict(view["slots"]),
                routing=result.routing,
                reviewer_id=result.reviewer_id,
                failure_type=str(label.get("failure_type")),
                synthetic=True,
            )
        )
    return tuple(items)


def clone_as_fixture_standin(
    source: ReviewedItem,
    *,
    call_id: str,
    event_id: str | None = None,
    consent_training: bool | None = None,
    stt_text: str | None = None,
    corrected_transcript: str | None = None,
    slots: Mapping[str, Any] | None = None,
) -> ReviewedItem:
    """Clone a reviewed fixture item into a labelled non-golden stand-in call."""
    event = deepcopy(dict(source.event))
    source_key = f"{event.get('call_id')}:{event.get('turn_index')}"
    event["event_id"] = event_id or str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"P2-13/fixture-standin/{source_key}/{call_id}")
    )
    event["call_id"] = call_id
    consent = event.get("consent")
    if isinstance(consent, dict):
        if consent_training is not None:
            consent["training"] = consent_training
    if stt_text is not None:
        stt = event.get("stt")
        if isinstance(stt, dict):
            stt["text"] = stt_text
    return ReviewedItem(
        event=event,
        corrected_transcript=(
            source.corrected_transcript if corrected_transcript is None else corrected_transcript
        ),
        slots=dict(source.slots if slots is None else slots),
        routing=source.routing,
        reviewer_id="fixture-standin-reviewer-synthetic",
        failure_type=source.failure_type,
        synthetic=True,
    )


def make_training_standins(
    reviewed: Sequence[ReviewedItem],
    *,
    golden_call_ids: Sequence[str],
    count: int = 4,
) -> tuple[ReviewedItem, ...]:
    """Select reviewed stand-ins covering gold and noisy parser example kinds."""
    required_routings = ("parser_wrong", "stt_wrong", "unusable_audio", "parser_wrong")
    used: set[int] = set()
    standins: list[ReviewedItem] = []
    for index, routing in enumerate(required_routings[:count], start=1):
        candidate = next(
            (item for item in reviewed if item.routing == routing and id(item) not in used),
            None,
        )
        if candidate is None:
            raise ValueError(f"no reviewed fixture item with routing {routing!r}")
        used.add(id(candidate))
        call_id = f"{FIXTURE_STANDIN_PREFIX}{index:02d}"
        if call_id in golden_call_ids:
            raise ValueError(f"stand-in call collides with sealed golden call: {call_id}")
        standins.append(clone_as_fixture_standin(candidate, call_id=call_id))
    return tuple(standins)


def build_dataset_bundle(
    items: Sequence[ReviewedItem],
    *,
    golden_call_ids: Sequence[str],
    output_dir: Path,
    excluded_call_ids: Iterable[str] = (),
    prefix: str = "loop",
    settings: DatasetSettings | None = None,
) -> DatasetBundle:
    """Build, hash, manifest and export one fixture-stand-in dataset.

    ``excluded_call_ids`` is passed as ``regression_call_ids`` so tombstones are
    excluded before the builder hashes examples, as required by P2-12.
    """
    config = settings or DatasetSettings()
    build = build_dataset(
        items,
        settings=config,
        golden_call_ids=golden_call_ids,
        regression_call_ids=tuple(excluded_call_ids),
        replay_pool=(),
    )
    manifest = build_manifest(
        build,
        created_at="2026-11-25T00:00:00Z",
        dataset_id_prefix=config.dataset_id_prefix,
    )
    exports = export_all(build.examples, output_dir, prefix=prefix)
    trainer_manifest = dict(manifest)
    trainer_manifest["sft_path"] = str(exports.sft_path)
    return DatasetBundle(
        build=build,
        manifest=manifest,
        exports=exports,
        trainer_manifest=trainer_manifest,
    )


def train_fake_parser(
    bundle: DatasetBundle,
    *,
    golden_call_ids: Sequence[str],
) -> TrainingBundle:
    """Train the deterministic fake LoRA stand-in through the public runner."""
    registry = TrainerRegistry()
    logger = FakeMlflowLogger()
    job_settings = TrainerSettings(
        git_sha=TRAIN_GIT_SHA,
        image_digest=TRAIN_IMAGE_DIGEST,
        seed=13,
    )
    parser_settings = ParserSettings(exclude_call_ids=tuple(golden_call_ids))
    registry.register(
        ParserLoraTrainer(
            settings=parser_settings,
            example_loader=SftJsonlExampleLoader(exclude_call_ids=golden_call_ids),
        )
    )
    runner = TrainerRunner(registry, job_settings, logger=logger)
    result = runner.run("parser-lora", bundle.trainer_manifest)
    artifact_uri = str(result["artifact_uri"])
    return TrainingBundle(
        result=result,
        logger=logger,
        converted_uri=f"converted://{artifact_uri.split('://', 1)[-1]}",
    )


def write_derived_eval_fixture(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write a labelled derived fixture with one non-ASCII slice marker.

    The committed fixture is never edited; the marker exists only so P1-07 has
    non-zero ``other_languages`` evidence for the P2-09 gate rehearsal.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    derived: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        copy = dict(row)
        if index == 0:
            copy["expected_transcript"] = str(copy["expected_transcript"]) + DERIVED_SLICE_MARKER
        derived.append(copy)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in derived),
        encoding="utf-8",
    )
    return path


def evaluate_converted_artifact(
    derived_fixture: Path,
    *,
    converted_uri: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Produce champion and challenger P1-07 reports for the gate rehearsal."""
    common: dict[str, Any] = {
        "fixtures_path": str(derived_fixture),
        "eval_sets": list(EVAL_SETS),
        "bootstrap_iterations": EVAL_BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": EVAL_BOOTSTRAP_SEED,
    }
    champion_config = EvalgateSettings(
        **common,
        champion_artifact="identity",
        challenger_artifact="identity",
    )
    challenger_config = EvalgateSettings(
        **common,
        champion_artifact="identity",
        challenger_artifact="oracle",
    )
    champion = evaluate(
        "fake://evalgate/champion-synthetic-fixture-standin",
        None,
        champion_config,
    )
    challenger = evaluate(converted_uri, None, challenger_config)
    return champion, challenger


def exported_text(exports: ExportResult) -> str:
    """Read all three exports as text for PII leak assertions."""
    return "".join(
        path.read_text(encoding="utf-8")
        for path in (exports.nemo_path, exports.hf_path, exports.sft_path)
    )
