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
from dataclasses import dataclass, replace
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
from services.evalgate import ArtifactKind, EvalgateSettings
from services.evalgate.artifact import (
    ArtifactBehaviour,
    ArtifactEvaluation,
    ArtifactSettings,
    ReadOnlyArtifactStore,
    create_fake_artifact,
    evaluate_artifact,
    validate_stt_report,
    write_stt_report,
)
from services.evalgate.gate import (
    CHAMPION_ALIAS,
    CollectingShadowEvidenceSource,
    PrefixArtifactConverter,
    PromotionRefusalLog,
    RefusalAuditRecord,
    RefusalCode,
    StaticShadowEvidenceSource,
)
from services.evalgate.harness import evaluate
from services.release.shadow import (
    AuditEvent,
    AuditSink,
    BoundedMirrorSink,
    CollectingAuditSink,
    ComparisonData,
    FakeClock,
    FakeOutcomeSource,
    MirrorReceipt,
    ModelSource,
    OutcomeRecord,
    ScriptedModelRunner,
    ShadowMirror,
    ShadowRunner,
    ShadowSettings,
    ShadowTurn,
    build_shadow_report,
    score_comparison,
    write_shadow_report,
)
from services.release.sync import (
    AliasSyncAgent,
    ArtifactRef,
    CollectingLogger,
    FakeArtifactStore,
    FakeServingStack,
    GateAliasChampionRegistry,
    ReleaseSyncSettings,
    ServingStateProbe,
    hash_payload,
)
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
from services.trainers.whisper import (
    HfJsonExampleLoader,
    WhisperLoraTrainer,
    WhisperSettings,
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

STT_RELEASE_LABEL = "SYNTHETIC/FIXTURE STAND-IN - NOT A PRODUCTION STT RELEASE"
STT_TASK_ID = "P3-12"
STT_CHAMPION_VERSION = "v1-stt-fixture-standin"
STT_CHALLENGER_VERSION = "v2-stt-fixture-standin"
STT_BOOKING_FIELDS: dict[str, Any] = {"room_type": "deluxe", "guests": 2}
STT_CHAMPION_BOOKING_FIELDS: dict[str, Any] = {"room_type": "standard", "guests": 2}
PII_FRAGMENTS: tuple[str, ...] = (
    "Ada Fixture",
    "Sam Sample",
    "ada.fixture@example.com",
    "sam.sample@example.com",
    "+1 555 010",
    CARD_NUMBER,
    CARD_NUMBER.replace(" ", ""),
    "Make that a deluxe room.",
    "The name is Mr. Smith.",
)


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


@dataclass(frozen=True, slots=True)
class SttTrainingBundle:
    """A fake Whisper LoRA result, its runtime seam and converted URI."""

    result: TrainerResult
    logger: FakeMlflowLogger
    trainer: WhisperLoraTrainer
    converted_uri: str


@dataclass(frozen=True, slots=True)
class ShadowEvidenceBundle:
    """A synthetic-labelled P3-07 report plus its call-path evidence."""

    report: dict[str, Any]
    path: Path
    candidate_version: str
    champion_version: str
    receipts: tuple[MirrorReceipt, ...]
    audit: CollectingAuditSink
    comparison: ComparisonData


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


def stt_training_items(
    fixture_rows: Sequence[Mapping[str, Any]],
    *,
    prefix: str = "stt-fixture-standin-",
) -> tuple[ReviewedItem, ...]:
    """Build labelled non-golden STT stand-ins from synthetic fixture labels."""
    events = fixture_events(fixture_rows)
    by_key = {(event["call_id"], event["turn_index"]): event for event in events}
    items: list[ReviewedItem] = []
    for index, row in enumerate(fixture_rows, start=1):
        event = by_key[(row["call_id"], row["turn_index"])]
        failure_type = str(row.get("failure_type", "clean"))
        item = ReviewedItem(
            event=deepcopy(event),
            corrected_transcript=str(row["expected_transcript"]),
            slots=dict(row["expected_slots"]),
            routing="both_fine" if failure_type == "clean" else failure_type,
            reviewer_id=SYNTHETIC_REVIEWER_ID,
            failure_type=failure_type,
            synthetic=True,
        )
        items.append(clone_as_fixture_standin(item, call_id=f"{prefix}{index:02d}"))
    return tuple(items)


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


def label_stt_report(report: dict[str, Any]) -> dict[str, Any]:
    """Attach the mandatory STT stand-in label to one generated report."""
    decoding = report.get("decoding_settings")
    if isinstance(decoding, dict):
        decoding["release_label"] = STT_RELEASE_LABEL
    decision = report.get("gate_decision")
    if isinstance(decision, dict) and isinstance(decision.get("reason"), str):
        decision["reason"] = f"{STT_RELEASE_LABEL}; {decision['reason']}"
    return report


def assert_no_pii_text(text: str) -> None:
    """Fail if a derived output carries fixture PII, raw text or audio bytes."""
    folded = text.casefold()
    assert not any(fragment.casefold() in folded for fragment in PII_FRAGMENTS)
    assert "RIFF" not in text
    assert "WAVE" not in text


def assert_labelled_outputs(paths: Sequence[Path]) -> None:
    """Assert that every derived output is labelled and free of fixture PII."""
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert STT_RELEASE_LABEL in text, f"missing STT stand-in label: {path.name}"
        assert_no_pii_text(text)


def train_fake_whisper(
    bundle: DatasetBundle,
    *,
    golden_call_ids: Sequence[str],
) -> SttTrainingBundle:
    """Run the public P3-01 fake Whisper LoRA plugin through the runner."""
    registry = TrainerRegistry()
    logger = FakeMlflowLogger()
    job_settings = TrainerSettings(
        git_sha=TRAIN_GIT_SHA,
        image_digest=TRAIN_IMAGE_DIGEST,
        model="fake-whisper-base-synthetic",
        seed=13,
    )
    whisper_settings = WhisperSettings(exclude_call_ids=tuple(golden_call_ids))
    trainer = WhisperLoraTrainer(
        settings=whisper_settings,
        example_loader=HfJsonExampleLoader(
            path=bundle.exports.hf_path,
            exclude_call_ids=golden_call_ids,
        ),
    )
    registry.register(trainer)
    runner = TrainerRunner(registry, job_settings, logger=logger)
    result = runner.run("whisper-lora", bundle.trainer_manifest)
    return SttTrainingBundle(
        result=result,
        logger=logger,
        trainer=trainer,
        converted_uri=PrefixArtifactConverter().convert(str(result["artifact_uri"])).artifact_uri,
    )


def stt_artifact_settings(
    tmp_path: Path,
    *,
    checkpoint_uri: str,
    behaviour: ArtifactBehaviour = "healthy",
) -> ArtifactSettings:
    """Build deterministic P3-04 settings over the read-only committed fixture."""
    return ArtifactSettings(
        fixtures_path="tests/fixtures/turns.jsonl",
        golden_artifact_dir="services/datasets/golden/artifacts",
        output_path=str(tmp_path / "reports" / "stt" / "stt_report.json"),
        artifact_store_dir=str(tmp_path / "reports" / "stt" / "artifacts"),
        checkpoint_uri=checkpoint_uri,
        artifact_behaviour=behaviour,
        runtime_label="fake-ctranslate2-stt-standin",
        hotwords=[],
        initial_prompt="",
        condition_on_previous_text=False,
        served_languages_csv="en,es",
        other_language_pattern=r"[A-Za-z]{3,}",
        bootstrap_iterations=EVAL_BOOTSTRAP_ITERATIONS,
        bootstrap_seed=EVAL_BOOTSTRAP_SEED,
    )


def evaluate_stt_artifact(
    settings: ArtifactSettings,
    behaviour: ArtifactBehaviour,
) -> tuple[Any, ArtifactEvaluation]:
    """Store and evaluate one exact P3-04 fake artifact and write its report."""
    artifact = create_fake_artifact(behaviour, settings.runtime_label)
    store = ReadOnlyArtifactStore(settings.artifact_store_dir)
    stored_path = store.put(artifact)
    assert store.load(artifact.sha256) == artifact
    assert stored_path.read_bytes() == artifact.descriptor_bytes()

    evaluation = evaluate_artifact(settings, artifact)
    label_stt_report(evaluation.report)
    label_stt_report(evaluation.champion_report)
    validate_stt_report(evaluation.report)
    validate_stt_report(evaluation.champion_report)
    report_path = write_stt_report(evaluation.report, settings.output_path)
    champion_path = write_stt_report(
        evaluation.champion_report, report_path.with_name("stt_champion_report.json")
    )
    assert report_path.exists() and champion_path.exists()
    return artifact, evaluation


def evaluate_stt_promotion_reports(
    derived_fixture: Path,
    *,
    champion_uri: str,
    challenger_uri: str,
    champion_artifact_kind: ArtifactKind = "identity",
    challenger_artifact_kind: ArtifactKind = "oracle",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build labelled, contract-valid P1-07 reports for an STT promotion gate."""
    common: dict[str, Any] = {
        "fixtures_path": str(derived_fixture),
        "eval_sets": list(EVAL_SETS),
        "bootstrap_iterations": EVAL_BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": EVAL_BOOTSTRAP_SEED,
    }
    champion_config = EvalgateSettings(
        **common,
        champion_artifact=champion_artifact_kind,
        challenger_artifact="identity",
    )
    challenger_config = EvalgateSettings(
        **common,
        champion_artifact=champion_artifact_kind,
        challenger_artifact=challenger_artifact_kind,
    )
    champion = evaluate(champion_uri, None, champion_config)
    challenger = evaluate(challenger_uri, None, challenger_config)
    assert champion["artifact_uri"] == champion_uri
    assert challenger["artifact_uri"] == challenger_uri
    return label_stt_report(champion), label_stt_report(challenger)


class LabelledAuditSink:
    """Audit seam that prefixes every P3-07 event with the STT stand-in label."""

    def __init__(self, delegate: AuditSink) -> None:
        """Wrap a concrete audit sink without changing its protocol."""
        self._delegate = delegate

    @property
    def delegate(self) -> AuditSink:
        """Return the wrapped sink for read-only event assertions."""
        return self._delegate

    def record(self, event: AuditEvent) -> None:
        """Append one labelled, PII-safe audit event."""
        self._delegate.record(replace(event, status=f"{STT_RELEASE_LABEL} | {event.status}"))


class LabelledRefusalLog:
    """Refusal seam that labels every hash-chained reason without losing codes."""

    def __init__(self, delegate: PromotionRefusalLog) -> None:
        """Wrap a concrete append-only refusal sink."""
        self._delegate = delegate

    @property
    def delegate(self) -> PromotionRefusalLog:
        """Return the wrapped sink for read-only chain assertions."""
        return self._delegate

    def record_refusal(
        self,
        *,
        actor_id: str,
        code: RefusalCode,
        reasons: Sequence[str],
        champion_version: str,
        challenger_version: str,
        candidate_artifact_uri: str,
        task_id: str = "P3-08",
    ) -> RefusalAuditRecord:
        """Append one labelled refusal record through the hash-chained delegate."""
        return self.delegate.record_refusal(
            actor_id=actor_id,
            code=code,
            reasons=(STT_RELEASE_LABEL, *(str(reason) for reason in reasons)),
            champion_version=champion_version,
            challenger_version=challenger_version,
            candidate_artifact_uri=candidate_artifact_uri,
            task_id=task_id,
        )


def make_labelled_refusal_log(path: Path) -> LabelledRefusalLog:
    """Build a labelled wrapper over the real hash-chained refusal log."""
    return LabelledRefusalLog(PromotionRefusalLog(path))


class LabelledStructuredLogger:
    """Structured logger seam that labels every P3-06 sync event."""

    def __init__(self, delegate: CollectingLogger) -> None:
        """Wrap the deterministic in-memory collecting logger."""
        self.delegate = delegate

    def info(self, event: str, **fields: object) -> None:
        """Record a labelled info event."""
        self._record("info", event, fields)

    def error(self, event: str, **fields: object) -> None:
        """Record a labelled error event."""
        self._record("error", event, fields)

    def warning(self, event: str, **fields: object) -> None:
        """Record a labelled warning event."""
        self._record("warning", event, fields)

    def _record(self, level: str, event: str, fields: dict[str, object]) -> None:
        """Delegate one event with an added release label field."""
        labelled = {**fields, "release_label": STT_RELEASE_LABEL}
        getattr(self.delegate, level)(event, **labelled)


def make_labelled_sync_logger() -> LabelledStructuredLogger:
    """Build a labelled wrapper over the fake collecting logger."""
    return LabelledStructuredLogger(CollectingLogger())


def build_synthetic_shadow_evidence(
    tmp_path: Path,
    *,
    fixture_rows: Sequence[Mapping[str, Any]],
    candidate_artifact_uri: str,
    champion_artifact_uri: str,
    candidate_version: str = STT_CHALLENGER_VERSION,
    champion_version: str = STT_CHAMPION_VERSION,
) -> ShadowEvidenceBundle:
    """Mirror synthetic turn references and build one P3-07 comparison artifact."""
    clock = FakeClock(start=datetime(2026, 11, 25, tzinfo=UTC))
    settings = ShadowSettings(
        task_id=STT_TASK_ID,
        max_pending=8,
        mirror_budget_ms=2.0,
        max_turns_per_drain=8,
        output_path=str(tmp_path / "reports" / "shadow" / "comparison.json"),
        report_status_label=STT_RELEASE_LABEL,
    )
    rows = tuple(fixture_rows[:6])
    turns = tuple(
        ShadowTurn(
            call_id=str(row["call_id"]),
            turn_index=int(row["turn_index"]),
            audio_uri=str(row["audio_path"]),
            synthetic=True,
            observed_at=clock.now(),
        )
        for row in rows
    )
    candidate = ScriptedModelRunner(
        source=ModelSource.CANDIDATE,
        artifact_uri=candidate_artifact_uri,
        model_id="fixture-standin-stt-candidate",
        clock=clock,
    )
    champion = ScriptedModelRunner(
        source=ModelSource.CHAMPION,
        artifact_uri=champion_artifact_uri,
        model_id="fixture-standin-stt-champion",
        clock=clock,
    )
    for turn in turns:
        candidate.set_slots(turn.call_id, turn.turn_index, STT_BOOKING_FIELDS)
        champion.set_slots(turn.call_id, turn.turn_index, STT_CHAMPION_BOOKING_FIELDS)

    call_ids = {turn.call_id for turn in turns}
    outcomes = {
        call_id: OutcomeRecord.from_payload(
            call_id,
            {
                "status": "confirmed",
                "final_record": STT_BOOKING_FIELDS,
                "synthetic": True,
                "updated_at": "2026-11-25T00:00:00Z",
            },
            settings.ignored_outcome_fields(),
        )
        for call_id in call_ids
    }

    collecting_audit = CollectingAuditSink()
    audit = LabelledAuditSink(collecting_audit)
    sink = BoundedMirrorSink(clock, settings.max_pending)
    mirror = ShadowMirror(settings, sink, clock, audit)
    receipts = tuple(mirror.mirror(turn) for turn in turns)
    if not all(receipt.accepted for receipt in receipts):
        raise RuntimeError("synthetic shadow mirror rejected a bounded turn reference")

    runner = ShadowRunner(settings, sink, candidate, champion, clock, audit)
    if runner.drain_once() != len(turns):
        raise RuntimeError("synthetic shadow runner did not drain the mirrored turns")
    comparison = score_comparison(
        settings=settings,
        results=runner.results,
        outcome_source=FakeOutcomeSource(outcomes),
        mirror_stats=mirror.stats,
        candidate=candidate,
        champion=champion,
        synthetic=True,
    )
    report_uri = str(Path(settings.output_path).resolve())
    report = build_shadow_report(comparison, report_uri)
    path = write_shadow_report(report, report_uri)
    source = CollectingShadowEvidenceSource(
        StaticShadowEvidenceSource({(candidate_version, champion_version): report})
    )
    assert source.requests == []
    return ShadowEvidenceBundle(
        report=report,
        path=path,
        candidate_version=candidate_version,
        champion_version=champion_version,
        receipts=receipts,
        audit=collecting_audit,
        comparison=comparison,
    )


def build_sync_rehearsal(
    tmp_path: Path,
    *,
    aliases: Any,
    artifact_uri: str,
    version: str,
    champion_artifact_uri: str | None = None,
) -> tuple[AliasSyncAgent, FakeArtifactStore, FakeServingStack, LabelledStructuredLogger]:
    """Wire a P3-06 fake sync agent over a read-only gate alias store."""
    store = FakeArtifactStore()
    artifacts: dict[str, ArtifactRef] = {}
    for artifact_version, uri in (
        (version, artifact_uri),
        (STT_CHAMPION_VERSION, champion_artifact_uri or artifact_uri),
    ):
        artifact_bytes = f"{STT_RELEASE_LABEL}:{artifact_version}".encode()
        digest = store.put(uri, artifact_bytes)
        assert digest == hash_payload(artifact_bytes)
        artifacts[artifact_version] = ArtifactRef(uri=uri, sha256=digest)
    registry = GateAliasChampionRegistry(aliases, artifacts)
    serving = FakeServingStack(
        stt_version=STT_CHAMPION_VERSION, parser_version=STT_CHAMPION_VERSION
    )
    logger = make_labelled_sync_logger()
    agent = AliasSyncAgent(
        settings=ReleaseSyncSettings(alias=CHAMPION_ALIAS, task_id=STT_TASK_ID),
        registry=registry,
        artifacts=store,
        serving=serving,
        probe=ServingStateProbe(serving),
        logger=logger,
    )
    del tmp_path
    return agent, store, serving, logger


def assert_sync_swapped(serving: FakeServingStack, *, version: str) -> None:
    """Assert that both STT and parser serving targets changed to one version."""
    for state in serving.states():
        assert state.current_version == version
        assert state.staged_version is None
        assert state.artifact_hash is not None
