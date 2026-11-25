"""Deterministic, injectable dataset -> train -> evaluate chain (P2-10).

The chain owns no ML code: each step is a callable supplied by the caller, so
tests and the fixture CLI run with fakes and without GPU/network/MLflow. The
chain's only job is ordering, lineage, and honest failure reporting.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from voiceloop_common import (
    Clock,
    ConfigError,
    ValidationError,
    VoiceLoopError,
    new_id,
    utcnow,
    validate,
)

from .settings import PipelineSettings
from .triggers import TriggerDecision

DATASET_MANIFEST_SCHEMA = "dataset_manifest.schema.json"
TRAINER_CONTRACT_SCHEMA = "trainer_contract.schema.json"
EVAL_REPORT_SCHEMA = "eval_report.schema.json"
VERIFIED_FAILURE_TYPES = frozenset({"stt_wrong", "parser_wrong", "both_fine", "clean"})


class DatasetBuilder(Protocol):
    """Callable returning a dataset manifest, optionally wrapped with verified IDs."""

    def __call__(self) -> Mapping[str, Any]:
        """Build and return a manifest (or a dict with a ``manifest`` key)."""


class TrainerStep(Protocol):
    """Callable implementing ``train(dataset_manifest) -> contract output``."""

    def __call__(self, manifest: Mapping[str, Any]) -> Mapping[str, Any]:
        """Train from a manifest and return a contract-shaped result."""


class EvaluatorStep(Protocol):
    """Callable implementing ``evaluate(artifact_uri, eval_sets) -> report.json``."""

    def __call__(self, artifact_uri: str, eval_sets: Sequence[str]) -> Mapping[str, Any]:
        """Evaluate an artifact and return a result containing ``report_path``."""


@dataclass(frozen=True, slots=True)
class PipelineSteps:
    """The three injectable steps executed in order by :func:`run_chain`."""

    dataset: DatasetBuilder
    trainer: TrainerStep
    evaluator: EvaluatorStep


@dataclass(frozen=True, slots=True)
class StepRecord:
    """One completed step's compact, JSON-serialisable result."""

    name: str
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LineageRecord:
    """Dataset hash -> trainer run -> evaluation report provenance (context.md section 2.9)."""

    dataset_hash: str
    dataset_id: str
    trainer_run_id: str
    artifact_uri: str
    eval_report_path: str
    git_sha: str | None = None
    image_digest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable lineage mapping."""
        result: dict[str, Any] = {
            "dataset_hash": self.dataset_hash,
            "dataset_id": self.dataset_id,
            "trainer_run_id": self.trainer_run_id,
            "artifact_uri": self.artifact_uri,
            "eval_report_path": self.eval_report_path,
        }
        if self.git_sha is not None:
            result["git_sha"] = self.git_sha
        if self.image_digest is not None:
            result["image_digest"] = self.image_digest
        return result


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Full audit record of one chain execution, including the trigger reason."""

    run_id: str
    started_at: str
    finished_at: str
    status: str
    trigger: dict[str, Any]
    steps: tuple[StepRecord, ...] = ()
    lineage: LineageRecord | None = None
    verified_ids: tuple[str, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record suitable for logs and state files."""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "trigger": dict(self.trigger),
            "steps": [{"name": step.name, "summary": dict(step.summary)} for step in self.steps],
            "lineage": self.lineage.to_dict() if self.lineage is not None else None,
            "verified_ids": list(self.verified_ids),
            "error": self.error,
        }


class PipelineError(VoiceLoopError):
    """Base error carrying the partial run record for reporting."""

    def __init__(self, message: str, *, record: RunRecord | None = None) -> None:
        """Store the message and, when available, the run record."""
        super().__init__(message)
        self.record = record


class PipelineStepError(PipelineError):
    """A step failed; the chain stopped and the record explains where."""

    def __init__(self, message: str, *, step: str, record: RunRecord | None = None) -> None:
        """Store the failing step name alongside the partial record."""
        super().__init__(message, record=record)
        self.step = step


class TriggerNotSatisfied(PipelineError):
    """No allowed trigger condition was met, so the chain was not started."""


def _iso(value: Any) -> str:
    """Return an RFC 3339 UTC string from an injected clock value."""
    if not isinstance(value, datetime):
        raise ConfigError("pipeline clock must return a datetime")
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return aware.astimezone(UTC).isoformat()


def _require_str(payload: Mapping[str, Any], key: str, step: str) -> str:
    """Read a required non-empty string from a step result."""
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{step} result must contain a non-empty string '{key}'")
    return value


def _optional_str(payload: Mapping[str, Any], key: str) -> str | None:
    """Read an optional non-empty string from a lineage mapping."""
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValidationError(f"lineage '{key}' must be a non-empty string when present")
    return value


def _trigger_dict(trigger: TriggerDecision | Mapping[str, Any]) -> dict[str, Any]:
    """Normalise a trigger decision or pre-serialised record into a dict."""
    if isinstance(trigger, TriggerDecision):
        return trigger.to_dict()
    return dict(trigger)


def _is_triggered(trigger: TriggerDecision | Mapping[str, Any]) -> bool:
    """Read the ``triggered`` flag from a decision or record."""
    if isinstance(trigger, TriggerDecision):
        return trigger.triggered
    return bool(trigger.get("triggered"))


def _unwrap_dataset(result: Mapping[str, Any]) -> tuple[Mapping[str, Any], tuple[str, ...]]:
    """Return the manifest and verified IDs from a dataset-step result."""
    if "manifest" not in result:
        return result, ()
    manifest = result["manifest"]
    if not isinstance(manifest, Mapping):
        raise ValidationError("dataset result 'manifest' must be a mapping")
    raw_ids = result.get("verified_ids", ())
    if not isinstance(raw_ids, Iterable) or isinstance(raw_ids, (str, bytes)):
        raise ValidationError("dataset result 'verified_ids' must be an iterable of strings")
    ids = tuple(raw_ids)
    if any(not isinstance(item, str) for item in ids):
        raise ValidationError("dataset result 'verified_ids' must contain only strings")
    return manifest, ids


def run_chain(
    *,
    steps: PipelineSteps,
    trigger: TriggerDecision | Mapping[str, Any],
    eval_sets: Sequence[str] = ("general_capability",),
    clock: Clock = utcnow,
) -> RunRecord:
    """Run dataset -> train -> evaluate in order and return a lineage-rich record.

    Args:
        steps: Injectable callables for each stage.
        trigger: Decision or serialised decision recorded on the run.
        eval_sets: Evaluation set names forwarded to the evaluator.
        clock: Injectable UTC clock for deterministic timestamps.

    Returns:
        A successful :class:`RunRecord` with all three step summaries.

    Raises:
        TriggerNotSatisfied: If the trigger policy did not allow a run.
        PipelineStepError: If any step fails; ``exception.record`` holds the
            partial record and ``exception.step`` names the failing stage.

    """
    record = RunRecord(
        run_id=new_id(),
        started_at=_iso(clock()),
        finished_at="",
        status="skipped",
        trigger=_trigger_dict(trigger),
    )
    if not _is_triggered(trigger):
        skipped = replace(record, finished_at=_iso(clock()), error="trigger policy not satisfied")
        raise TriggerNotSatisfied("trigger policy not satisfied", record=skipped)

    completed: list[StepRecord] = []
    current_step = "dataset"
    try:
        dataset_result = steps.dataset()
        manifest, verified_ids = _unwrap_dataset(dataset_result)
        content_hash = _require_str(manifest, "content_hash", "dataset")
        dataset_id = _require_str(manifest, "dataset_id", "dataset")
        completed.append(
            StepRecord(
                "dataset",
                {
                    "dataset_id": dataset_id,
                    "content_hash": content_hash,
                    "examples": manifest.get("examples"),
                },
            )
        )

        current_step = "train"
        train_result = steps.trainer(manifest)
        artifact_uri = _require_str(train_result, "artifact_uri", "train")
        trainer_run_id = _require_str(train_result, "run_id", "train")
        lineage = train_result.get("lineage")
        if not isinstance(lineage, Mapping):
            raise ValidationError("train result must contain a lineage mapping")
        dataset_hash = _require_str(lineage, "dataset_hash", "train lineage")
        if dataset_hash != content_hash:
            raise ValidationError(
                "trainer lineage dataset_hash "
                f"{dataset_hash!r} does not match manifest {content_hash!r}"
            )
        completed.append(
            StepRecord(
                "train",
                {
                    "run_id": trainer_run_id,
                    "artifact_uri": artifact_uri,
                    "dataset_hash": dataset_hash,
                },
            )
        )

        current_step = "evaluate"
        eval_result = steps.evaluator(artifact_uri, tuple(eval_sets))
        report_path = _require_str(eval_result, "report_path", "evaluate")
        reported_artifact = eval_result.get("artifact_uri")
        if reported_artifact is not None and reported_artifact != artifact_uri:
            raise ValidationError(
                f"evaluated artifact {reported_artifact!r} "
                f"is not the trained artifact {artifact_uri!r}"
            )
        completed.append(
            StepRecord(
                "evaluate",
                {
                    "report_path": report_path,
                    "artifact_uri": artifact_uri,
                    "eval_sets": list(eval_sets),
                },
            )
        )
    except Exception as exc:
        failed = replace(
            record,
            finished_at=_iso(clock()),
            status="failed",
            steps=tuple(completed),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise PipelineStepError(
            f"{current_step} step failed: {type(exc).__name__}: {exc}",
            step=current_step,
            record=failed,
        ) from exc

    lineage_record = LineageRecord(
        dataset_hash=content_hash,
        dataset_id=dataset_id,
        trainer_run_id=trainer_run_id,
        artifact_uri=artifact_uri,
        eval_report_path=report_path,
        git_sha=_optional_str(lineage, "git_sha"),
        image_digest=_optional_str(lineage, "image_digest"),
    )
    return replace(
        record,
        run_id=trainer_run_id,
        finished_at=_iso(clock()),
        status="success",
        steps=tuple(completed),
        lineage=lineage_record,
        verified_ids=verified_ids,
    )


def read_fixture_labels(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Read the synthetic fixture labels manifest as JSON Lines.

    Args:
        path: Repo-relative or absolute path to ``turns.jsonl``.

    Returns:
        One decoded mapping per non-empty line.

    Raises:
        ConfigError: If the file cannot be read or a line is not an object.

    """
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigError(f"cannot read fixture labels at {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid JSON in {path} line {line_number}: {exc}") from exc
        if not isinstance(payload, dict) or "call_id" not in payload or "turn_index" not in payload:
            raise ConfigError(f"{path} line {line_number} must be a labelled turn object")
        rows.append(payload)
    return tuple(rows)


def _label_id(row: Mapping[str, Any]) -> str:
    """Build a stable ID for a labelled turn."""
    return f"{row['call_id']}:{row['turn_index']}"


def verified_example_ids(labels: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    """Select consented, usable fixture turns as verified training examples.

    ``unusable_audio`` and rows without training consent are excluded; the
    synthetic fixture labels stand in for human verification here only.
    """
    ids: set[str] = set()
    for row in labels:
        consent = row.get("consent")
        if not isinstance(consent, Mapping) or consent.get("training") is not True:
            continue
        if row.get("failure_type") not in VERIFIED_FAILURE_TYPES:
            continue
        ids.add(_label_id(row))
    return tuple(sorted(ids))


def build_fixture_steps(
    settings: PipelineSettings | None = None,
    *,
    clock: Clock = utcnow,
) -> PipelineSteps:
    """Build fake steps that run the full chain on synthetic fixtures.

    The fake builder validates contract-shaped outputs, filters by consent,
    hashes canonical labels, and splits by call ID so no call crosses splits.
    It is a plumbing fixture, not a replacement for the P2-05 dataset builder.
    """
    config = settings or PipelineSettings()

    def dataset() -> Mapping[str, Any]:
        """Build a schema-valid manifest from consented synthetic labels."""
        labels = read_fixture_labels(config.fixtures_path)
        ids = verified_example_ids(labels)
        id_set = set(ids)
        by_id = {_label_id(row): row for row in labels if _label_id(row) in id_set}
        canonical = json.dumps([by_id[item] for item in ids], sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        calls = sorted({str(by_id[item]["call_id"]) for item in ids})
        train_end = max(1, len(calls) - 2) if calls else 0
        split = {
            "train": calls[:train_end],
            "val": calls[train_end : train_end + 1],
            "test": calls[train_end + 1 :],
        }
        consent_dropped = sum(
            1
            for row in labels
            if not isinstance(row.get("consent"), Mapping)
            or row["consent"].get("training") is not True
        )
        manifest: dict[str, Any] = {
            "schema_version": "1.0.0",
            "dataset_id": f"{config.dataset_id_prefix}-{digest[:12]}",
            "content_hash": digest,
            "source_call_ids": calls,
            "splits": split,
            "filter_counts": {
                "consent_dropped": consent_dropped,
                "pii_dropped": 0,
                "replay_added": 0,
                "golden_excluded": 0,
                "regression_excluded": 0,
                "synthetic_tagged": len(ids),
            },
            "created_at": _iso(clock()),
            "examples": len(ids),
            "synthetic": True,
        }
        validate(manifest, DATASET_MANIFEST_SCHEMA)
        return {"manifest": manifest, "verified_ids": list(ids)}

    def trainer(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return a deterministic contract-shaped fake training result."""
        digest = _require_str(manifest, "content_hash", "trainer")
        result: dict[str, Any] = {
            "schema_version": "1.0.0",
            "contract_part": "output",
            "artifact_uri": f"fake://trainer-artifacts/{digest[:12]}/deployable",
            "metrics": {"train_loss": 0.0},
            "run_id": f"fake-run-{digest[:12]}",
            "lineage": {
                "dataset_hash": digest,
                "git_sha": config.trainer_git_sha,
                "image_digest": config.trainer_image_digest,
            },
        }
        validate(result, TRAINER_CONTRACT_SCHEMA)
        return result

    def evaluator(artifact_uri: str, eval_sets: Sequence[str]) -> Mapping[str, Any]:
        """Return a schema-valid fake evaluation report with a report path."""
        if any(name != "general_capability" for name in eval_sets):
            raise ConfigError(
                "the fixture evaluator only supports 'general_capability'; "
                "synthetic fixtures are not the sealed golden set"
            )
        metrics = {
            "wer": 0.0,
            "entity_error": 0.0,
            "slot_f1": 1.0,
            "schema_valid_rate": 1.0,
            "e2e_booking_accuracy": 1.0,
        }
        interval = {"lower": 0.0, "upper": 0.0, "level": 0.95}
        report: dict[str, Any] = {
            "schema_version": "1.0.0",
            "artifact_uri": artifact_uri,
            "generated_at": _iso(clock()),
            "eval_sets": list(eval_sets),
            "metrics": metrics,
            "confidence_intervals": {name: dict(interval) for name in metrics},
            "slice_metrics": {"general_capability": {"wer": 0.0}},
        }
        validate(report, EVAL_REPORT_SCHEMA)
        return {
            "report_path": config.eval_report_path,
            "report": report,
            "artifact_uri": artifact_uri,
        }

    return PipelineSteps(dataset=dataset, trainer=trainer, evaluator=evaluator)
