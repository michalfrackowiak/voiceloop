"""Read-only golden loading and paired-comparison tests."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from services.evalgate.benchmarks.compare import compare_candidates, load_benchmark_inputs
from services.evalgate.benchmarks.settings import (
    EXPECTED_GOLDEN_CONTENT_HASH,
    BenchmarkSettings,
)
from services.evalgate.metrics import CORE_METRIC_NAMES
from voiceloop_common import ConfigError, ValidationError


def _golden_hashes(directory: Path) -> dict[str, str]:
    """Hash every committed golden artifact before and after a run."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def test_golden_standin_is_read_only_and_verified(
    benchmark_settings: BenchmarkSettings, repo_root: Path
) -> None:
    """The sealed v1 synthetic stand-in loads with its committed hash and 36 turns."""
    artifact_dir = repo_root / benchmark_settings.golden_artifact_dir
    before = _golden_hashes(artifact_dir)
    inputs = load_benchmark_inputs(benchmark_settings)
    after = _golden_hashes(artifact_dir)
    assert before == after

    assert len(inputs.general_rows) == len(inputs.golden_rows) == 36
    assert inputs.provenance["turns"] == 36
    assert inputs.provenance["calls"] == 6
    assert inputs.provenance["golden_content_hash"] == EXPECTED_GOLDEN_CONTENT_HASH
    assert inputs.provenance["golden_seal_ok"] is True
    assert inputs.provenance["golden_synthetic"] is True
    assert inputs.provenance["golden_workflow_stand_in"] is True
    assert inputs.provenance["golden_used_for_training_or_tuning"] is False
    assert inputs.provenance["source_pairs_identical"] is True


def test_two_candidates_compared_with_directional_deltas(
    benchmark_settings: BenchmarkSettings,
) -> None:
    """identity - degraded has positive primary and negative WER delta CIs."""
    data = compare_candidates(benchmark_settings)
    assert data.candidates == ("identity", "degraded")
    assert set(data.provenance) >= {"turns", "calls", "public_leaderboards_used"}
    assert {evaluation.name for evaluation in data.sets} == {
        "golden_standin",
        "general_capability",
    }
    for evaluation in data.sets:
        assert evaluation.turns == 36
        assert set(evaluation.deltas) == set(CORE_METRIC_NAMES)
        for metric, delta in evaluation.deltas.items():
            assert isinstance(metric, str) and metric
            assert delta["level"] == 0.95
            assert delta["lower"] <= delta["point"] <= delta["upper"]
        primary = evaluation.deltas["e2e_booking_accuracy"]
        assert primary["lower"] > 0.0
        assert evaluation.deltas["wer"]["upper"] < 0.0
        assert evaluation.metrics_by_candidate["identity"]["e2e_booking_accuracy"] == pytest.approx(
            30 / 36
        )
        assert (
            evaluation.metrics_by_candidate["degraded"]["e2e_booking_accuracy"]
            < evaluation.metrics_by_candidate["identity"]["e2e_booking_accuracy"]
        )


def test_reversed_candidate_order_reverses_delta_sign(
    benchmark_settings: BenchmarkSettings,
) -> None:
    """Candidate order defines the delta direction, not the runner names."""
    reversed_settings = benchmark_settings.model_copy(
        update={"candidates_csv": "degraded,identity"}
    )
    data = compare_candidates(reversed_settings)
    assert data.candidates == ("degraded", "identity")
    for evaluation in data.sets:
        assert evaluation.deltas["e2e_booking_accuracy"]["upper"] < 0.0
        assert evaluation.deltas["wer"]["lower"] > 0.0


def test_comparison_requires_exactly_two_candidates(
    benchmark_settings: BenchmarkSettings,
) -> None:
    """A single candidate cannot form an A - B comparison."""
    with pytest.raises(ConfigError, match="exactly two"):
        compare_candidates(benchmark_settings.model_copy(update={"candidates_csv": "identity"}))


def test_tampered_golden_copy_is_rejected(
    benchmark_settings: BenchmarkSettings, tmp_path: Path, repo_root: Path
) -> None:
    """A modified copy of the seal fails before scoring; the source stays intact."""
    source = repo_root / benchmark_settings.golden_artifact_dir
    source_before = _golden_hashes(source)
    copy = tmp_path / "golden-copy"
    shutil.copytree(source, copy)
    labels_path = copy / "golden.v1.labels.json"
    document: dict[str, Any] = json.loads(labels_path.read_text(encoding="utf-8"))
    document["labels"][0]["corrected_transcript"] = "tampered"
    labels_path.write_text(json.dumps(document), encoding="utf-8")

    settings = benchmark_settings.model_copy(update={"golden_artifact_dir": str(copy)})
    with pytest.raises(ValidationError):
        compare_candidates(settings)
    assert _golden_hashes(source) == source_before
