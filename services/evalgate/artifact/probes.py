"""Deterministic synthetic probes and safety metrics for STT artifacts.

All probe inputs come from the committed synthetic fixtures and the read-only
sealed v1 golden stand-in. No real audio, human label, model runtime, network or
GPU is involved. Probe results deliberately contain counts and error codes, not
transcripts, so reports and structured errors cannot carry fixture text or PII.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import fmean
from typing import Any

from services.datasets.golden.seal import verify_seal
from services.evalgate.bootstrap import ConfidenceInterval, bootstrap_deltas, percentile_interval
from services.evalgate.harness import load_event, load_fixture_rows, slice_names_for_row
from services.evalgate.metrics import EvaluationError, tokenize
from voiceloop_common import ValidationError

from .artifact import FakeArtifact, create_fake_artifact
from .decoding import DecodeCombination, DecodingSettings
from .settings import ArtifactSettings

REPO_ROOT = Path(__file__).resolve().parents[3]
SILENCE_KINDS = frozenset({"silence", "noise"})


def _resolve(path: str | Path) -> Path:
    """Resolve a repo-relative path against the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def verify_inputs(settings: ArtifactSettings) -> dict[str, Any]:
    """Verify committed fixture and golden hashes without modifying either.

    Seal verification is called with no audit context, so no audit entry is
    appended. The golden stand-in is never trained on or tuned on.

    Args:
        settings: Typed settings carrying expected content hashes.

    Returns:
        Provenance fields for the generated report.

    Raises:
        ValidationError: If either committed hash or the golden seal changed.

    """
    fixture_path = _resolve(settings.fixtures_path)
    try:
        fixture_hash = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValidationError(f"cannot read fixture manifest: {fixture_path}") from exc
    if fixture_hash != settings.expected_fixture_sha256:
        raise ValidationError(
            f"committed fixture hash changed; refusing artifact evaluation: {fixture_hash!r}"
        )
    verification = verify_seal(settings.golden_artifact_dir, settings.golden_version)
    verification.assert_ok()
    manifest = verification.manifest
    if manifest is None:  # pragma: no cover - guaranteed by a passing seal
        raise ValidationError("verified golden stand-in has no manifest")
    if manifest.content_hash != settings.expected_golden_content_hash:
        raise ValidationError(
            "golden stand-in content hash changed; refusing artifact evaluation: "
            f"{manifest.content_hash!r}"
        )
    if not manifest.synthetic or not manifest.workflow_stand_in:
        raise ValidationError("golden stand-in is no longer labelled synthetic/workflow stand-in")
    if manifest.review_mode != "simulated_synthetic_labels":
        raise ValidationError("golden stand-in review mode is not the expected synthetic mode")
    return {
        "fixtures_path": settings.fixtures_path,
        "fixture_sha256": fixture_hash,
        "fixture_hash_verified": True,
        "golden_artifact_dir": settings.golden_artifact_dir,
        "golden_version": manifest.version,
        "golden_content_hash": manifest.content_hash,
        "golden_seal_ok": True,
        "golden_synthetic": manifest.synthetic,
        "golden_workflow_stand_in": manifest.workflow_stand_in,
        "golden_review_mode": manifest.review_mode,
        "golden_used_for_training_or_tuning": False,
        "real_audio_or_human_labels_used": False,
        "limitation": (
            "SYNTHETIC/FIXTURE STAND-IN: tone/noise/silence carriers and simulated "
            "labels, not human-reviewed audio or a deployed artifact."
        ),
    }


@dataclass(frozen=True)
class ProbeInputs:
    """Read-only fixture rows, events, prior-hypothesis paths and provenance."""

    rows: tuple[dict[str, Any], ...]
    events: tuple[Mapping[str, Any], ...]
    prior_turn_counts: tuple[int, ...]
    provenance: Mapping[str, Any]


def _load_prior_turn_count(event: Mapping[str, Any]) -> int:
    """Count committed prior-turn hypotheses for the previous-text probe seam."""
    context = event.get("dialogue_context")
    if not isinstance(context, Mapping):
        return 0
    reference = context.get("prior_turns_ref")
    if not isinstance(reference, str) or not reference:
        return 0
    path = _resolve(reference)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvaluationError(f"cannot read fixture context: {path}") from exc
    count = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except ValueError as exc:
            raise EvaluationError(f"invalid fixture context line: {path}") from exc
        if isinstance(decoded, Mapping) and isinstance(decoded.get("turn_index"), int):
            count += 1
    return count


def load_probe_inputs(settings: ArtifactSettings) -> ProbeInputs:
    """Load and align all committed synthetic probe inputs."""
    rows = tuple(load_fixture_rows(settings.fixtures_path))
    events = tuple(load_event(str(row["event_path"])) for row in rows)
    counts = tuple(_load_prior_turn_count(event) for event in events)
    provenance = verify_inputs(settings)
    return ProbeInputs(rows=rows, events=events, prior_turn_counts=counts, provenance=provenance)


def fake_transcribe(
    artifact: FakeArtifact,
    row: Mapping[str, Any],
    event: Mapping[str, Any],
    decoding: DecodingSettings,
    settings: ArtifactSettings,
    prior_turn_count: int,
) -> tuple[str, dict[str, Any]]:
    """Produce one deterministic transcript and slot dict from a fake artifact.

    Args:
        artifact: Content-addressed fake artifact with an injected behaviour.
        row: Committed synthetic label row (used for its audio kind only).
        event: Committed synthetic turn event replayed by the healthy fake.
        decoding: :class:`~services.evalgate.artifact.decoding.DecodingSettings`.
        settings: Typed settings supplying deterministic injection parameters.
        prior_turn_count: Number of committed prior turns (previous-text seam).

    Returns:
        Predicted transcript and parser slots. Neither is stored in reports.

    """
    del prior_turn_count  # Recorded as a probe seam; the fakes ignore its content.
    stt = event.get("stt")
    parser = event.get("parser")
    baseline_text = str(stt.get("text", "")) if isinstance(stt, Mapping) else ""
    baseline_slots = (
        dict(parser.get("output", {}))
        if isinstance(parser, Mapping) and isinstance(parser.get("output"), Mapping)
        else {}
    )
    reference = str(row.get("expected_transcript", ""))
    audio_kind = str(row.get("audio_kind", ""))
    if audio_kind in SILENCE_KINDS and not tokenize(reference):
        # Committed synthetic labels mark these turns as having no speech; the
        # healthy fake returns nothing rather than replaying the deliberately
        # hallucinated STT text stored in the fixture event.
        baseline_text, baseline_slots = "", {}
    if artifact.behaviour == "empty":
        return "", {}
    if artifact.behaviour == "truncated":
        tokens = tokenize(baseline_text)
        keep = max(1, int(len(tokens) * settings.truncation_ratio))
        return " ".join(tokens[:keep]), {}
    if artifact.behaviour == "hallucinating":
        if audio_kind in SILENCE_KINDS and not tokenize(reference):
            return settings.hallucination_text, {}
        return baseline_text, baseline_slots
    if artifact.behaviour == "forgetting":
        if str(decoding.language).lower() != settings.language.strip().lower():
            return "", {}
        return baseline_text, baseline_slots
    return baseline_text, baseline_slots


@dataclass(frozen=True)
class ProbeError:
    """A structured, PII-free failure produced by one artifact probe."""

    code: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable structured error."""
        return {"code": self.code, "message": self.message, "details": dict(self.details)}


@dataclass(frozen=True)
class ProbeResult:
    """Per-turn probe outcome with counts only, never transcript text."""

    combination: str
    case_id: str | None
    call_id: str
    turn_index: int
    audio_kind: str
    slices: tuple[str, ...]
    status: str
    reference_token_count: int
    predicted_token_count: int
    baseline_token_count: int
    errors: tuple[ProbeError, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ProbeSuiteResult:
    """Aggregate of all deterministic combination probes for one artifact."""

    results: tuple[ProbeResult, ...]
    errors: tuple[ProbeError, ...]
    combinations: tuple[str, ...]
    status: str

    def summary(self, max_details: int = 25) -> dict[str, Any]:
        """Return counts and a bounded list of structured error details."""
        counts = Counter(error.code for error in self.errors)
        return {
            "status": self.status,
            "turns": len(self.results),
            "combinations": list(self.combinations),
            "error_count": len(self.errors),
            "errors_by_code": dict(sorted(counts.items())),
            "error_details": [error.to_dict() for error in self.errors[:max_details]],
            "label_source": "committed synthetic fixtures",
        }


def _classify_probe(
    *,
    combination: str,
    row: Mapping[str, Any],
    reference: str,
    transcript: str,
    baseline: str,
    settings: ArtifactSettings,
    slices: tuple[str, ...],
    multilingual: bool,
) -> tuple[str, tuple[ProbeError, ...]]:
    """Detect empty, truncated, hallucinated and forgotten outputs."""
    details: dict[str, Any] = {
        "case_id": row.get("case_id"),
        "combination": combination,
        "call_id": row.get("call_id"),
        "turn_index": row.get("turn_index"),
        "reference_token_count": len(tokenize(reference)),
        "predicted_token_count": len(tokenize(transcript)),
    }
    reference_tokens = tokenize(reference)
    predicted_tokens = tokenize(transcript)
    baseline_tokens = tokenize(baseline)
    errors: list[ProbeError] = []
    if multilingual and reference_tokens and not predicted_tokens:
        errors.append(
            ProbeError(
                "multilingual_forgetting",
                "fake artifact produced no output for a served non-English language",
                details,
            )
        )
    elif reference_tokens and not predicted_tokens:
        errors.append(
            ProbeError(
                "empty_output",
                "fake artifact returned an empty transcript for a speaking turn",
                details,
            )
        )
    elif (
        baseline_tokens
        and predicted_tokens
        and len(predicted_tokens) < len(baseline_tokens)
        and tuple(baseline_tokens[: len(predicted_tokens)]) == tuple(predicted_tokens)
    ):
        errors.append(
            ProbeError(
                "truncated_output",
                "fake artifact returned a prefix-truncated transcript",
                {**details, "baseline_token_count": len(baseline_tokens)},
            )
        )
    if (
        not reference_tokens
        and predicted_tokens
        and str(row.get("audio_kind", "")) in SILENCE_KINDS
    ):
        errors.append(
            ProbeError(
                "silence_hallucination",
                "fake artifact hallucinated text on silence or line noise",
                details,
            )
        )
    if not errors:
        return "ok", ()
    return errors[0].code, tuple(errors)


def run_probe_suite(
    inputs: ProbeInputs,
    artifact: FakeArtifact,
    combinations: Sequence[DecodeCombination],
    settings: ArtifactSettings,
) -> ProbeSuiteResult:
    """Run every deterministic combination over every committed fixture turn.

    Args:
        inputs: Read-only probe inputs.
        artifact: Content-addressed fake artifact under test.
        combinations: Decoding combinations to exercise.
        settings: Typed settings for slice classification and error limits.

    Returns:
        A suite result; ``status`` is ``fail`` if any structured error fired.

    """
    if len({len(inputs.rows), len(inputs.events), len(inputs.prior_turn_counts)}) != 1:
        raise ValidationError("probe inputs are not aligned")
    results: list[ProbeResult] = []
    errors: list[ProbeError] = []
    for combination in combinations:
        for row, event, prior_count in zip(
            inputs.rows, inputs.events, inputs.prior_turn_counts, strict=True
        ):
            transcript, _ = fake_transcribe(
                artifact, row, event, combination.settings, settings, prior_count
            )
            stt = event.get("stt")
            baseline = str(stt.get("text", "")) if isinstance(stt, Mapping) else ""
            status, probe_errors = _classify_probe(
                combination=combination.name,
                row=row,
                reference=str(row.get("expected_transcript", "")),
                transcript=transcript,
                baseline=baseline,
                settings=settings,
                slices=tuple(
                    sorted(slice_names_for_row(row, None, settings.other_language_pattern))
                ),
                multilingual=combination.settings.language.lower() != settings.language.lower(),
            )
            result = ProbeResult(
                combination=combination.name,
                case_id=row.get("case_id"),
                call_id=str(row.get("call_id", "")),
                turn_index=int(row.get("turn_index", -1)),
                audio_kind=str(row.get("audio_kind", "")),
                slices=tuple(
                    sorted(slice_names_for_row(row, None, settings.other_language_pattern))
                ),
                status=status,
                reference_token_count=len(tokenize(str(row.get("expected_transcript", "")))),
                predicted_token_count=len(tokenize(transcript)),
                baseline_token_count=len(tokenize(baseline)),
                errors=probe_errors,
            )
            results.append(result)
            errors.extend(probe_errors)
    return ProbeSuiteResult(
        results=tuple(results),
        errors=tuple(errors),
        combinations=tuple(combination.name for combination in combinations),
        status="fail" if errors else "pass",
    )


def _indicator_interval(
    champion: Sequence[float],
    challenger: Sequence[float],
    settings: ArtifactSettings,
) -> ConfidenceInterval:
    """Return a paired-bootstrap CI for a challenger-champion rate difference."""
    deltas = bootstrap_deltas(
        champion,
        challenger,
        iterations=settings.bootstrap_iterations,
        seed=settings.bootstrap_seed,
    )
    return percentile_interval(deltas, settings.bootstrap_level)


def hallucination_summary(
    inputs: ProbeInputs,
    artifact: FakeArtifact,
    combination: DecodeCombination,
    settings: ArtifactSettings,
) -> dict[str, Any]:
    """Measure no-speech hallucination on committed silence/noise fixtures.

    The rate is the fraction of empty-reference silence/noise turns where the
    artifact emits at least one token. Its CI bounds the paired challenger minus
    healthy-fake difference; it is a diagnostic, never a promotion metric.
    """
    indices = [
        index
        for index, row in enumerate(inputs.rows)
        if str(row.get("audio_kind", "")) in SILENCE_KINDS
        and not tokenize(str(row.get("expected_transcript", "")))
    ]
    if not indices:
        raise EvaluationError("committed fixtures contain no silence/no-speech turns")

    def indicators(candidate: FakeArtifact) -> list[float]:
        values: list[float] = []
        for index in indices:
            transcript, _ = fake_transcribe(
                candidate,
                inputs.rows[index],
                inputs.events[index],
                combination.settings,
                settings,
                inputs.prior_turn_counts[index],
            )
            values.append(1.0 if tokenize(transcript) else 0.0)
        return values

    challenger = indicators(artifact)
    interval = _indicator_interval(
        indicators(create_fake_artifact("healthy")), challenger, settings
    )
    cases = sorted(
        str(inputs.rows[index].get("case_id") or "unnamed")
        for position, index in enumerate(indices)
        if challenger[position] > 0
    )
    return {
        "turns": len(indices),
        "hallucinated_turns": int(sum(challenger)),
        "rate": fmean(challenger),
        "delta_ci": {
            "lower": interval.lower,
            "upper": interval.upper,
            "level": interval.level,
        },
        "firing_cases": cases,
        "label_source": "committed synthetic fixture audio_kind and empty expected_transcript",
        "promotion_metric": False,
    }


def multilingual_summary(
    inputs: ProbeInputs,
    artifact: FakeArtifact,
    combination: DecodeCombination,
    settings: ArtifactSettings,
) -> dict[str, Any]:
    """Measure forgetting on synthetic language-tagged fixture surrogates.

    With the roadmap default (English only) there are no committed multilingual
    fixtures, so the result is explicitly ``not_evaluated`` and the report's
    gate-visible ``other_languages`` slice keeps zero turns. Configured non-English
    languages produce a deterministic surrogate diagnostic only: no human label
    is invented and the result must never be used as promotion evidence.
    """
    extra_languages = settings.non_english_languages()
    base_status = {
        "served_languages": list(settings.served_languages()),
        "non_english_languages": list(extra_languages),
        "surrogate_only": True,
        "human_labels": False,
        "gate_evidence": False,
        "label_source": "committed synthetic fixtures with a deterministic language tag",
    }
    if not extra_languages:
        interval = _indicator_interval([0.0], [0.0], settings)
        return {
            **base_status,
            "turns": 0,
            "surrogate_turns": 0,
            "forgetting_rate": 0.0,
            "delta_ci": {
                "lower": interval.lower,
                "upper": interval.upper,
                "level": interval.level,
            },
            "status": "not_evaluated_no_committed_multilingual_fixtures",
        }

    def indicators(candidate: FakeArtifact) -> list[float]:
        values: list[float] = []
        for language in extra_languages:
            decode = replace(combination.settings, language=language)
            for row, event, prior_count in zip(
                inputs.rows, inputs.events, inputs.prior_turn_counts, strict=True
            ):
                transcript, _ = fake_transcribe(
                    candidate, row, event, decode, settings, prior_count
                )
                stt = event.get("stt")
                baseline = str(stt.get("text", "")) if isinstance(stt, Mapping) else ""
                reference_tokens = tokenize(str(row.get("expected_transcript", "")))
                predicted_tokens = tokenize(transcript)
                baseline_tokens = tokenize(baseline)
                degraded = bool(reference_tokens) and (
                    not predicted_tokens
                    or (
                        len(predicted_tokens) < len(baseline_tokens)
                        and tuple(baseline_tokens[: len(predicted_tokens)])
                        == tuple(predicted_tokens)
                    )
                )
                values.append(1.0 if degraded else 0.0)
        return values

    challenger = indicators(artifact)
    interval = _indicator_interval(
        indicators(create_fake_artifact("healthy")), challenger, settings
    )
    return {
        **base_status,
        "turns": len(extra_languages) * len(inputs.rows),
        "surrogate_turns": len(extra_languages) * len(inputs.rows),
        "forgetting_rate": fmean(challenger),
        "delta_ci": {
            "lower": interval.lower,
            "upper": interval.upper,
            "level": interval.level,
        },
        "status": "diagnostic_only_synthetic_language_tag",
    }


__all__ = [
    "ProbeError",
    "ProbeInputs",
    "ProbeResult",
    "ProbeSuiteResult",
    "fake_transcribe",
    "hallucination_summary",
    "load_probe_inputs",
    "multilingual_summary",
    "run_probe_suite",
    "verify_inputs",
]
