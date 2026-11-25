"""CLI and orchestration for deterministic synthetic augmentation (P3-03).

Run with ``python -m services.datasets.augment.cli``. The P2-05 manifest and
sealed golden stand-in are read-only inputs; no TTS engine, codec, network, GPU,
promotion threshold or MLflow alias is touched. All generated rows are labelled
stand-ins and are filtered against injected golden/regression call lists.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from voiceloop_common import ConfigError, ValidationError, validate

from .manifest import (
    AugmentedVariant,
    SourceHashes,
    TransformRecord,
    assert_source_hashes_unchanged,
    build_manifest,
    load_dataset_manifest,
    snapshot_source_hashes,
    write_manifest,
)
from .noise import NoiseProfile, TelephonyBand, add_line_noise, apply_telephony_band
from .settings import AugmentSettings
from .variants import (
    AudioStandIn,
    DeterministicFakeTts,
    TtsBackend,
    build_variant_specs,
)


@dataclass(frozen=True, slots=True)
class AugmentationRun:
    """Result of one augmentation batch with before/after provenance."""

    manifest: dict[str, Any]
    source_hashes_before: SourceHashes
    source_hashes_after: SourceHashes
    eligible_calls: tuple[str, ...]
    variants: tuple[AugmentedVariant, ...]


def _parse_optional_csv(raw: str | None) -> tuple[str, ...] | None:
    """Parse an optional CLI CSV; ``None`` means 'use the safe default'."""
    if raw is None:
        return None
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _default_golden_ids(settings: AugmentSettings) -> tuple[str, ...]:
    """Load sealed golden call IDs read-only through the P1-06 public API."""
    from services.datasets import load_golden_call_ids

    return tuple(load_golden_call_ids(path=settings.golden_manifest_path))


def build_augmentation(
    settings: AugmentSettings | None = None,
    *,
    tts: TtsBackend | None = None,
    dataset_manifest: Mapping[str, Any] | None = None,
    golden_call_ids: Sequence[str] | None = None,
    regression_call_ids: Sequence[str] | None = None,
    created_at: str | None = None,
) -> AugmentationRun:
    """Generate tagged variants, prove exclusions and verify read-only inputs.

    Args:
        settings: Typed ``VOICELOOP_AUGMENT_`` settings; defaults to a fresh object.
        tts: Optional :class:`~services.datasets.augment.variants.TtsBackend`;
            the deterministic fake is used when omitted.
        dataset_manifest: Optional in-memory read-only P2-05 manifest; otherwise
            loaded and schema-validated from ``settings.dataset_manifest_path``.
        golden_call_ids: Injected sealed evaluation calls; defaults to the
            read-only sealed stand-in loaded through the P1-06 public API.
        regression_call_ids: Injected regression calls to exclude.
        created_at: Optional timestamp override for deterministic manifests.

    Returns:
        Validated manifest plus before/after hashes of read-only inputs.

    Raises:
        ConfigError: If the P2-05 manifest input is missing or unreadable.
        ValidationError: If hashes, lineage, exclusion or PII checks fail.

    """
    config = settings or AugmentSettings()
    before = snapshot_source_hashes(config)
    if dataset_manifest is not None:
        source = dict(dataset_manifest)
        validate(source, "dataset_manifest")
    else:
        source = load_dataset_manifest(config)
    golden = tuple(golden_call_ids) if golden_call_ids is not None else _default_golden_ids(config)
    regression = tuple(regression_call_ids) if regression_call_ids is not None else ()
    excluded = set(golden) | set(regression)
    eligible_calls = tuple(sorted(set(source["source_call_ids"]) - excluded))

    backend = tts or DeterministicFakeTts()
    noise = NoiseProfile(level=config.noise_level)
    telephony = TelephonyBand(
        low_hz=config.telephony_low_hz,
        high_hz=config.telephony_high_hz,
        sample_rate_hz=config.telephony_sample_rate_hz,
    )
    generated: list[AugmentedVariant] = []
    for spec in build_variant_specs(eligible_calls, config):
        samples = backend.synthesize(
            token=spec.token,
            sample_rate_hz=spec.sample_rate_hz,
            duration_ms=spec.duration_ms,
            seed=spec.seed,
        )
        records = [TransformRecord(name="fake_tts", params=backend.params(spec=spec))]
        if spec.noisy:
            samples = add_line_noise(samples, seed=spec.seed, level=noise.level, profile=noise.name)
            records.append(TransformRecord(name="line_noise", params=noise.params(seed=spec.seed)))
        if spec.telephony:
            samples = apply_telephony_band(samples, band=telephony)
            records.append(TransformRecord(name="telephony_band", params=telephony.params()))
        audio = AudioStandIn(samples, spec.sample_rate_hz, spec.duration_ms)
        generated.append(AugmentedVariant(spec=spec, audio=audio, transforms=tuple(records)))

    manifest = build_manifest(
        generated,
        source_manifest=source,
        golden_call_ids=golden,
        regression_call_ids=regression,
        created_at=created_at,
    )
    after = assert_source_hashes_unchanged(config, before)
    return AugmentationRun(
        manifest=manifest,
        source_hashes_before=before,
        source_hashes_after=after,
        eligible_calls=eligible_calls,
        variants=tuple(generated),
    )


def _build_parser(base: AugmentSettings) -> argparse.ArgumentParser:
    """Build the augmentation CLI parser from typed defaults."""
    parser = argparse.ArgumentParser(
        prog="python -m services.datasets.augment.cli",
        description=(
            "Generate deterministic SYNTHETIC/FIXTURE STAND-IN variants from a "
            "read-only P2-05 dataset manifest, excluding golden/regression calls."
        ),
    )
    parser.add_argument(
        "--manifest", default=base.dataset_manifest_path, help="P2-05 manifest path."
    )
    parser.add_argument("--output", default=base.output_path, help="Output manifest path.")
    parser.add_argument(
        "--golden-ids",
        default=None,
        help="Comma-separated sealed call IDs; omit to load the sealed stand-in; "
        "pass an empty string only for fixture rehearsals.",
    )
    parser.add_argument(
        "--regression-ids", default=None, help="Comma-separated regression call IDs."
    )
    parser.add_argument("--golden-manifest", default=base.golden_manifest_path)
    parser.add_argument("--fixtures", default=base.fixtures_path)
    parser.add_argument("--seed", type=int, default=base.seed)
    parser.add_argument("--duration-ms", type=int, default=base.duration_ms)
    parser.add_argument("--noise-level", type=int, default=base.noise_level)
    parser.add_argument("--failure-classes", default=base.failure_classes_csv)
    parser.add_argument("--name-tokens", default=base.name_tokens_csv)
    return parser


def _error_code(exc: Exception) -> str:
    """Map a known exception class to a stable structured CLI error code."""
    if isinstance(exc, ConfigError):
        return "config_error"
    if isinstance(exc, ValidationError):
        return "validation_error"
    return "internal_error"


def main(argv: Sequence[str] | None = None) -> int:
    """Run augmentation, write one manifest and print a PII-free summary.

    Returns:
        0 on success; 2 on a known config/validation error with a structured
        ``{code, message}`` object on stderr.

    """
    base = AugmentSettings()
    args = _build_parser(base).parse_args(argv)
    updates: dict[str, object] = {
        "dataset_manifest_path": args.manifest,
        "output_path": args.output,
        "golden_manifest_path": args.golden_manifest,
        "fixtures_path": args.fixtures,
        "seed": args.seed,
        "duration_ms": args.duration_ms,
        "noise_level": args.noise_level,
        "failure_classes_csv": args.failure_classes,
        "name_tokens_csv": args.name_tokens,
    }
    try:
        settings = AugmentSettings.model_validate({**base.model_dump(), **updates})
        golden_ids = _parse_optional_csv(args.golden_ids)
        regression_ids = _parse_optional_csv(args.regression_ids)
        run = build_augmentation(
            settings,
            golden_call_ids=golden_ids,
            regression_call_ids=regression_ids or (),
        )
        destination = write_manifest(
            run.manifest,
            settings.output_path,
            settings=settings,
            forbidden_paths=(settings.dataset_manifest_path,),
        )
    except (ConfigError, ValidationError, PydanticValidationError) as exc:
        print(json.dumps({"code": _error_code(exc), "message": str(exc)}), file=sys.stderr)
        return 2

    manifest = run.manifest
    print(f"Wrote augmentation manifest: {destination}")
    print(
        f"Synthetic variants: {manifest['counts']['variants']}; "
        f"source calls: {manifest['counts']['source_calls']}; "
        f"transforms: {','.join(manifest['transform_names'])}"
    )
    print(f"Golden content hash unchanged: {run.source_hashes_after.golden_content_hash[:12]}")
    print(f"Fixture hash unchanged: {run.source_hashes_after.fixtures_sha256[:12]}")
    print("SYNTHETIC/FIXTURE STAND-IN - NOT REAL AUDIO; no golden/regression rows written.")
    print("Promotion: not_evaluated (no thresholds, aliases or gate decisions changed).")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via python -m
    raise SystemExit(main())
