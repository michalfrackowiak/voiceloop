"""Conservative LoRA backend seam for the Whisper trainer (P3-01.1).

SYNTHETIC / FIXTURE TRAINING STAND-IN. ``FakeTransformersPeftBackend`` performs
no torch/transformers/PEFT work and produces deterministic pseudo-deltas, never
learned weights.

Evidence for conservative scaling: ``context.md`` §10 records that over-scaled
LoRA can produce hallucinated text on silence and cites a single reported fix
from effective scaling 2.0 down to 0.2 (explicitly flagged as one data point to
verify on our own data). The P2-08 fixture set includes a
``silence_hallucination`` case, so the risk class is present in our rehearsal
data. This module therefore defaults to alpha=1.0, r=8 (effective scaling
0.125, well below 2.0 and below the reported 0.2 fix) and hard-fails above a
configurable ceiling of 2.0. This is engineering guidance only, **not a tuned
production value**: a real fine-tune must re-check silence and other-language
slices on the exact converted artifact (P3-04) before any promotion claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from voiceloop_common import ValidationError

from .dataset import (
    STANDIN_WARNING as STANDIN_WARNING,
)
from .dataset import (
    WhisperTrainingExample,
    canonical_json,
    sha256_text,
)

DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 1.0
MAX_LORA_SCALING = 2.0
ADAPTER_VERSION = "1.0.0"
_DIGEST_LENGTH = 64


@dataclass(frozen=True, slots=True)
class LoraConfig:
    """Validated LoRA shape and its effective scaling ``alpha / r``."""

    r: int = DEFAULT_LORA_R
    alpha: float = DEFAULT_LORA_ALPHA
    max_scaling: float = MAX_LORA_SCALING

    @property
    def scaling(self) -> float:
        """Return the effective LoRA scaling factor."""
        return self.alpha / self.r

    def validate(self) -> None:
        """Fail closed on invalid shapes or non-conservative scaling."""
        if isinstance(self.r, bool) or not isinstance(self.r, int) or self.r < 1:
            raise ValidationError("LoRA rank must be a positive integer")
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, (int, float)):
            raise ValidationError("LoRA alpha must be numeric")
        if self.alpha <= 0:
            raise ValidationError("LoRA alpha must be positive")
        if isinstance(self.max_scaling, bool) or not isinstance(self.max_scaling, (int, float)):
            raise ValidationError("LoRA max_scaling must be numeric")
        if self.max_scaling <= 0 or self.max_scaling > MAX_LORA_SCALING:
            raise ValidationError("LoRA max_scaling must be positive and at most 2.0")
        if self.scaling > self.max_scaling:
            raise ValidationError(
                f"LoRA scaling alpha/r={self.scaling:.6f} exceeds bound {self.max_scaling}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable LoRA configuration record."""
        return {
            "alpha": self.alpha,
            "max_scaling": self.max_scaling,
            "r": self.r,
            "scaling": self.scaling,
        }


def validate_conservative_lora(
    r: int, alpha: float, max_scaling: float = MAX_LORA_SCALING
) -> LoraConfig:
    """Build and validate a conservative LoRA configuration.

    Args:
        r: LoRA rank.
        alpha: LoRA scaling alpha.
        max_scaling: Hard upper bound for ``alpha / r`` (default 2.0).

    Returns:
        The validated :class:`LoraConfig`.

    Raises:
        ValidationError: If the shape or scaling is unsafe.

    """
    config = LoraConfig(r=r, alpha=alpha, max_scaling=max_scaling)
    config.validate()
    return config


@dataclass(frozen=True, slots=True)
class LoraDelta:
    """One deterministic pseudo-weight update keyed by example digest."""

    example_id: str
    target_text_sha256: str
    sample_rate_hz: int
    replay: bool
    weight: float

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, JSON-serialisable delta record."""
        return {
            "example_id": self.example_id,
            "replay": self.replay,
            "sample_rate_hz": self.sample_rate_hz,
            "target_text_sha256": self.target_text_sha256,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> LoraDelta:
        """Validate a decoded delta before it enters an adapter artifact."""
        expected = {"example_id", "replay", "sample_rate_hz", "target_text_sha256", "weight"}
        if set(raw) != expected:
            raise ValidationError("LoRA delta has unexpected fields")
        if not isinstance(raw["example_id"], str) or not raw["example_id"]:
            raise ValidationError("LoRA delta example_id must be a non-empty string")
        if (
            not isinstance(raw["target_text_sha256"], str)
            or len(raw["target_text_sha256"]) != _DIGEST_LENGTH
        ):
            raise ValidationError("LoRA delta target_text_sha256 must be a sha256 hex digest")
        rate = raw["sample_rate_hz"]
        if isinstance(rate, bool) or not isinstance(rate, int) or rate < 1000:
            raise ValidationError("LoRA delta sample_rate_hz must be a positive integer")
        if not isinstance(raw["replay"], bool):
            raise ValidationError("LoRA delta replay must be a boolean")
        weight = raw["weight"]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValidationError("LoRA delta weight must be numeric")
        return cls(
            example_id=str(raw["example_id"]),
            target_text_sha256=str(raw["target_text_sha256"]),
            sample_rate_hz=int(rate),
            replay=bool(raw["replay"]),
            weight=float(weight),
        )


@dataclass(frozen=True, slots=True)
class FakeLoraAdapter:
    """Content-hash-addressed stand-in LoRA adapter descriptor.

    Contains hyperparameters, effective scaling and digest-only deltas. It is
    never a trained checkpoint and is labelled as such on every export.
    """

    dataset_hash: str
    data_manifest_hash: str
    base_model: str
    lora: dict[str, Any]
    training: dict[str, Any]
    deltas: tuple[LoraDelta, ...]
    schema_version: str = ADAPTER_VERSION
    synthetic_training_standin: bool = True
    warning: str = STANDIN_WARNING

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical adapter payload for hashing and export."""
        return {
            "base_model": self.base_model,
            "data_manifest_hash": self.data_manifest_hash,
            "dataset_hash": self.dataset_hash,
            "deltas": [
                delta.to_dict() for delta in sorted(self.deltas, key=lambda d: d.example_id)
            ],
            "lora": dict(self.lora),
            "schema_version": self.schema_version,
            "synthetic_training_standin": self.synthetic_training_standin,
            "training": dict(self.training),
            "warning": self.warning,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FakeLoraAdapter:
        """Validate a decoded adapter before it enters the merge seam."""
        expected = {
            "base_model",
            "data_manifest_hash",
            "dataset_hash",
            "deltas",
            "lora",
            "schema_version",
            "synthetic_training_standin",
            "training",
            "warning",
        }
        if set(raw) != expected:
            raise ValidationError("fake LoRA adapter has unexpected fields")
        if raw["schema_version"] != ADAPTER_VERSION:
            raise ValidationError(f"unsupported adapter version: {raw['schema_version']!r}")
        for key in ("dataset_hash", "data_manifest_hash"):
            value = raw[key]
            if not isinstance(value, str) or len(value) != _DIGEST_LENGTH:
                raise ValidationError(f"adapter {key} must be a sha256 hex digest")
        if not isinstance(raw["base_model"], str) or not isinstance(raw["training"], Mapping):
            raise ValidationError("adapter base_model/training fields are malformed")
        if not isinstance(raw["lora"], Mapping):
            raise ValidationError("adapter lora field must be an object")
        lora = dict(raw["lora"])
        validate_conservative_lora(int(lora["r"]), float(lora["alpha"]), float(lora["max_scaling"]))
        raw_deltas = raw["deltas"]
        if not isinstance(raw_deltas, list) or not raw_deltas:
            raise ValidationError("adapter deltas must be a non-empty list")
        deltas = tuple(LoraDelta.from_dict(delta) for delta in raw_deltas)
        if len({delta.example_id for delta in deltas}) != len(deltas):
            raise ValidationError("adapter deltas contain duplicate example_id values")
        return cls(
            dataset_hash=str(raw["dataset_hash"]),
            data_manifest_hash=str(raw["data_manifest_hash"]),
            base_model=str(raw["base_model"]),
            lora=lora,
            training=dict(raw["training"]),
            deltas=deltas,
            schema_version=str(raw["schema_version"]),
            synthetic_training_standin=bool(raw["synthetic_training_standin"]),
            warning=str(raw["warning"]),
        )

    def content_hash(self) -> str:
        """Return the SHA-256 content hash of the canonical adapter bytes."""
        return sha256_text(canonical_json(self.to_dict()))


class LoraTrainingBackend(Protocol):
    """Injectable training seam; a real one owns transformers/PEFT imports."""

    def train(
        self,
        examples: Sequence[WhisperTrainingExample],
        *,
        model: str,
        seed: int,
        dataset_hash: str,
        data_manifest_hash: str,
        params: Mapping[str, Any],
        lora: LoraConfig,
    ) -> FakeLoraAdapter:
        """Train on digest-only examples and return a hash-addressed adapter."""
        ...


class FakeTransformersPeftBackend:
    """Deterministic stand-in named after the intended real stack.

    No ``torch``, ``transformers`` or ``peft`` module is imported. Weights are
    derived from a stable hash so the artifact is reproducible from examples,
    seed, base model, hyperparameters and LoRA scaling.
    """

    def train(
        self,
        examples: Sequence[WhisperTrainingExample],
        *,
        model: str,
        seed: int,
        dataset_hash: str,
        data_manifest_hash: str,
        params: Mapping[str, Any],
        lora: LoraConfig,
    ) -> FakeLoraAdapter:
        """Derive reproducible pseudo-deltas from the mixed examples.

        Args:
            examples: Primary plus replay examples, already sorted.
            model: Base model ID from core settings; recorded only as a digest
                input, never hard-coded here.
            seed: Core job seed.
            dataset_hash: P2-05 manifest ``content_hash``.
            data_manifest_hash: Hash of the task-local mix manifest.
            params: Whisper hyperparameters from
                :meth:`~.settings.WhisperSettings.as_training_params`.
            lora: Validated conservative LoRA configuration.

        Returns:
            A labelled fake adapter whose deltas are sorted by example ID.

        Raises:
            ValidationError: If the LoRA configuration is not conservative.

        """
        lora.validate()
        prefix = "|".join(
            (
                str(seed),
                model,
                canonical_json(dict(params)),
                canonical_json(lora.to_dict()),
                dataset_hash,
                data_manifest_hash,
            )
        )
        deltas = tuple(
            LoraDelta(
                example_id=example.example_id,
                target_text_sha256=example.target_text_sha256,
                sample_rate_hz=example.sample_rate_hz,
                replay=example.replay,
                weight=round(
                    int(sha256_text("|".join((prefix, example.example_id)))[:8], 16) / 0xFFFFFFFF,
                    6,
                ),
            )
            for example in examples
        )
        return FakeLoraAdapter(
            dataset_hash=dataset_hash,
            data_manifest_hash=data_manifest_hash,
            base_model=model,
            lora=lora.to_dict(),
            training={"backend": "fake", "params": dict(params), "seed": seed},
            deltas=tuple(sorted(deltas, key=lambda delta: delta.example_id)),
        )
