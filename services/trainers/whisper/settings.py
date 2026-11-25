"""Typed settings for the P3-01 Whisper trainer.

Every field reads the ``VOICELOOP_WHISPER_`` prefix. The defaults describe the
deterministic SYNTHETIC/FIXTURE STAND-IN backend: no base model name is
hard-coded, no heavy dependency is imported and no GPU job is started. The
conservative LoRA default (alpha=1.0, r=8, scaling=0.125) is far below the 2.0
scaling reported as a hallucination risk in ``context.md`` §10. Evidence and
limitations are documented in :mod:`services.trainers.whisper.lora` and the
package README.

Job-level lineage (git SHA, pinned image digest, artifact URI prefix, model and
seed) stays in the P2-06 ``TrainerSettings`` (``VOICELOOP_TRAINERS_*``); this
class owns only Whisper-specific hyperparameters and decoding defaults.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from voiceloop_common import BaseSettings, settings_config

WhisperBackend = Literal["fake", "transformers_peft"]

DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 1.0
MAX_LORA_SCALING = 2.0
DEFAULT_TARGET_SAMPLE_RATE_HZ = 16000


class WhisperSettings(BaseSettings):
    """Environment-backed Whisper hyperparameters for one stand-in training run.

    ``backend="fake"`` is the only shipped implementation. A non-fake label is
    deliberately inert: the trainer fails closed until a real
    :class:`~.lora.LoraTrainingBackend` is injected. Decoding defaults mirror
    the CTranslate2 risk notes in ``context.md`` §10 (no initial prompt, no
    hotwords, no condition-on-previous-text) and carry no guest text.
    """

    model_config = settings_config("VOICELOOP_WHISPER_")

    backend: WhisperBackend = Field(
        default="fake",
        description=(
            "Training backend label. Only 'fake' ships here; a non-fake value "
            "requires an injected LoraTrainingBackend and is rejected otherwise."
        ),
    )
    lora_r: int = Field(default=DEFAULT_LORA_R, ge=1, description="LoRA rank (conservative).")
    lora_alpha: float = Field(
        default=DEFAULT_LORA_ALPHA,
        gt=0.0,
        description="LoRA alpha; alpha/r is the effective scaling (default 0.125).",
    )
    max_lora_scaling: float = Field(
        default=MAX_LORA_SCALING,
        gt=0.0,
        description="Hard upper bound on alpha/r; runs above it fail closed.",
    )
    epochs: int = Field(default=1, ge=1, description="Recorded fine-tune epochs.")
    learning_rate: float = Field(default=1e-4, gt=0.0, description="Recorded learning rate.")
    replay_fraction: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Fraction of primary examples added from the verified replay pool.",
    )
    replay_seed: int = Field(default=0, description="Seed for deterministic replay selection.")
    language: str = Field(default="en", min_length=1, description="Decoding language tag.")
    task: Literal["transcribe", "translate"] = Field(
        default="transcribe", description="Whisper task recorded in decoding settings."
    )
    beam_size: int = Field(default=5, ge=1, description="Recorded CTranslate2 beam size.")
    temperature: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Recorded decoding temperature."
    )
    condition_on_previous_text: bool = Field(
        default=False,
        description=(
            "Disabled by default: converted CTranslate2 models can emit empty or "
            "truncated output with it enabled (context.md section 10)."
        ),
    )
    preserve_native_sample_rate: bool = Field(
        default=True,
        description=(
            "Never resample 8 kHz telephony audio in the fake preprocessor; "
            "source rate is tagged instead (context.md section 16)."
        ),
    )
    target_sample_rate_hz: int = Field(
        default=DEFAULT_TARGET_SAMPLE_RATE_HZ,
        ge=1,
        description=(
            "Whisper feature-extractor rate recorded for real adapters; the fake "
            "path preserves the source rate and does not resample."
        ),
    )
    exclude_call_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Golden/regression call IDs that must never enter Whisper training "
            "(context.md section 2 invariants 5 and 7). Supply from P1-06/P2-05."
        ),
    )

    @property
    def lora_scaling(self) -> float:
        """Return the effective LoRA scaling ``alpha / r``."""
        return self.lora_alpha / self.lora_r

    @model_validator(mode="after")
    def _enforce_conservative_scaling(self) -> WhisperSettings:
        """Reject LoRA settings whose effective scaling exceeds the hard bound."""
        if self.lora_scaling > self.max_lora_scaling:
            raise ValueError(
                "LoRA scaling alpha/r must not exceed "
                f"VOICELOOP_WHISPER_MAX_LORA_SCALING={self.max_lora_scaling}"
            )
        return self

    def as_training_params(self) -> dict[str, Any]:
        """Return JSON-serialisable hyperparameters recorded on the fake adapter.

        Returns:
            A stable mapping used for deterministic pseudo-training and for the
            ``params`` projection of the trainer contract config.

        """
        return {
            "backend": self.backend,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "replay_fraction": self.replay_fraction,
            "replay_seed": self.replay_seed,
            "language": self.language,
            "task": self.task,
            "preserve_native_sample_rate": self.preserve_native_sample_rate,
            "target_sample_rate_hz": self.target_sample_rate_hz,
        }

    def as_decoding_values(self) -> dict[str, Any]:
        """Return PII-free decoding values for :class:`~.convert.DecodingSettings`.

        Returns:
            A mapping whose hotword list and initial prompt are deliberately
            empty. Real runtime-specific boosting belongs in the injected
            backend, never in a synthetic artifact (P2-08 boost config is a
            separate seam).

        """
        return {
            "language": self.language,
            "task": self.task,
            "beam_size": self.beam_size,
            "temperature": self.temperature,
            "condition_on_previous_text": self.condition_on_previous_text,
            "initial_prompt": "",
            "hotwords": (),
            "word_timestamps": False,
            "vad_filter": False,
        }
