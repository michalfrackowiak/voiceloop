"""Typed settings for the parser trainer (P2-07).

Only the deterministic in-memory stand-in backend is available by default.
Real Unsloth/TRL/LLaMA-Factory, GLiNER2 and vLLM integrations are injected as
Protocols by production code (see ``adapter.py`` and ``trainer.py``); importing
them here would add GPU/network dependencies to the default test suite.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from voiceloop_common import BaseSettings, settings_config

ParserBackend = Literal["fake", "unsloth", "trl", "llama_factory"]


class ParserSettings(BaseSettings):
    """Parser-specific configuration read from ``VOICELOOP_PARSER_*``.

    Job-level lineage, base model, seed and artifact prefix stay in the core
    ``TrainerSettings`` (``VOICELOOP_TRAINERS_*``) so the runner remains the
    single owner of contract validation and MLflow logging. All defaults here
    are local-safe and describe the SYNTHETIC/FIXTURE TRAINING STAND-IN.
    """

    model_config = settings_config("VOICELOOP_PARSER_")

    backend: ParserBackend = Field(
        default="fake",
        description=(
            "Training backend label. Only 'fake' ships here; a non-fake value "
            "requires an injected SlotTrainingBackend and is rejected otherwise."
        ),
    )
    epochs: int = Field(default=1, ge=1, description="Number of SFT epochs recorded in the run.")
    learning_rate: float = Field(default=5e-4, gt=0.0, description="Recorded learning rate.")
    lora_rank: int = Field(default=8, ge=1, description="Recorded LoRA rank (conservative).")
    lora_alpha: float = Field(default=16.0, gt=0.0, description="Recorded LoRA scaling alpha.")
    exclude_call_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Golden/regression call IDs that must never enter parser training "
            "(context.md section 2 invariants 5 and 7). Supply from P1-06/P2-05."
        ),
    )

    def as_training_params(self) -> dict[str, Any]:
        """Return the JSON-serialisable hyperparameters recorded on the adapter.

        Returns:
            A stable parameter mapping used for deterministic pseudo-training
            and available for callers/orchestration to project into
            ``config.params``.

        """
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
        }
