"""Typed settings for the Parakeet/NeMo stand-in trainer (P3-02).

Only the deterministic fake backend ships in this package. A real NeMo build
must be injected through the Protocols in :mod:`~.trainer` and
:mod:`~.confidences`; importing NeMo/torch here would make the default test
suite depend on GPU and network access.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from voiceloop_common import BaseSettings, settings_config

PROVENANCE_LABEL = "SYNTHETIC/FIXTURE STAND-IN - NOT A REAL NEMO FINE-TUNE"
DEFAULT_CHECKPOINT_URI = "fake://nemo-checkpoints/parakeet-standin"
DEFAULT_CONFIDENCE_MODEL_ID = "fake-parakeet-standin"

NemoBackend = Literal["fake", "nemo"]


class NemoSettings(BaseSettings):
    """NeMo-specific configuration read from ``VOICELOOP_NEMO_*``.

    Job-level lineage, base model, seed and artifact prefix stay in the P2-06
    core ``TrainerSettings`` so the runner remains the single owner of contract
    validation and MLflow logging. Defaults here are local-safe and describe
    the synthetic fixture stand-in only.

    """

    model_config = settings_config("VOICELOOP_NEMO_")

    backend: NemoBackend = Field(
        default="fake",
        description=(
            "Training backend label. Only 'fake' ships here; a non-fake value "
            "requires an injected NeMoFinetuneBackend and is rejected otherwise."
        ),
    )
    manifest_path: str = Field(
        default="",
        description=(
            "Path to the read-only P2-05 NeMo ASR manifest JSONL export. "
            "Empty means the caller must inject a source or a manifest['nemo_path'] sidecar."
        ),
    )
    checkpoint_uri: str = Field(
        default=DEFAULT_CHECKPOINT_URI,
        min_length=1,
        description=(
            "Starting checkpoint reference for the fake fine-tune. Only fake:// "
            "URIs resolve without an injected CheckpointSource."
        ),
    )
    epochs: int = Field(default=1, ge=1, description="Recorded fine-tune epoch count.")
    learning_rate: float = Field(default=1e-4, gt=0.0, description="Recorded learning rate.")
    batch_size: int = Field(default=4, ge=1, description="Recorded batch size.")
    confidence_model_id: str = Field(
        default=DEFAULT_CONFIDENCE_MODEL_ID,
        min_length=1,
        description="STT model id written to the schema-shaped turn event.",
    )

    def as_training_params(self) -> dict[str, Any]:
        """Return JSON-serialisable NeMo hyperparameters recorded on the artifact.

        Returns:
            A stable mapping used by the deterministic fake fine-tune and
            available to orchestration for ``TrainerSettings.params``.

        """
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
        }
