"""Typed, environment-backed settings for the trainer core.

Every field is read from ``VOICELOOP_TRAINERS_*``. Defaults are fake/local-safe;
production orchestration must inject a real pinned image digest and git SHA.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from voiceloop_common import BaseSettings, settings_config

DEFAULT_IMAGE_DIGEST = "ghcr.io/voiceloop/trainer@sha256:" + "0" * 64
DEFAULT_GIT_SHA = "0" * 40
DEFAULT_ARTIFACT_URI_PREFIX = "fake://trainer-artifacts"


class TrainerSettings(BaseSettings):
    """Configuration for one trainer job; every field reads ``VOICELOOP_TRAINERS_*``.

    ``image_digest`` is a full pinned image reference (``repo@sha256:<64 hex>``)
    so lineage always identifies the exact training container. ``git_sha`` is the
    training-code commit; the all-zero default is a fake for local tests only.
    ``model`` is intentionally empty by default: base-model selection belongs
    to settings, never to a hard-coded name in plugin code.
    """

    model_config = settings_config("VOICELOOP_TRAINERS_")

    image_digest: str = Field(
        default=DEFAULT_IMAGE_DIGEST,
        pattern=r"^[^\s@]+@sha256:[0-9a-f]{64}$",
        description="Pinned training image reference, e.g. repo@sha256:<64 hex digits>.",
    )
    git_sha: str = Field(
        default=DEFAULT_GIT_SHA,
        pattern=r"^[0-9a-f]{7,40}$",
        description="Git commit of the training code; inject the real SHA in production.",
    )
    artifact_uri_prefix: str = Field(
        default=DEFAULT_ARTIFACT_URI_PREFIX,
        min_length=1,
        description="URI prefix for the deployable artifact (merged adapter or converted model).",
    )
    model: str = Field(
        default="",
        description="Base model resolved from settings; empty means the plugin needs no model.",
    )
    seed: int = Field(default=0, description="Random seed recorded in the contract config.")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Trainer-specific hyperparameters, parsed from a JSON env value.",
    )

    def as_contract_config(self, trainer_name: str) -> dict[str, Any]:
        """Build the trainer contract's ``config`` object for a named plugin.

        Args:
            trainer_name: Registered plugin name selected for this run.

        Returns:
            A JSON-serialisable config dict satisfying the 1.0.0 contract's
            TrainRequest.config branch. Optional keys are omitted when unset.

        """
        config: dict[str, Any] = {"trainer": trainer_name}
        if self.model:
            config["model"] = self.model
        config["seed"] = self.seed
        if self.params:
            config["params"] = dict(self.params)
        return config
