"""Typed settings for P2-12 lineage, deletion and access control.

Only environment variables prefixed with ``VOICELOOP_LINEAGE_`` are read. No
secrets, credentials or real guest identifiers live in this module: the
pseudonym domain is a public, stable domain separator, not a signing key.
"""

from __future__ import annotations

from pydantic import Field

from voiceloop_common import BaseSettings, settings_config


class LineageSettings(BaseSettings):
    """Environment-backed configuration for the governance lineage package."""

    model_config = settings_config("VOICELOOP_LINEAGE_")

    pseudonym_domain: str = Field(default="voiceloop-lineage-v1", min_length=1)
    audit_log_path: str = Field(
        default="reports/lineage/lineage_audit.jsonl",
        min_length=1,
    )

    @property
    def task_id(self) -> str:
        """Return the roadmap task ID used in audit metadata."""
        return "P2-12"
