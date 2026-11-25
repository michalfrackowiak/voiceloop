"""Typed, environment-backed settings for triage sampling, clustering and queueing."""

from __future__ import annotations

from pydantic import Field

from voiceloop_common import BaseSettings, settings_config


class TriageQueueSettings(BaseSettings):
    """Configuration for P2-02; every field reads ``VOICELOOP_TRIAGE_QUEUE_*``.

    Defaults are deterministic starting points, not learned parameters. The
    random-slice fraction is constrained to the 3–5% band required by
    ``context.md`` §2 invariant 6. ``random_slice_target`` is an optional
    integer override for small pools (notably the 36 synthetic fixtures, where
    no integer count lands inside that band); production should leave it unset.
    ``daily_cap`` follows the roadmap's "5 hours/week reviewer capacity"
    starting point (roughly 30 short items per working day) and should be tuned
    from measured reviewer throughput, not guessed further.
    """

    model_config = settings_config("VOICELOOP_TRIAGE_QUEUE_")

    # Context.md §2.6: 3–5% of turns bypass the failure filter.
    random_slice_fraction: float = Field(default=0.04, ge=0.03, le=0.05)
    random_slice_target: int | None = Field(default=None, ge=0)
    random_seed: int = 20261125

    # Failure filter and near-duplicate grouping.
    failure_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    cluster_similarity_threshold: float = Field(default=0.82, ge=0.0, le=1.0)

    # Daily queue budget, ordering and review UX.
    daily_cap: int = Field(default=30, ge=1)
    interleave_every: int = Field(default=5, ge=1)
    blind_share: float = Field(default=0.20, ge=0.0, le=1.0)
    low_confidence_ceiling: float = Field(default=0.60, ge=0.0, le=1.0)
