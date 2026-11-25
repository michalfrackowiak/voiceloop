"""Typed, environment-backed settings for triage failure scoring (P2-01)."""

from __future__ import annotations

from pydantic import Field

from voiceloop_common import BaseSettings, settings_config


class TriageScoringSettings(BaseSettings):
    """Configuration for P2-01; every field reads ``VOICELOOP_TRIAGE_*``.

    Thresholds are deterministic starting points, not learned parameters. The
    final-record mismatch weight is deliberately the largest: it compares the
    parser hypothesis with the booking system's recorded outcome (context.md
    section 6). No secrets or base-model names live here.
    """

    model_config = settings_config("VOICELOOP_TRIAGE_")

    # Detection thresholds.
    min_word_conf: float = Field(default=0.60, ge=0.0, le=1.0)
    mean_word_conf: float = Field(default=0.75, ge=0.0, le=1.0)
    no_speech_prob_ceiling: float = Field(default=0.50, ge=0.0, le=1.0)
    avg_logprob_floor: float = -0.60
    compression_ratio_ceiling: float = 2.5
    parse_logprob_floor: float = -1.0

    # Weights. final_record_mismatch > all other weights, as required.
    weight_low_word_confidence: float = Field(default=0.35, ge=0.0)
    weight_decoder_stats: float = Field(default=0.30, ge=0.0)
    weight_schema_invalid: float = Field(default=0.25, ge=0.0)
    weight_low_parse_probability: float = Field(default=0.20, ge=0.0)
    weight_repair_behaviour: float = Field(default=0.15, ge=0.0)
    weight_hypothesis_parser_mismatch: float = Field(default=0.50, ge=0.0)
    weight_final_record_mismatch: float = Field(default=1.00, ge=0.0)
    weight_teacher_disagreement: float = Field(default=0.30, ge=0.0)

    # Optional teacher: disabled by default; a fake adapter is the safe fallback.
    teacher_enabled: bool = False
