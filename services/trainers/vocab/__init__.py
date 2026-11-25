"""Offline vocabulary and rules tier (P2-08).

This package generates deterministic alias tables, post-ASR normalisation rules
and a JSON phrase-boosting config for an injectable STT runtime seam. It never
loads an STT model, audio decoder, GPU or network client. All fixture-derived
evidence is synthetic; no result here is evidence that a real runtime applied
phrase boosting.
"""

from __future__ import annotations

from .aliases import AliasEntry, AliasTable, derive_alias_table
from .boosting import (
    BoostConfig,
    FakeSttRuntime,
    Hotword,
    SttRuntime,
    Substitution,
    apply_boosting,
    generate_boosting_config,
    write_boosting_config,
)
from .normalise import normalise_slots, normalise_transcript
from .rules import RuleOutcome, VocabRuleSet, check_golden_regression
from .settings import VocabSettings

__all__ = [
    "AliasEntry",
    "AliasTable",
    "BoostConfig",
    "FakeSttRuntime",
    "Hotword",
    "RuleOutcome",
    "SttRuntime",
    "Substitution",
    "VocabRuleSet",
    "VocabSettings",
    "apply_boosting",
    "check_golden_regression",
    "derive_alias_table",
    "generate_boosting_config",
    "normalise_slots",
    "normalise_transcript",
    "write_boosting_config",
]
