"""Synthetic base-model benchmark harness for P3-05.

This package compares two settings-selectable candidate runners on our own
fixture labels, including the sealed v1 synthetic golden stand-in. It uses the
committed P1-07 evaluation primitives read-only and contains **no** Whisper,
Parakeet, Qwen3-ASR or other real model imports, GPU work or network access.
All numbers are SYNTHETIC/FIXTURE STAND-IN results, never public leaderboard
results, and ``e2e_booking_accuracy`` from audio remains the primary metric.
"""

from __future__ import annotations

from .compare import compare_candidates, load_benchmark_inputs
from .runners import (
    CandidateRunner,
    DegradedRunner,
    IdentityRunner,
    available_candidates,
    get_candidate,
    register_candidate,
)
from .settings import BenchmarkSettings

__all__ = [
    "BenchmarkSettings",
    "CandidateRunner",
    "DegradedRunner",
    "IdentityRunner",
    "available_candidates",
    "compare_candidates",
    "get_candidate",
    "load_benchmark_inputs",
    "register_candidate",
]
