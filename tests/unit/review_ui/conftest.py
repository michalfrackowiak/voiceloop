"""Shared fixtures for the P4-03 review UI tests."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from services.review_adapter.metrics.cli import build_inputs
from services.review_adapter.metrics.events import InMemoryReportSink
from services.review_adapter.metrics.report import generate_metrics_report
from services.review_adapter.metrics.settings import ReviewMetricsSettings
from ui.constants import SCHEMA_PATH
from ui.form import FormDescriptor, build_form_descriptor
from ui.hashes import sha256_file


@pytest.fixture(scope="module")
def booking_descriptor() -> FormDescriptor:
    """Build the descriptor from the real hash-pinned booking schema."""
    schema: Any = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return build_form_descriptor(schema, sha256_file(SCHEMA_PATH))


@pytest.fixture(scope="module")
def p2_04_report() -> dict[str, Any]:
    """Generate the committed synthetic P2-04 report through its real seams."""
    settings = ReviewMetricsSettings()
    inputs = replace(build_inputs(settings), sink=InMemoryReportSink())
    report, _ = generate_metrics_report(inputs)
    return report
