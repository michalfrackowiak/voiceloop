"""Tests for JSON structured logging configuration."""

import json

import pytest
import structlog

from voiceloop_common import configure_logging, log_context


def test_configure_logging_emits_json_with_service_name(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("unit-test")

    structlog.get_logger().info("ready", turn_index=1)

    record = json.loads(capsys.readouterr().out.strip())
    assert record["service"] == "unit-test"
    assert record["event"] == "ready"
    assert record["level"] == "info"
    assert record["turn_index"] == 1
    assert "timestamp" in record


def test_configure_logging_falls_back_to_info_on_unknown_level(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("unit-test", "NOT_A_LEVEL")

    structlog.get_logger().warning("still logged")

    record = json.loads(capsys.readouterr().out.strip())
    assert record["event"] == "still logged"


def test_configure_logging_filters_below_level(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("unit-test", "ERROR")

    structlog.get_logger().info("hidden")

    assert capsys.readouterr().out == ""


def test_log_context_is_included_in_events(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("unit-test")
    log_context(call_id="call-1")

    structlog.get_logger().info("turn")

    record = json.loads(capsys.readouterr().out.strip())
    assert record["call_id"] == "call-1"
