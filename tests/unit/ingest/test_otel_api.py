"""Tests for the OpenTelemetry-compatible ingest endpoint and attribute mapping."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from services.ingest.models import Receipt
from services.ingest.otel import ATTRIBUTE_MAPPING
from services.ingest.repository import InMemoryRepository
from services.ingest.service import IngestService


def _span(attributes: dict[str, Any], span_id: str = "b7ad6b7169203331") -> dict[str, Any]:
    """Build a minimal span-like payload with a deterministic identity."""
    return {
        "name": "voiceloop.turn",
        "traceId": "0af7651916cd43dd8448eb211c80319c",
        "spanId": span_id,
        "attributes": attributes,
    }


def test_span_attributes_map_to_a_validated_turn(
    client: TestClient, service: IngestService, repository: InMemoryRepository
) -> None:
    span = _span(
        {
            "voiceloop.call_id": "call-otel",
            "voiceloop.turn_index": "2",
            "voiceloop.ts": "2026-11-25T09:00:00Z",
            "voiceloop.consent.recording": {"boolValue": True},
            "voiceloop.consent.training": {"boolValue": False},
            "voiceloop.audio.uri": "s3://bucket/otel.wav",
            "voiceloop.audio.sample_rate_hz": {"intValue": "8000"},
            "hypothesis": "two guests please",
            "voiceloop.stt.model_id": "stt-stub",
            "voiceloop.stt.model_version": "1.0.0",
            "voiceloop.stt.words": '[{"w":"two","start_ms":0,"end_ms":100,"conf":0.9}]',
            "voiceloop.parser.model_id": "parser-stub",
            "voiceloop.parser.output": {"guests": 2},
            "voiceloop.dialogue_context.expected_slot": "guests",
        }
    )

    response = client.post("/v1/otel/spans", json=span)

    assert response.status_code == 202, response.text
    receipt = Receipt.model_validate(response.json())
    assert receipt.event_id
    assert service.process_pending() == 1

    turn = repository.get_turn(receipt.event_id)
    assert turn is not None
    assert turn["call_id"] == "call-otel"
    assert turn["turn_index"] == 2
    assert turn["ts"] == "2026-11-25T09:00:00Z"
    assert turn["consent"] == {"recording": True, "training": False}
    assert turn["audio"] == {"uri": "s3://bucket/otel.wav", "sample_rate_hz": 8000}
    assert turn["stt"]["text"] == "two guests please"
    assert turn["stt"]["words"][0]["w"] == "two"
    assert turn["parser"] == {"model_id": "parser-stub", "output": {"guests": 2}}
    assert turn["dialogue_context"] == {"expected_slot": "guests"}


def test_otlp_envelope_and_multiple_spans(client: TestClient, service: IngestService) -> None:
    attributes = {
        "call_id": "call-otlp",
        "turn_index": 0,
        "audio_uri": "s3://bucket/otlp.wav",
        "stt.text": "",
    }
    envelope = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            _span(attributes, span_id="1111111111111111"),
                            _span({**attributes, "turn_index": 1}, span_id="2222222222222222"),
                        ]
                    }
                ]
            }
        ]
    }

    response = client.post("/v1/otel/spans", json=envelope)

    assert response.status_code == 202, response.text
    assert response.json()["count"] == 2
    assert len(response.json()["receipts"]) == 2
    assert service.process_pending() == 2


def test_otel_attribute_list_and_outcome_are_supported(
    client: TestClient, service: IngestService, repository: InMemoryRepository
) -> None:
    attributes = [
        {"key": "voiceloop.call_id", "value": {"stringValue": "call-list"}},
        {"key": "voiceloop.turn_index", "value": {"intValue": "0"}},
        {"key": "voiceloop.audio.uri", "value": {"stringValue": "s3://bucket/list.wav"}},
        {"key": "voiceloop.outcome.status", "value": {"stringValue": "confirmed"}},
        {"key": "voiceloop.outcome.final_record", "value": {"stringValue": '{"guests": 2}'}},
    ]

    span = {"name": "voiceloop.turn", "spanId": "abc123", "attributes": attributes}
    response = client.post("/v1/otel/spans", json=span)

    assert response.status_code == 202, response.text
    assert service.process_pending() == 1
    outcome = repository.get_outcome("call-list")
    assert outcome is not None
    assert outcome["status"] == "confirmed"
    assert outcome["final_record"] == {"guests": 2}


def test_missing_required_attribute_returns_structured_error(client: TestClient) -> None:
    span = _span({"voiceloop.turn_index": 0, "voiceloop.audio.uri": "s3://bucket/x.wav"})

    response = client.post("/v1/otel/spans", json=span)

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_error"
    assert body["details"]["field"] == "voiceloop.call_id"


def test_attribute_mapping_table_covers_core_fields() -> None:
    fields = {target for _, target in ATTRIBUTE_MAPPING}

    assert {
        "call_id",
        "turn_index",
        "audio.uri",
        "stt.text",
        "stt.model_id",
        "consent.recording",
    } <= fields
