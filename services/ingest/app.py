"""FastAPI application for turn, audio, outcome and OpenTelemetry ingest."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast
from urllib.parse import quote
from uuid import NAMESPACE_URL, uuid5

import structlog
from fastapi import Body, FastAPI, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from services.ingest.errors import IngestHTTPError, error_body, schema_error_details
from services.ingest.models import (
    AudioUpload,
    AudioUploadRequest,
    OutcomeRequest,
    Receipt,
    StoredAudio,
)
from services.ingest.otel import span_to_event, unwrap_spans
from services.ingest.repository import IngestRepository, InMemoryRepository
from services.ingest.service import IngestService
from services.ingest.settings import IngestSettings
from services.ingest.storage import InMemoryObjectStore, ObjectStore
from services.ingest.validation import validate_turn, warmup
from voiceloop_common import ConfigError, configure_logging, utcnow
from voiceloop_common.errors import StorageError, TransientError
from voiceloop_common.errors import ValidationError as ContractValidationError

logger = structlog.get_logger(__name__)


def _stamp() -> str:
    """Return the current UTC time as an RFC 3339 string."""
    return utcnow().isoformat().replace("+00:00", "Z")


def _outcome_carrier(call_id: str, payload: OutcomeRequest) -> dict[str, Any]:
    """Wrap an outcome payload in a minimal event for authoritative validation."""
    stamp = payload.updated_at or _stamp()
    event_id = payload.event_id or str(
        uuid5(NAMESPACE_URL, f"urn:voiceloop:outcome:{call_id}:{stamp}")
    )
    outcome: dict[str, Any] = {"status": payload.status}
    if payload.final_record is not None:
        outcome["final_record"] = payload.final_record
    if payload.updated_at is not None:
        outcome["updated_at"] = payload.updated_at
    event: dict[str, Any] = {
        "event_id": event_id,
        "call_id": call_id,
        "turn_index": 0,
        "audio": {"uri": f"voiceloop://outcome/{quote(call_id, safe='')}"},
        "stt": {"text": ""},
        "outcome": outcome,
    }
    validate_turn(event)
    return event


def _backends(
    settings: IngestSettings,
    repository: IngestRepository | None,
    object_store: ObjectStore | None,
) -> tuple[IngestRepository, ObjectStore]:
    """Select typed backends; real drivers are not wired in this task."""
    if repository is None and settings.repository_backend != "memory":
        raise ConfigError(
            "Postgres ingest backend needs a driver; set VOICELOOP_INGEST_REPOSITORY_BACKEND=memory"
        )
    if object_store is None and settings.object_store_backend != "memory":
        raise ConfigError(
            "S3 ingest backend requires a driver; use VOICELOOP_INGEST_OBJECT_STORE_BACKEND=memory"
        )
    repo = repository or InMemoryRepository()
    store = object_store or InMemoryObjectStore(
        bucket=settings.audio_bucket,
        scheme=settings.audio_uri_scheme,
    )
    return repo, store


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start and stop the write worker for the application lifetime."""
    service = cast(IngestService, app.state.ingest)
    if app.state.start_worker:
        service.start()
    try:
        yield
    finally:
        service.stop()


def create_app(
    settings: IngestSettings | None = None,
    repository: IngestRepository | None = None,
    object_store: ObjectStore | None = None,
    *,
    autostart_worker: bool = True,
    service: IngestService | None = None,
) -> FastAPI:
    """Build an ingest app, optionally with in-memory fakes injected.

    Args:
        settings: Typed settings; read from the environment when omitted.
        repository: Optional repository fake for deterministic tests.
        object_store: Optional object-store fake for deterministic tests.
        autostart_worker: When False, writes stay queued until ``process_pending``.
        service: Optional pre-wired service, used by tests to share one queue.

    Returns:
        A configured FastAPI application.

    """
    config = settings or IngestSettings()
    configure_logging("ingest", config.log_level)
    warmup()
    repo, store = _backends(config, repository, object_store)
    app_service = service or IngestService(config, repo, store, autostart_worker=autostart_worker)

    app = FastAPI(
        title="VoiceLoop ingest",
        version="1.0.0",
        description="Validated turn-event ingest with fail-open, idempotent writes.",
        lifespan=_lifespan,
    )
    app.state.ingest = app_service
    app.state.start_worker = autostart_worker

    @app.exception_handler(IngestHTTPError)
    async def handle_ingest_error(_: Request, exc: IngestHTTPError) -> JSONResponse:
        """Return the shared structured error body for expected HTTP failures."""
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(exc.code, exc.message, exc.details),
        )

    @app.exception_handler(ContractValidationError)
    async def handle_contract_error(_: Request, exc: ContractValidationError) -> JSONResponse:
        """Return a structured 422 for shared-schema validation failures."""
        return JSONResponse(
            status_code=422,
            content=error_body(
                "validation_error",
                "Contract validation failed.",
                schema_error_details(exc),
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        """Return field-level validation errors without echoing submitted data."""
        details = {
            "errors": [
                {
                    "field": ".".join(str(part) for part in error.get("loc", [])),
                    "type": error.get("type", "invalid"),
                }
                for error in exc.errors()
            ]
        }
        return JSONResponse(
            status_code=422,
            content=error_body("validation_error", "Request payload is invalid.", details),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Map framework 404/405 responses to the shared structured error shape."""
        codes = {404: "not_found", 405: "method_not_allowed"}
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(codes.get(exc.status_code, "http_error"), str(exc.detail)),
        )

    @app.exception_handler(StorageError)
    @app.exception_handler(TransientError)
    async def handle_storage_error(_: Request, exc: Exception) -> JSONResponse:
        """Surface synchronous storage failures as structured 503 responses."""
        logger.error("ingest_storage_error", error=type(exc).__name__)
        return JSONResponse(
            status_code=503,
            content=error_body("storage_error", "Storage is temporarily unavailable."),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        """Log and contain unexpected errors without leaking payload details."""
        logger.error("ingest_unexpected_error", error=type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content=error_body("internal_error", "Ingest could not process the request."),
        )

    @app.post("/v1/turns", response_model=Receipt, status_code=202, tags=["turns"])
    async def post_turn(
        request: Request,
        response: Response,
        body: Any = Body(...),  # noqa: B008 - FastAPI request body marker
    ) -> Receipt:
        """Validate a turn event and enqueue its write without blocking on storage."""
        started = time.perf_counter()
        if not isinstance(body, dict):
            raise IngestHTTPError(
                422,
                "validation_error",
                "Turn event must be a JSON object.",
                {"field": "<root>", "reason": "expected_object"},
            )
        validate_turn(body)
        receipt, duplicate = app_service.enqueue_turn(body)
        if duplicate:
            response.headers["X-VoiceLoop-Idempotent"] = "hit"
        logger.info(
            "turn_ingest_accepted",
            event_id=receipt.event_id,
            call_id=receipt.call_id,
            turn_index=body.get("turn_index"),
            duplicate=duplicate,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return receipt

    @app.get("/v1/turns/{event_id}", tags=["turns"])
    async def get_turn(event_id: str) -> dict[str, Any]:
        """Return a stored turn and its current receipt."""
        turn = app_service.repository.get_turn(event_id)
        if turn is None:
            raise IngestHTTPError(
                404, "not_found", "No turn exists for that event id.", {"event_id": event_id}
            )
        return {"turn": turn, "receipt": app_service.repository.get_receipt(event_id)}

    @app.get("/v1/calls/{call_id}", tags=["calls"])
    async def get_call(call_id: str) -> dict[str, Any]:
        """Return a call with its stored turns and merged outcome."""
        call = app_service.repository.get_call(call_id)
        if call is None:
            raise IngestHTTPError(
                404, "not_found", "No call exists for that id.", {"call_id": call_id}
            )
        return {
            "call": call,
            "turns": app_service.repository.list_turns(call_id),
            "outcome": app_service.repository.get_outcome(call_id),
        }

    @app.patch(
        "/v1/calls/{call_id}/outcome", response_model=Receipt, status_code=202, tags=["calls"]
    )
    async def patch_outcome(
        request: Request,
        response: Response,
        call_id: str,
        payload: OutcomeRequest,
    ) -> Receipt:
        """Validate and enqueue a late outcome, merged by call id."""
        event = _outcome_carrier(call_id, payload)
        receipt, duplicate = app_service.enqueue_outcome(
            call_id, str(event["event_id"]), cast(dict[str, Any], event["outcome"])
        )
        if duplicate:
            response.headers["X-VoiceLoop-Idempotent"] = "hit"
        logger.info(
            "outcome_ingest_accepted",
            event_id=receipt.event_id,
            call_id=call_id,
            status=payload.status,
            duplicate=duplicate,
        )
        return receipt

    @app.post("/v1/audio/uploads", response_model=AudioUpload, status_code=201, tags=["audio"])
    async def create_audio_upload(payload: AudioUploadRequest, request: Request) -> AudioUpload:
        """Issue a presigned PUT URL for one turn's audio."""
        upload = app_service.issue_upload(payload)
        logger.info(
            "audio_upload_issued",
            call_id=payload.call_id,
            turn_index=payload.turn_index,
            key=upload.key,
        )
        return upload

    @app.put(
        "/v1/audio/objects/{key:path}", response_model=StoredAudio, status_code=201, tags=["audio"]
    )
    async def put_audio(
        key: str, request: Request, upload_id: str = Query(min_length=1)
    ) -> StoredAudio:
        """Accept bytes for a presigned local upload (fake object-store path)."""
        data = await request.body()
        stored = app_service.store_audio(upload_id, key, data)
        logger.info("audio_upload_stored", key=key, size_bytes=stored.size_bytes)
        return stored

    @app.post("/v1/otel/spans", status_code=202, tags=["otel"])
    async def post_spans(
        request: Request,
        body: Any = Body(...),  # noqa: B008 - FastAPI request body marker
    ) -> Any:
        """Map span attributes to turn events and enqueue them through the same path."""
        spans = unwrap_spans(body)
        if not spans:
            raise IngestHTTPError(
                422,
                "validation_error",
                "No spans were found in the payload.",
                {"field": "spans", "reason": "empty"},
            )
        receipts: list[Receipt] = []
        for span in spans:
            event = span_to_event(span)
            validate_turn(event)
            receipt, _ = app_service.enqueue_turn(event)
            receipts.append(receipt)
        logger.info("otel_spans_accepted", span_count=len(receipts))
        if len(receipts) == 1:
            return receipts[0]
        return {"count": len(receipts), "receipts": receipts}

    @app.get("/healthz", tags=["health"])
    async def healthz() -> dict[str, Any]:
        """Report liveness plus queue depth and write counters."""
        return {"status": "ok", **app_service.stats()}

    return app


app = create_app()
