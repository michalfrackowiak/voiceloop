-- VoiceLoop ingest schema, version 1 (P1-01).
-- Apply once to Postgres before running the ingest service with the real
-- repository adapter. The in-memory fake mirrors these tables.
-- Audio BYTES are never stored here: turns carry only an object-store URI and
-- format metadata. Audio lives in the S3-compatible object store.

CREATE TABLE model_versions (
    model_version_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('stt', 'parser')),
    model_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at TIMESTAMPTZ,
    UNIQUE (kind, model_id, model_version)
);

CREATE TABLE calls (
    call_id TEXT PRIMARY KEY,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- First non-null value seen for the call is preserved; never cleared.
    consent_recording BOOLEAN,
    consent_training BOOLEAN,
    turn_count INTEGER NOT NULL DEFAULT 0 CHECK (turn_count >= 0)
);

CREATE TABLE turns (
    event_id UUID PRIMARY KEY,
    call_id TEXT NOT NULL REFERENCES calls(call_id),
    turn_index INTEGER NOT NULL CHECK (turn_index >= 0),
    ts TIMESTAMPTZ,
    consent_recording BOOLEAN,
    consent_training BOOLEAN,
    audio_uri TEXT NOT NULL,
    audio_sample_rate_hz INTEGER CHECK (audio_sample_rate_hz IS NULL OR audio_sample_rate_hz > 0),
    audio_channels INTEGER CHECK (audio_channels IS NULL OR audio_channels > 0),
    audio_duration_ms INTEGER CHECK (audio_duration_ms IS NULL OR audio_duration_ms >= 0),
    audio_state TEXT NOT NULL DEFAULT 'referenced',
    stt_model_version_id TEXT REFERENCES model_versions(model_version_id),
    stt_text TEXT NOT NULL,
    stt_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    parser_model_version_id TEXT REFERENCES model_versions(model_version_id),
    parser_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    dialogue_context JSONB NOT NULL DEFAULT '{}'::jsonb,
    outcome_snapshot JSONB,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (call_id, turn_index)
);

CREATE INDEX turns_call_id_turn_index_idx ON turns (call_id, turn_index);

CREATE TABLE outcomes (
    call_id TEXT PRIMARY KEY REFERENCES calls(call_id),
    status TEXT NOT NULL CHECK (
        status IN ('confirmed', 'edited_by_staff', 'abandoned', 'transferred', 'unknown')
    ),
    final_record JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL,
    source_event_id UUID,
    merged_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    merge_count INTEGER NOT NULL DEFAULT 1 CHECK (merge_count > 0)
);

-- Idempotency ledger: the receipt returned on a duplicate event_id must match
-- the original, including when the write is still pending.
CREATE TABLE ingest_receipts (
    event_id UUID PRIMARY KEY,
    -- Immutable first response; duplicates return this exact receipt.
    original_receipt JSONB NOT NULL,
    -- Live status, updated by the write worker.
    receipt JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
