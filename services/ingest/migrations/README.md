# Ingest migrations

`001_initial.sql` creates the real Postgres schema for the in-memory
repository's interface: `calls`, `turns`, `outcomes`, `model_versions` and the
`ingest_receipts` idempotency ledger. Apply in filename order with your usual
migration tool or `psql -f 001_initial.sql`.

Audio bytes are deliberately absent: `turns.audio_uri` points at the
S3-compatible object store and the remaining `audio_*` columns are metadata
only. Model ids/versions are deduplicated in `model_versions` and referenced by
`turns`. Late outcomes merge into one `outcomes` row per `call_id`.

The default service uses `VOICELOOP_INGEST_REPOSITORY_BACKEND=memory`, so no
database or driver is needed for tests. A Postgres adapter must implement
`services.ingest.repository.IngestRepository` against these tables.
