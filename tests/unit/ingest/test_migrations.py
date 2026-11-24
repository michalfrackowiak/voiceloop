"""Checks that the SQL migration matches the repository's metadata-only schema."""

from __future__ import annotations

from pathlib import Path

MIGRATION = (
    Path(__file__).resolve().parents[3] / "services" / "ingest" / "migrations" / "001_initial.sql"
)


def test_initial_migration_defines_required_tables() -> None:
    sql = MIGRATION.read_text(encoding="utf-8").lower()

    for table in ("model_versions", "calls", "turns", "outcomes", "ingest_receipts"):
        assert f"create table {table}" in sql


def test_initial_migration_keeps_audio_out_of_postgres() -> None:
    sql = MIGRATION.read_text(encoding="utf-8").lower()

    assert "audio_uri text not null" in sql
    assert "audio_sample_rate_hz integer" in sql
    assert "bytea" not in sql
    assert "audio_bytes" not in sql
    assert "jsonb" in sql


def test_initial_migration_carries_invariants() -> None:
    sql = MIGRATION.read_text(encoding="utf-8").lower()

    assert "consent_recording boolean" in sql
    assert "consent_training boolean" in sql
    assert "status in ('confirmed'" in sql
    assert "references calls(call_id)" in sql
    assert "unique (call_id, turn_index)" in sql
    assert "create index turns_call_id_turn_index_idx" in sql
