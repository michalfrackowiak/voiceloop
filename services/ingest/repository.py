"""Persistence boundary for ingest metadata, with an in-memory Postgres fake.

The interface stores only call/turn/outcome/model-version *metadata*. Audio
bytes belong to the object store; at most an audio URI and its format metadata
are kept here. The SQL migration in ``migrations/`` implements the same tables
for a real Postgres deployment.
"""

from __future__ import annotations

import hashlib
import threading
from copy import deepcopy
from typing import Any, Protocol

from voiceloop_common import utcnow


def model_version_id(kind: str, model_id: str, model_version: str) -> str:
    """Return a stable identifier for a ``(kind, model_id, version)`` triple."""
    raw = f"{kind}\x1f{model_id}\x1f{model_version}".encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def _stamp() -> str:
    """Return the current UTC time as an RFC 3339 string."""
    return utcnow().isoformat().replace("+00:00", "Z")


class IngestRepository(Protocol):
    """Storage interface shared by the in-memory fake and a future Postgres adapter."""

    def claim_event(self, event_id: str, receipt: dict[str, Any]) -> dict[str, Any] | None:
        """Atomically claim an event id, returning the original receipt if present."""
        ...

    def save_receipt(self, event_id: str, receipt: dict[str, Any]) -> None:
        """Store or replace the receipt for an event id."""
        ...

    def get_receipt(self, event_id: str) -> dict[str, Any] | None:
        """Return the stored receipt for an event id, if any."""
        ...

    def save_call(
        self, call_id: str, consent: dict[str, bool] | None, seen_at: str | None = None
    ) -> dict[str, Any]:
        """Create or update a call row, preserving first-seen consent flags."""
        ...

    def get_call(self, call_id: str) -> dict[str, Any] | None:
        """Return a call row by id."""
        ...

    def save_turn(self, event: dict[str, Any], *, audio_state: str) -> dict[str, Any]:
        """Store turn metadata from a validated event."""
        ...

    def get_turn(self, event_id: str) -> dict[str, Any] | None:
        """Return a stored turn by event id."""
        ...

    def list_turns(self, call_id: str | None = None) -> list[dict[str, Any]]:
        """Return stored turns, optionally filtered by call id."""
        ...

    def merge_outcome(
        self, call_id: str, outcome: dict[str, Any], *, source_event_id: str
    ) -> dict[str, Any]:
        """Merge an outcome into the single row for a call."""
        ...

    def get_outcome(self, call_id: str) -> dict[str, Any] | None:
        """Return the merged outcome for a call."""
        ...

    def save_model_version(
        self, kind: str, model_id: str, model_version: str, seen_at: str | None = None
    ) -> None:
        """Record a model id/version pair seen in a turn event."""
        ...

    def list_model_versions(self) -> list[dict[str, Any]]:
        """Return all recorded model id/version pairs."""
        ...


class InMemoryRepository:
    """Thread-safe in-memory implementation of :class:`IngestRepository`."""

    def __init__(self) -> None:
        """Create empty stores for calls, turns, outcomes, models and receipts."""
        self._lock = threading.RLock()
        self._original_receipts: dict[str, dict[str, Any]] = {}
        self._receipts: dict[str, dict[str, Any]] = {}
        self._calls: dict[str, dict[str, Any]] = {}
        self._turns: dict[str, dict[str, Any]] = {}
        self._outcomes: dict[str, dict[str, Any]] = {}
        self._models: dict[str, dict[str, Any]] = {}

    def claim_event(self, event_id: str, receipt: dict[str, Any]) -> dict[str, Any] | None:
        """Atomically claim an event id, returning the original receipt if present.

        The claim snapshot is immutable so a duplicate returns the exact result
        of the first submission, even after the worker updates the live receipt.
        """
        with self._lock:
            original = self._original_receipts.get(event_id)
            if original is not None:
                return deepcopy(original)
            snapshot = deepcopy(receipt)
            self._original_receipts[event_id] = snapshot
            self._receipts[event_id] = deepcopy(snapshot)
            return None

    def save_receipt(self, event_id: str, receipt: dict[str, Any]) -> None:
        """Store or replace the receipt for an event id."""
        with self._lock:
            self._receipts[event_id] = deepcopy(receipt)

    def get_receipt(self, event_id: str) -> dict[str, Any] | None:
        """Return the stored receipt for an event id, if any."""
        with self._lock:
            receipt = self._receipts.get(event_id)
            return deepcopy(receipt) if receipt is not None else None

    def save_call(
        self, call_id: str, consent: dict[str, bool] | None, seen_at: str | None = None
    ) -> dict[str, Any]:
        """Create or update a call row, preserving first-seen consent flags."""
        stamp = seen_at or _stamp()
        with self._lock:
            call = self._calls.get(call_id)
            if call is None:
                call = {
                    "call_id": call_id,
                    "first_seen_at": stamp,
                    "last_seen_at": stamp,
                    "consent_recording": None,
                    "consent_training": None,
                    "turn_count": 0,
                }
                self._calls[call_id] = call
            else:
                call["last_seen_at"] = stamp
            if consent:
                for flag in ("recording", "training"):
                    key = f"consent_{flag}"
                    if call[key] is None and flag in consent:
                        call[key] = bool(consent[flag])
            return deepcopy(call)

    def get_call(self, call_id: str) -> dict[str, Any] | None:
        """Return a call row by id."""
        with self._lock:
            call = self._calls.get(call_id)
            return deepcopy(call) if call is not None else None

    def save_turn(self, event: dict[str, Any], *, audio_state: str) -> dict[str, Any]:
        """Store turn metadata from a validated event (never audio bytes)."""
        record: dict[str, Any] = {
            "event_id": event["event_id"],
            "call_id": event["call_id"],
            "turn_index": event["turn_index"],
            "ts": event.get("ts"),
            "consent": deepcopy(event.get("consent")),
            "audio": deepcopy(event["audio"]),
            "stt": deepcopy(event["stt"]),
            "parser": deepcopy(event.get("parser")),
            "dialogue_context": deepcopy(event.get("dialogue_context")),
            "outcome": deepcopy(event.get("outcome")),
            "audio_state": audio_state,
            "ingested_at": _stamp(),
        }
        with self._lock:
            self._turns[record["event_id"]] = record
            call = self._calls.get(record["call_id"])
            if call is None:
                self.save_call(record["call_id"], record["consent"])
                call = self._calls[record["call_id"]]
            call["turn_count"] += 1
            call["last_seen_at"] = record["ingested_at"]
            return deepcopy(record)

    def get_turn(self, event_id: str) -> dict[str, Any] | None:
        """Return a stored turn by event id."""
        with self._lock:
            turn = self._turns.get(event_id)
            return deepcopy(turn) if turn is not None else None

    def list_turns(self, call_id: str | None = None) -> list[dict[str, Any]]:
        """Return stored turns, optionally filtered by call id, ordered by call/turn."""
        with self._lock:
            turns = [
                deepcopy(t)
                for t in self._turns.values()
                if call_id is None or t["call_id"] == call_id
            ]
        return sorted(turns, key=lambda t: (t["call_id"], t["turn_index"]))

    def merge_outcome(
        self, call_id: str, outcome: dict[str, Any], *, source_event_id: str
    ) -> dict[str, Any]:
        """Merge an outcome into the single row for a call.

        Later ``updated_at`` values win for status; ``final_record`` objects are
        merged key by key, and an ``unknown`` status never overwrites a known
        one.
        """
        stamp = _stamp()
        with self._lock:
            current = self._outcomes.get(call_id)
            incoming_at = outcome.get("updated_at") or stamp
            if current is None:
                record = {
                    "call_id": call_id,
                    "status": outcome["status"],
                    "final_record": deepcopy(outcome.get("final_record") or {}),
                    "updated_at": incoming_at,
                    "source_event_id": source_event_id,
                    "merged_at": stamp,
                    "merge_count": 1,
                }
            else:
                newer = incoming_at >= current["updated_at"]
                status = outcome["status"]
                if not newer or (status == "unknown" and current["status"] != "unknown"):
                    status = current["status"]
                record = {
                    "call_id": call_id,
                    "status": status,
                    "final_record": {
                        **current.get("final_record", {}),
                        **(outcome.get("final_record") or {}),
                    },
                    "updated_at": max(incoming_at, current["updated_at"]),
                    "source_event_id": source_event_id if newer else current["source_event_id"],
                    "merged_at": stamp,
                    "merge_count": current["merge_count"] + 1,
                }
            self._outcomes[call_id] = record
            if call_id not in self._calls:
                self.save_call(call_id, None, seen_at=stamp)
            return deepcopy(record)

    def get_outcome(self, call_id: str) -> dict[str, Any] | None:
        """Return the merged outcome for a call."""
        with self._lock:
            outcome = self._outcomes.get(call_id)
            return deepcopy(outcome) if outcome is not None else None

    def save_model_version(
        self, kind: str, model_id: str, model_version: str, seen_at: str | None = None
    ) -> None:
        """Record a model id/version pair seen in a turn event."""
        identifier = model_version_id(kind, model_id, model_version)
        stamp = seen_at or _stamp()
        with self._lock:
            existing = self._models.get(identifier)
            if existing is None:
                self._models[identifier] = {
                    "model_version_id": identifier,
                    "kind": kind,
                    "model_id": model_id,
                    "model_version": model_version,
                    "first_seen_at": stamp,
                }
            else:
                existing["last_seen_at"] = stamp

    def list_model_versions(self) -> list[dict[str, Any]]:
        """Return all recorded model id/version pairs."""
        with self._lock:
            return [deepcopy(item) for item in self._models.values()]
