"""Read-only hash verification for the P4-03 synthetic UI harness."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import (
    EXPECTED_BOOKING_SCHEMA_SHA256,
    EXPECTED_FIXTURE_SHA256,
    EXPECTED_GOLDEN_CONTENT_HASH,
    FIXTURES_PATH,
    GOLDEN_MANIFEST_PATH,
    SCHEMA_PATH,
)
from .errors import UiValidationError


@dataclass(frozen=True)
class HashVerification:
    """Pinned hashes observed during one verification pass."""

    fixture_sha256: str
    booking_schema_sha256: str
    golden_content_hash: str
    golden_synthetic: bool
    golden_workflow_stand_in: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the verification result."""
        return {
            "fixture_sha256": self.fixture_sha256,
            "fixture_hash_ok": self.fixture_sha256 == EXPECTED_FIXTURE_SHA256,
            "booking_schema_sha256": self.booking_schema_sha256,
            "booking_schema_hash_ok": self.booking_schema_sha256 == EXPECTED_BOOKING_SCHEMA_SHA256,
            "golden_content_hash": self.golden_content_hash,
            "golden_hash_ok": self.golden_content_hash == EXPECTED_GOLDEN_CONTENT_HASH,
            "golden_synthetic": self.golden_synthetic,
            "golden_workflow_stand_in": self.golden_workflow_stand_in,
        }


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in binary mode."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_read_only_inputs() -> HashVerification:
    """Verify all pinned read-only inputs and fail closed on any mismatch.

    Returns:
        The observed hashes and golden provenance flags.

    Raises:
        UiValidationError: If any pinned hash differs or the golden stand-in
            no longer declares itself synthetic.

    """
    manifest = json.loads(GOLDEN_MANIFEST_PATH.read_text(encoding="utf-8"))
    result = HashVerification(
        fixture_sha256=sha256_file(FIXTURES_PATH),
        booking_schema_sha256=sha256_file(SCHEMA_PATH),
        golden_content_hash=str(manifest["content_hash"]),
        golden_synthetic=bool(manifest["synthetic"]),
        golden_workflow_stand_in=bool(manifest["workflow_stand_in"]),
    )
    checks = result.to_dict()
    for key in ("fixture_hash_ok", "booking_schema_hash_ok", "golden_hash_ok"):
        if not checks[key]:
            raise UiValidationError(f"read-only input hash mismatch: {key}")
    if not result.golden_synthetic or not result.golden_workflow_stand_in:
        raise UiValidationError("golden stand-in provenance flags changed")
    return result
