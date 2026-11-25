"""Independent tamper matrix for the committed v1 sealed artifact.

Run from the repository root:

    .\\.venv\\Scripts\\python.exe tests/unit/golden/tamper_matrix.py

Every case mutates exactly one field in a temporary copy. The script exits non-zero
if the baseline fails or any single-field mutation still verifies.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from services.datasets.golden import verify_seal

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = REPO_ROOT / "services/datasets/golden/artifacts"
LABELS_NAME = "golden.v1.labels.json"


@dataclass(frozen=True)
class Mutation:
    """One single-field mutation applied to the labels document."""

    name: str
    field: str
    value: Any


MUTATIONS = (
    Mutation("source_id", "source_id", "call-0001:99"),
    Mutation("call_id", "call_id", "call-0002"),
    Mutation("turn_index", "turn_index", 99),
    Mutation("corrected_transcript", "corrected_transcript", "tampered transcript"),
    Mutation("slots", "slots", {"guests": 2}),
    Mutation("first_reviewer_id", "first_reviewer_id", "reviewer-z"),
    Mutation("second_reviewer_id", "second_reviewer_id", "reviewer-z"),
    Mutation("agreed", "agreed", False),
    Mutation("labels_document_version", "__version__", "golden.v2"),
)


def main() -> int:
    """Print verification results for the baseline and each mutation."""
    with tempfile.TemporaryDirectory() as temporary:
        target = Path(temporary) / "artifacts"
        shutil.copytree(SOURCE_DIR, target)
        labels_path = target / LABELS_NAME
        original = labels_path.read_text(encoding="utf-8")
        baseline = verify_seal(target)
        print(f"baseline: ok={baseline.ok} errors={list(baseline.errors)}")
        failures = [] if baseline.ok else ["baseline"]

        for mutation in MUTATIONS:
            document = json.loads(original)
            if mutation.field == "__version__":
                document["version"] = mutation.value
            else:
                document["labels"][0][mutation.field] = mutation.value
            labels_path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
            report = verify_seal(target)
            print(f"{mutation.name}: ok={report.ok} errors={list(report.errors)}")
            if report.ok:
                failures.append(mutation.name)
            labels_path.write_text(original, encoding="utf-8")

    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("PASSED: baseline verifies and every mutation is rejected")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a script
    raise SystemExit(main())
