"""Offline CLI for the P4-03 synthetic review UI harness."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .build import build_artifacts
from .constants import GO_NO_GO_PATH, SYNTHETIC_LABEL, USABILITY_ACCEPTANCE, USABILITY_STATUS
from .errors import UiValidationError
from .go_no_go import decide_from_files
from .hashes import verify_read_only_inputs


def main(argv: Sequence[str] | None = None) -> int:
    """Run hash verification, go/no-go decisioning and static artifact build.

    Returns:
        ``0`` on success; ``2`` on a fail-closed validation error. No network,
        audio playback, human study or promotion decision is performed.

    """
    parser = argparse.ArgumentParser(
        description="Deterministic P4-03 synthetic review UI harness (no real usability evidence)."
    )
    parser.add_argument("--verify-hashes", action="store_true", help="Print pinned hash checks.")
    parser.add_argument("--go-no-go", action="store_true", help="Decide from P2-04 metrics.")
    parser.add_argument("--metrics", type=Path, help="Read-only P2-04 metrics report path.")
    parser.add_argument("--measured", type=Path, help="Optional real-click/time/edit comparison.")
    parser.add_argument("--output", type=Path, help="Go/no-go report destination.")
    parser.add_argument("--build", action="store_true", help="Regenerate static HTML and JSON.")
    args = parser.parse_args(argv)
    try:
        if args.verify_hashes:
            print(json.dumps(verify_read_only_inputs().to_dict(), indent=2, sort_keys=True))
        if args.go_no_go:
            if args.metrics is None:
                raise UiValidationError("--metrics is required with --go-no-go")
            report = decide_from_files(args.metrics, args.measured, args.output or GO_NO_GO_PATH)
            print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        if args.build:
            paths = build_artifacts(decision=None)
            print(json.dumps(paths, indent=2, sort_keys=True))
        if not (args.verify_hashes or args.go_no_go or args.build):
            parser.error("choose --verify-hashes, --go-no-go and/or --build")
    except (UiValidationError, OSError, json.JSONDecodeError) as exc:
        print(
            json.dumps({"code": "validation_error", "message": str(exc)}),
            file=sys.stderr,
        )
        return 2
    print(f"STATUS: {USABILITY_STATUS} | {SYNTHETIC_LABEL} | {USABILITY_ACCEPTANCE}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via python -m
    raise SystemExit(main())
