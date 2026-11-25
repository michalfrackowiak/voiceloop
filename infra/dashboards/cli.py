"""CLI entry point: ``python -m infra.dashboards.cli --output <path>``.

The command renders a static artifact from local synthetic fixtures and the
committed example snapshot by default. It never calls a network service.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

from pydantic import ValidationError as PydanticValidationError

from voiceloop_common import ConfigError, ValidationError

from .data import (
    FailureProvider,
    ModelStateProvider,
    QueueDepthProvider,
    ReviewThroughputProvider,
    collect_dashboard,
)
from .render import render_dashboard
from .settings import DashboardSettings


def _build_parser() -> argparse.ArgumentParser:
    """Build the dashboard CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m infra.dashboards.cli",
        description="Render a static VoiceLoop dashboard from local/injected data only.",
    )
    parser.add_argument(
        "--output", required=True, metavar="PATH", help="File path for the rendered artifact."
    )
    parser.add_argument(
        "--format",
        choices=["html", "markdown"],
        default=None,
        help="Artifact format; defaults to VOICELOOP_DASHBOARDS_FORMAT or html.",
    )
    parser.add_argument("--fixtures", metavar="PATH", help="Override the fixture labels manifest.")
    parser.add_argument("--snapshot", metavar="PATH", help="Override the local example snapshot.")
    parser.add_argument("--title", help="Override the dashboard title for this render.")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: DashboardSettings | None = None,
    failure_provider: FailureProvider | None = None,
    queue_provider: QueueDepthProvider | None = None,
    review_provider: ReviewThroughputProvider | None = None,
    model_provider: ModelStateProvider | None = None,
    stdout: TextIO | None = None,
) -> int:
    """Render the dashboard and write it to ``--output``.

    Args:
        argv: Argument list; ``sys.argv[1:]`` when omitted.
        settings: Injected typed settings, primarily for tests.
        failure_provider: Injected failure-label provider override.
        queue_provider: Injected queue-depth provider override.
        review_provider: Injected review-throughput provider override.
        model_provider: Injected model-state provider override.
        stdout: Injected text stream, primarily for tests.

    Returns:
        0 on success; 2 on configuration, input, render or write errors.

    """
    args = _build_parser().parse_args(argv)
    out = stdout if stdout is not None else sys.stdout
    try:
        updates: dict[str, Any] = {}
        if args.format is not None:
            updates["format"] = args.format
        if args.fixtures is not None:
            updates["fixtures_path"] = args.fixtures
        if args.snapshot is not None:
            updates["snapshot_path"] = args.snapshot
        if args.title is not None:
            updates["title"] = args.title
        config = (settings or DashboardSettings()).model_copy(update=updates)
        data = collect_dashboard(
            settings=config,
            failure_provider=failure_provider,
            queue_provider=queue_provider,
            review_provider=review_provider,
            model_provider=model_provider,
        )
        rendered = render_dashboard(data, format=config.format)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    except (ConfigError, ValidationError, PydanticValidationError, OSError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=out)
        return 2
    print(f"Wrote dashboard: {output}", file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
