"""Deterministic static HTML/Markdown renderers for P2-11 dashboards.

Renderers are pure functions of :class:`DashboardData`. They perform no IO, use
no templates from a network, and escape all injected label text.
"""

from __future__ import annotations

import html
from collections.abc import Sequence

from voiceloop_common import ValidationError

from .data import DashboardData, MetricProvenance
from .settings import DashboardFormat

SYNTHETIC_BANNER = "SYNTHETIC / LOCAL DATA — NOT LIVE METRICS"
MIXED_BANNER = "MIXED SOURCES — VERIFY PROVENANCE BEFORE USE"
STATIC_NOTE = "Static render; no network or external dashboard service."
FOOTER = (
    "Metrics are read-only observations. No promotion, rollback, or call-path action is performed."
)
HEADERS = ("Section", "Metric", "Value", "Source", "Observed at", "Synthetic", "Stale", "Note")


def _observed_display(provenance: MetricProvenance) -> str:
    """Return an explicit unknown marker when no timestamp exists."""
    return provenance.observed_at or "unknown (no timestamp)"


def _stale_display(provenance: MetricProvenance) -> str:
    """Return a human-readable staleness label."""
    if not provenance.stale:
        return "no"
    if provenance.observed_at is None:
        return "yes — no timestamp"
    return "yes — older than freshness window"


def _rows(data: DashboardData) -> list[tuple[str, ...]]:
    """Flatten dashboard sections into escaped-by-later render string rows."""
    rows: list[tuple[str, ...]] = []
    for section in data.sections:
        for metric in section.metrics:
            provenance = metric.provenance
            rows.append(
                (
                    section.title,
                    metric.label,
                    metric.value,
                    provenance.source,
                    _observed_display(provenance),
                    "yes" if provenance.synthetic else "no",
                    _stale_display(provenance),
                    provenance.note or "-",
                )
            )
    return rows


def _markdown_cell(value: str) -> str:
    """Escape a value for one Markdown table cell."""
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _render_markdown(data: DashboardData, rows: Sequence[tuple[str, ...]]) -> str:
    """Render a deterministic Markdown table."""
    banner = SYNTHETIC_BANNER if data.synthetic else MIXED_BANNER
    lines = [
        f"# {data.title}",
        "",
        f"> **{banner}**",
        ">",
        f"> Generated at: {data.rendered_at}. {STATIC_NOTE}",
        "",
        "| " + " | ".join(HEADERS) + " |",
        "| " + " | ".join("---" for _ in HEADERS) + " |",
    ]
    lines.extend("| " + " | ".join(_markdown_cell(cell) for cell in row) + " |" for row in rows)
    lines.extend(
        [
            "",
            FOOTER,
            "",
        ]
    )
    return "\n".join(lines)


def _escape_cell(value: str) -> str:
    """Escape one HTML table cell value."""
    return html.escape(value, quote=True)


def _render_html(data: DashboardData, rows: Sequence[tuple[str, ...]]) -> str:
    """Render a self-contained, dependency-free HTML document."""
    banner = SYNTHETIC_BANNER if data.synthetic else MIXED_BANNER
    lines = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{_escape_cell(data.title)}</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:2rem;color:#222}",
        ".banner{background:#fff3cd;border:1px solid #e0a800;padding:.75rem;font-weight:bold}",
        "table{border-collapse:collapse;width:100%}",
        "th,td{border:1px solid #ccc;padding:.4rem;text-align:left;vertical-align:top}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{_escape_cell(data.title)}</h1>",
        f'<p class="banner">{_escape_cell(banner)}</p>',
        f"<p>Generated at: {_escape_cell(data.rendered_at)}. {STATIC_NOTE}</p>",
        "<table>",
        "<thead><tr>"
        + "".join(f"<th>{_escape_cell(header)}</th>" for header in HEADERS)
        + "</tr></thead>",
        "<tbody>",
    ]
    for row in rows:
        lines.append("<tr>" + "".join(f"<td>{_escape_cell(cell)}</td>" for cell in row) + "</tr>")
    lines.extend(
        [
            "</tbody>",
            "</table>",
            f"<p>{FOOTER}</p>",
            "</body>",
            "</html>",
            "",
        ]
    )
    return "\n".join(lines)


def render_dashboard(data: DashboardData, *, format: DashboardFormat = "html") -> str:
    """Render immutable dashboard data to a static artifact.

    Args:
        data: Data assembled by :func:`infra.dashboards.data.collect_dashboard`.
        format: ``"html"`` or ``"markdown"``.

    Returns:
        The complete rendered document as a string.

    Raises:
        ValidationError: If an unsupported format is requested.

    """
    rows = _rows(data)
    if format == "markdown":
        return _render_markdown(data, rows)
    if format == "html":
        return _render_html(data, rows)
    raise ValidationError(f"unsupported dashboard format: {format}")
