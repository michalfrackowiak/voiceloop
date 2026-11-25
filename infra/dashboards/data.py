"""Data providers and immutable snapshots for the offline dashboard (P2-11).

Failure rate is computed from the read-only synthetic fixture labels manifest.
Queue depth, review throughput and model/promotion state are supplied through
protocols; the default providers read a clearly labelled local example snapshot
or report that no data is available. Nothing in this module infers a live value,
contacts a service, or performs a promotion.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from voiceloop_common import ConfigError, ValidationError, utcnow

from .settings import DashboardSettings

REPO_ROOT = Path(__file__).resolve().parents[2]

FAILURE_LABELS: frozenset[str] = frozenset({"stt_wrong", "parser_wrong", "unusable_audio"})
UNKNOWN_VERSION = "unknown / no version recorded"
UNKNOWN_PROMOTION = "unknown / no promotion recorded"
NO_QUEUE_DATA = "unavailable — no queue provider configured"
NO_REVIEW_DATA = "unavailable — no review provider configured"


@dataclass(frozen=True)
class MetricProvenance:
    """Source and freshness labels attached to every displayed metric."""

    source: str
    observed_at: str | None
    synthetic: bool
    stale: bool
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)


@dataclass(frozen=True)
class Metric:
    """One preformatted metric value with its provenance."""

    label: str
    value: str
    provenance: MetricProvenance

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)


@dataclass(frozen=True)
class DashboardSection:
    """A titled group of metrics."""

    title: str
    metrics: tuple[Metric, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)


@dataclass(frozen=True)
class DashboardData:
    """Complete deterministic input for a static dashboard render."""

    title: str
    rendered_at: str
    synthetic: bool
    sections: tuple[DashboardSection, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)


@dataclass(frozen=True)
class FailureSnapshot:
    """Counts derived from fixture failure labels (never from scores alone)."""

    total_turns: int
    failed_turns: int
    source: str
    observed_at: str | None = None
    synthetic: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        """Validate that counts are non-negative and bounded."""
        if self.total_turns < 0 or self.failed_turns < 0:
            raise ValidationError("failure counts must be non-negative")
        if self.failed_turns > self.total_turns:
            raise ValidationError("failed_turns cannot exceed total_turns")


@dataclass(frozen=True)
class QueueSnapshot:
    """Queue depth, or ``None`` when no queue source is configured."""

    depth: int | None
    source: str
    observed_at: str | None = None
    synthetic: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        """Validate that a present depth is non-negative."""
        if self.depth is not None and self.depth < 0:
            raise ValidationError("queue depth must be non-negative or null")


@dataclass(frozen=True)
class ReviewSnapshot:
    """Completed review count within an explicit observation window."""

    completed: int | None
    window_start: str | None
    window_end: str | None
    source: str
    observed_at: str | None = None
    synthetic: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        """Validate that a present completion count is non-negative."""
        if self.completed is not None and self.completed < 0:
            raise ValidationError("review completed count must be non-negative or null")


@dataclass(frozen=True)
class PromotionRecord:
    """Read-only promotion fact with its own provenance.

    ``alias`` defaults to ``@champion`` because context.md section 2 invariant
    10 defines promotion as an alias flip. This class never flips an alias.
    """

    version: str
    promoted_at: str
    source: str
    alias: str = "@champion"
    actor: str | None = None
    audit_ref: str | None = None

    def __post_init__(self) -> None:
        """Validate required promotion fields."""
        for name in ("version", "promoted_at", "source", "alias"):
            if not getattr(self, name).strip():
                raise ValidationError(f"promotion {name} must be a non-empty string")


@dataclass(frozen=True)
class ModelStateSnapshot:
    """In-service model version and last promotion from an injected source."""

    version_in_service: str | None
    promotion: PromotionRecord | None
    source: str
    observed_at: str | None = None
    synthetic: bool = False
    note: str = ""


@runtime_checkable
class FailureProvider(Protocol):
    """Supplies fixture-derived failure counts."""

    def failure_snapshot(self) -> FailureSnapshot:
        """Return a failure-count snapshot."""
        ...


@runtime_checkable
class QueueDepthProvider(Protocol):
    """Supplies an observed queue depth, or reports unavailability."""

    def queue_snapshot(self) -> QueueSnapshot:
        """Return a queue-depth snapshot."""
        ...


@runtime_checkable
class ReviewThroughputProvider(Protocol):
    """Supplies a completed-review count over a declared window."""

    def review_snapshot(self) -> ReviewSnapshot:
        """Return a review-throughput snapshot."""
        ...


@runtime_checkable
class ModelStateProvider(Protocol):
    """Supplies model version and promotion state with provenance."""

    def model_snapshot(self) -> ModelStateSnapshot:
        """Return a model-state snapshot."""
        ...


def resolve_local_path(value: str | Path) -> Path:
    """Resolve a repo-relative path against the repository root.

    Args:
        value: Absolute path, or path relative to the repository root.

    Returns:
        An absolute path; the input is not opened or modified here.

    """
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def failure_rate(snapshot: FailureSnapshot) -> float | None:
    """Return the failed/total ratio, or ``None`` when no turns were counted.

    Args:
        snapshot: Counts derived from fixture failure labels.

    Returns:
        A ratio in ``[0, 1]``, or ``None`` for an empty denominator.

    """
    if snapshot.total_turns == 0:
        return None
    return snapshot.failed_turns / snapshot.total_turns


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSON Lines file into dictionaries without modifying it."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read fixture labels at {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValidationError(f"expected a JSON object at {path}:{line_number}")
        rows.append(payload)
    return rows


@dataclass(frozen=True)
class FixtureFailureProvider:
    """Read-only failure-label provider for the synthetic fixture manifest."""

    path: str = "tests/fixtures/turns.jsonl"

    def failure_snapshot(self) -> FailureSnapshot:
        """Count fixture turns whose ``failure_type`` is a known failure label.

        Returns:
            A synthetic snapshot with no observation timestamp; the fixture
            manifest records labels, not when an observation was made.

        Raises:
            ValidationError: If a row lacks a label or is not marked synthetic.

        """
        rows = _read_jsonl(resolve_local_path(self.path))
        for line_number, row in enumerate(rows, start=1):
            label = row.get("failure_type")
            if not isinstance(label, str) or not label:
                raise ValidationError(f"fixture row {line_number} has no valid failure_type")
            if row.get("synthetic") is not True:
                raise ValidationError(f"fixture row {line_number} is not marked synthetic: true")
        failed = sum(1 for row in rows if row["failure_type"] in FAILURE_LABELS)
        return FailureSnapshot(
            total_turns=len(rows),
            failed_turns=failed,
            source=self.path,
            observed_at=None,
            synthetic=True,
            note="Synthetic fixture labels; triage scores are hints, not human labels.",
        )


def _optional_str(value: Any, field: str) -> str | None:
    """Return a non-empty string or ``None``, rejecting other JSON types."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field} must be a non-empty string or null")
    return value


def _required_str(payload: dict[str, Any], field: str) -> str:
    """Return a required non-empty string from a JSON object."""
    value = _optional_str(payload.get(field), field)
    if value is None:
        raise ValidationError(f"{field} is required")
    return value


def _optional_count(value: Any, field: str) -> int | None:
    """Return a non-negative integer or ``None``, rejecting booleans."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{field} must be a non-negative integer or null")
    return value


@dataclass(frozen=True)
class JsonSnapshotProvider:
    """Provider for a committed, explicitly labelled local JSON snapshot.

    The snapshot is an example/local data source, not a live adapter. Missing
    model version or promotion values stay unknown; no value is synthesised.
    """

    path: str = "infra/dashboards/examples/synthetic-snapshot.json"

    def _load(self) -> dict[str, Any]:
        """Load and validate the snapshot envelope."""
        path = resolve_local_path(self.path)
        try:
            payload: Any = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ConfigError(f"cannot read dashboard snapshot at {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ValidationError(f"invalid JSON in {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValidationError(f"dashboard snapshot at {path} must be a JSON object")
        if not isinstance(payload.get("synthetic"), bool):
            raise ValidationError(f"dashboard snapshot at {path} needs a boolean 'synthetic'")
        return payload

    def _section(self, payload: dict[str, Any], name: str) -> dict[str, Any]:
        """Return one required object section of the snapshot."""
        section = payload.get(name)
        if not isinstance(section, dict):
            raise ValidationError(f"dashboard snapshot section '{name}' must be an object")
        return section

    def _note(self, payload: dict[str, Any], section: dict[str, Any]) -> str:
        """Combine envelope and section notes into one display string."""
        parts = [
            _optional_str(payload.get("notice"), "notice"),
            _optional_str(section.get("note"), "note"),
        ]
        return "; ".join(part for part in parts if part)

    def _observed_at(self, payload: dict[str, Any], section: dict[str, Any]) -> str | None:
        """Prefer a section observation time over the envelope timestamp."""
        value = section.get("observed_at", payload.get("generated_at"))
        return _optional_str(value, "observed_at")

    def queue_snapshot(self) -> QueueSnapshot:
        """Return the snapshot's queue depth, or ``None`` when absent."""
        payload = self._load()
        section = self._section(payload, "queue")
        return QueueSnapshot(
            depth=_optional_count(section.get("depth"), "queue.depth"),
            source=self.path,
            observed_at=self._observed_at(payload, section),
            synthetic=bool(payload["synthetic"]),
            note=self._note(payload, section),
        )

    def review_snapshot(self) -> ReviewSnapshot:
        """Return the snapshot's review count and explicit time window."""
        payload = self._load()
        section = self._section(payload, "review")
        return ReviewSnapshot(
            completed=_optional_count(section.get("completed"), "review.completed"),
            window_start=_optional_str(section.get("window_start"), "review.window_start"),
            window_end=_optional_str(section.get("window_end"), "review.window_end"),
            source=self.path,
            observed_at=self._observed_at(payload, section),
            synthetic=bool(payload["synthetic"]),
            note=self._note(payload, section),
        )

    def model_snapshot(self) -> ModelStateSnapshot:
        """Return injected model state; null fields remain unknown."""
        payload = self._load()
        section = self._section(payload, "model")
        promotion_data = section.get("promotion")
        promotion: PromotionRecord | None = None
        if promotion_data is not None:
            if not isinstance(promotion_data, dict):
                raise ValidationError("model.promotion must be an object or null")
            promotion = PromotionRecord(
                version=_required_str(promotion_data, "version"),
                promoted_at=_required_str(promotion_data, "promoted_at"),
                source=_required_str(promotion_data, "source"),
                alias=_optional_str(promotion_data.get("alias"), "alias") or "@champion",
                actor=_optional_str(promotion_data.get("actor"), "actor"),
                audit_ref=_optional_str(promotion_data.get("audit_ref"), "audit_ref"),
            )
        return ModelStateSnapshot(
            version_in_service=_optional_str(
                section.get("version_in_service"), "model.version_in_service"
            ),
            promotion=promotion,
            source=self.path,
            observed_at=self._observed_at(payload, section),
            synthetic=bool(payload["synthetic"]),
            note=self._note(payload, section),
        )


@dataclass(frozen=True)
class FakeQueueDepthProvider:
    """Injected queue fake for tests and local demonstrations."""

    depth: int | None
    source: str = "fake://queue"
    observed_at: str | None = None
    synthetic: bool = True
    note: str = "Injected fake; not live queue data."

    def queue_snapshot(self) -> QueueSnapshot:
        """Return the configured fake queue snapshot."""
        return QueueSnapshot(
            depth=self.depth,
            source=self.source,
            observed_at=self.observed_at,
            synthetic=self.synthetic,
            note=self.note,
        )


@dataclass(frozen=True)
class FakeReviewThroughputProvider:
    """Injected review fake for tests and local demonstrations."""

    completed: int | None
    window_start: str | None = None
    window_end: str | None = None
    source: str = "fake://review"
    observed_at: str | None = None
    synthetic: bool = True
    note: str = "Injected fake; not live review data."

    def review_snapshot(self) -> ReviewSnapshot:
        """Return the configured fake review snapshot."""
        return ReviewSnapshot(
            completed=self.completed,
            window_start=self.window_start,
            window_end=self.window_end,
            source=self.source,
            observed_at=self.observed_at,
            synthetic=self.synthetic,
            note=self.note,
        )


@dataclass(frozen=True)
class FakeModelStateProvider:
    """Injected model-state fake for tests and local demonstrations."""

    version_in_service: str | None = None
    promotion: PromotionRecord | None = None
    source: str = "fake://model-state"
    observed_at: str | None = None
    synthetic: bool = True
    note: str = "Injected fake; not a real promotion record."

    def model_snapshot(self) -> ModelStateSnapshot:
        """Return the configured fake model-state snapshot."""
        return ModelStateSnapshot(
            version_in_service=self.version_in_service,
            promotion=self.promotion,
            source=self.source,
            observed_at=self.observed_at,
            synthetic=self.synthetic,
            note=self.note,
        )


@dataclass(frozen=True)
class UnavailableQueueDepthProvider:
    """Queue seam used when no queue source has been wired."""

    def queue_snapshot(self) -> QueueSnapshot:
        """Return an explicit unavailable marker, never a zero."""
        return QueueSnapshot(
            depth=None,
            source="not configured",
            observed_at=None,
            synthetic=True,
            note="No queue source configured; nothing inferred.",
        )


@dataclass(frozen=True)
class UnavailableReviewThroughputProvider:
    """Review seam used when no review source has been wired."""

    def review_snapshot(self) -> ReviewSnapshot:
        """Return an explicit unavailable marker, never a zero."""
        return ReviewSnapshot(
            completed=None,
            window_start=None,
            window_end=None,
            source="not configured",
            observed_at=None,
            synthetic=True,
            note="No review source configured; nothing inferred.",
        )


@dataclass(frozen=True)
class UnavailableModelStateProvider:
    """Model-state seam used when no release source has been wired."""

    def model_snapshot(self) -> ModelStateSnapshot:
        """Return unknown version and promotion, with no invented timestamp."""
        return ModelStateSnapshot(
            version_in_service=None,
            promotion=None,
            source="not configured",
            observed_at=None,
            synthetic=True,
            note="No model-state source configured; nothing inferred.",
        )


def _as_utc(value: datetime) -> datetime:
    """Attach UTC to a naive datetime; leave aware datetimes unchanged."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _parse_observed_at(value: str | None) -> datetime | None:
    """Parse an RFC3339-like timestamp, returning ``None`` when unusable."""
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return _as_utc(parsed)


def _format_moment(value: datetime) -> str:
    """Format a UTC datetime deterministically for display."""
    return _as_utc(value).astimezone(UTC).isoformat().replace("+00:00", "Z")


def _provenance(
    source: str,
    observed_at: str | None,
    synthetic: bool,
    note: str,
    *,
    now: datetime,
    stale_after_minutes: int,
) -> MetricProvenance:
    """Build provenance and compute the display-only staleness flag."""
    observed = _parse_observed_at(observed_at)
    stale = observed is None or (now - observed) > timedelta(minutes=stale_after_minutes)
    return MetricProvenance(
        source=source,
        observed_at=observed_at,
        synthetic=synthetic,
        stale=stale,
        note=note,
    )


def _review_value(snapshot: ReviewSnapshot) -> str:
    """Format review throughput without inventing a per-time rate."""
    if snapshot.completed is None:
        return NO_REVIEW_DATA
    if snapshot.window_start and snapshot.window_end:
        window = f"{snapshot.window_start} to {snapshot.window_end}"
    else:
        window = "window unknown"
    return f"{snapshot.completed} completed | {window}"


def _promotion_value(promotion: PromotionRecord | None) -> str:
    """Format the last promotion, or the mandated unknown fallback."""
    if promotion is None:
        return UNKNOWN_PROMOTION
    details = f"{promotion.version} at {promotion.promoted_at} ({promotion.alias})"
    if promotion.actor:
        details += f" by {promotion.actor}"
    if promotion.audit_ref:
        details += f" audit={promotion.audit_ref}"
    return details


def collect_dashboard(
    *,
    settings: DashboardSettings,
    failure_provider: FailureProvider | None = None,
    queue_provider: QueueDepthProvider | None = None,
    review_provider: ReviewThroughputProvider | None = None,
    model_provider: ModelStateProvider | None = None,
    now: datetime | None = None,
) -> DashboardData:
    """Assemble deterministic dashboard data from local defaults and fakes.

    Args:
        settings: Typed local settings for paths, title and freshness window.
        failure_provider: Optional fixture-label provider; defaults to the
            read-only synthetic fixture manifest.
        queue_provider: Optional queue seam; defaults to the example snapshot.
        review_provider: Optional review seam; defaults to the example snapshot.
        model_provider: Optional model-state seam; defaults to the example
            snapshot, which records no version and no promotion.
        now: Injected UTC clock value for deterministic rendering.

    Returns:
        Immutable dashboard data with provenance on every metric.

    """
    moment = _as_utc(now if now is not None else utcnow())
    failures = (
        failure_provider or FixtureFailureProvider(settings.fixtures_path)
    ).failure_snapshot()
    queue = (queue_provider or JsonSnapshotProvider(settings.snapshot_path)).queue_snapshot()
    review = (review_provider or JsonSnapshotProvider(settings.snapshot_path)).review_snapshot()
    model = (model_provider or JsonSnapshotProvider(settings.snapshot_path)).model_snapshot()

    def provenance(
        source: str, observed_at: str | None, synthetic: bool, note: str
    ) -> MetricProvenance:
        """Build provenance using this call's clock and freshness setting."""
        return _provenance(
            source,
            observed_at,
            synthetic,
            note,
            now=moment,
            stale_after_minutes=settings.stale_after_minutes,
        )

    rate = failure_rate(failures)
    failure_display = (
        "n/a (0 turns)"
        if rate is None
        else f"{failures.failed_turns}/{failures.total_turns} ({rate:.1%})"
    )
    failure_prov = provenance(
        failures.source, failures.observed_at, failures.synthetic, failures.note
    )
    queue_prov = provenance(queue.source, queue.observed_at, queue.synthetic, queue.note)
    review_prov = provenance(review.source, review.observed_at, review.synthetic, review.note)

    version_prov = provenance(
        model.source,
        model.observed_at,
        model.synthetic,
        model.note or "Version supplied by injected state; never inferred from a score.",
    )
    promotion_source = model.source
    promotion_note = model.note
    if model.promotion is not None:
        promotion_source = f"{model.source}; promotion: {model.promotion.source}"
        audit = f" audit={model.promotion.audit_ref}" if model.promotion.audit_ref else ""
        promotion_note = (
            f"{promotion_note} Promotion alias flip record{audit}."
            if promotion_note
            else (f"Promotion alias flip record{audit}.")
        )
    promotion_prov = provenance(
        promotion_source,
        model.observed_at,
        model.synthetic,
        promotion_note,
    )

    metrics = (
        Metric("Failure rate (fixture labels)", failure_display, failure_prov),
        Metric("Turns counted", str(failures.total_turns), failure_prov),
        Metric(
            "Queue depth",
            NO_QUEUE_DATA if queue.depth is None else str(queue.depth),
            queue_prov,
        ),
        Metric("Review throughput (completed in window)", _review_value(review), review_prov),
        Metric(
            "Model version in service",
            model.version_in_service or UNKNOWN_VERSION,
            version_prov,
        ),
        Metric("Last promotion", _promotion_value(model.promotion), promotion_prov),
    )
    sections = (
        DashboardSection("Operational signals (P2-11.1)", metrics[:4]),
        DashboardSection("Release state (P2-11.2)", metrics[4:]),
    )
    return DashboardData(
        title=settings.title,
        rendered_at=_format_moment(moment),
        synthetic=all(metric.provenance.synthetic for metric in metrics),
        sections=sections,
    )
