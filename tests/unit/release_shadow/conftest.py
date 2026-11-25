"""Shared deterministic fake world for P3-07 tests (no sleeps or services)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from services.release.shadow import (
    BoundedMirrorSink,
    CollectingAuditSink,
    FakeClock,
    FakeOutcomeSource,
    MirrorStats,
    ModelSource,
    OutcomeRecord,
    ScriptedModelRunner,
    ShadowMirror,
    ShadowRunner,
    ShadowSettings,
    ShadowTurn,
)
from services.release.shadow.settings import DEFAULT_IGNORED_OUTCOME_FIELDS


@dataclass
class World:
    """One wired fake shadow stack with injectable behaviour knobs."""

    settings: ShadowSettings
    clock: FakeClock
    audit: CollectingAuditSink
    sink: BoundedMirrorSink
    mirror: ShadowMirror
    candidate: ScriptedModelRunner
    champion: ScriptedModelRunner
    runner: ShadowRunner
    outcomes: FakeOutcomeSource

    def add_turn(
        self,
        call_id: str,
        turn_index: int,
        *,
        fields: dict[str, Any] | None = None,
        champion_fields: dict[str, Any] | None = None,
        synthetic: bool = True,
        audio_uri: str | None = None,
    ) -> ShadowTurn:
        """Create a safe turn reference and script both fake runners."""
        turn = ShadowTurn(
            call_id=call_id,
            turn_index=turn_index,
            audio_uri=audio_uri or f"fake://audio/{call_id}/{turn_index}.wav",
            synthetic=synthetic,
            observed_at=self.clock.now(),
        )
        if fields is not None:
            self.candidate.set_slots(call_id, turn_index, fields)
        if champion_fields is not None:
            self.champion.set_slots(call_id, turn_index, champion_fields)
        return turn

    def add_outcome(
        self,
        call_id: str,
        status: str,
        final_record: dict[str, Any],
        *,
        synthetic: bool = True,
    ) -> None:
        """Add a call-level outcome projected to digests immediately."""
        self.outcomes.add(
            call_id,
            OutcomeRecord.from_payload(
                call_id,
                {"status": status, "final_record": final_record, "updated_at": ""},
                self.settings.ignored_outcome_fields(),
                synthetic,
            ),
        )

    def drain(self) -> int:
        """Drain one configured batch of queued shadow work."""
        return self.runner.drain_once()

    @property
    def stats(self) -> MirrorStats:
        """Return the mirror's cumulative counters."""
        return self.mirror.stats


@pytest.fixture
def shadow_settings(tmp_path: Path) -> ShadowSettings:
    """Return settings with a small capacity and a temp report path."""
    return ShadowSettings(
        max_pending=4,
        mirror_budget_ms=2.0,
        max_turns_per_drain=8,
        output_path=str(tmp_path / "comparison.json"),
        ignored_outcome_fields_csv=DEFAULT_IGNORED_OUTCOME_FIELDS,
    )


@pytest.fixture
def world_factory(shadow_settings: ShadowSettings) -> Callable[..., World]:
    """Return a factory that wires an isolated fake shadow stack."""

    def make(
        *,
        max_pending: int = 4,
        budget_ms: float = 2.0,
        max_turns_per_drain: int = 8,
        candidate_fail: bool = False,
        champion_fail: bool = False,
        candidate_steps: float = 0.0,
        champion_steps: float = 0.0,
        error_message: str = "fake runner failure",
    ) -> World:
        clock = FakeClock()
        settings = ShadowSettings(
            task_id=shadow_settings.task_id,
            max_pending=max_pending,
            mirror_budget_ms=budget_ms,
            max_turns_per_drain=max_turns_per_drain,
            output_path=shadow_settings.output_path,
            fixtures_path=shadow_settings.fixtures_path,
            scorable_statuses_csv=shadow_settings.scorable_statuses_csv,
            ignored_outcome_fields_csv=shadow_settings.ignored_outcome_fields_csv,
        )
        audit = CollectingAuditSink()
        sink = BoundedMirrorSink(clock=clock, capacity=max_pending)
        mirror = ShadowMirror(settings, sink, clock, audit)
        candidate = ScriptedModelRunner(
            source=ModelSource.CANDIDATE,
            artifact_uri="fake://shadow/candidate-v1",
            model_id="fake-candidate",
            clock=clock,
            clock_steps=candidate_steps,
            fail=candidate_fail,
            error_message=error_message,
        )
        champion = ScriptedModelRunner(
            source=ModelSource.CHAMPION,
            artifact_uri="fake://shadow/champion-v1",
            model_id="fake-champion",
            clock=clock,
            clock_steps=champion_steps,
        )
        runner = ShadowRunner(settings, sink, candidate, champion, clock, audit)
        return World(
            settings=settings,
            clock=clock,
            audit=audit,
            sink=sink,
            mirror=mirror,
            candidate=candidate,
            champion=champion,
            runner=runner,
            outcomes=FakeOutcomeSource({}),
        )

    return make
