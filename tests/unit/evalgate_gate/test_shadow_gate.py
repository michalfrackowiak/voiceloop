"""P3-08.1: gate refuses STT promotion without valid shadow evidence."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from services.evalgate.gate import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    CollectingRefusalLog,
    CollectingShadowEvidenceSource,
    GateError,
    PromotionRefusalLog,
    ShadowEvidenceSettings,
    StaticShadowEvidenceSource,
)
from tests.unit.evalgate_gate.conftest import make_report
from tests.unit.evalgate_gate.shadow_fakes import (
    CANDIDATE_URI,
    CHAMPION_URI,
    NOW,
    REPORT_URI,
    make_shadow_report,
    make_stt_gate,
)

SOURCE_URI = "checkpoint://stt-v4"
TIMESTAMP = NOW.isoformat(timespec="seconds").replace("+00:00", "Z")


def _champion() -> dict[str, Any]:
    """Return a champion eval report with full slice coverage and fixed time."""
    report = make_report(artifact_uri=CHAMPION_URI, e2e_accuracy=0.70)
    report["generated_at"] = TIMESTAMP
    return report


def _better() -> dict[str, Any]:
    """Return a challenger eval report that clears the P2-09 rules."""
    report = make_report(artifact_uri=CANDIDATE_URI, e2e_accuracy=0.80, e2e_delta_lower=0.05)
    report["generated_at"] = TIMESTAMP
    return report


def _source(report: dict[str, Any] | None = None) -> StaticShadowEvidenceSource:
    """Return a fake shadow source keyed by the v4/v3 version pair."""
    reports = {} if report is None else {("v4", "v3"): report}
    return StaticShadowEvidenceSource(reports)


def _run(gate: Any, *, reason: str = "shadow rehearsal", change: str = "stt") -> Any:
    """Run the gate with an STT change by default."""
    return gate.run(
        SOURCE_URI,
        _champion(),
        challenger_version="v4",
        actor_id="ml-engineer",
        reason=reason,
        change_kind=change,
    )


def test_stt_promotion_without_shadow_source_is_refused(tmp_path: Path) -> None:
    log = CollectingRefusalLog()
    gate, store, promotion_log = make_stt_gate(
        tmp_path, _better(), shadow_source=None, refusal_log=log
    )

    result = _run(gate)

    assert result.decision.passed is False
    assert result.refusal is not None
    assert result.refusal.code == "shadow_evidence_missing"
    assert result.promotion is None
    assert store.snapshot() == {CHAMPION_ALIAS: "v3", CHALLENGER_ALIAS: "v4"}
    assert promotion_log.entries() == ()
    assert len(log.records) == 1
    assert log.records[0].action == "shadow_refusal"


def test_valid_shadow_evidence_allows_stt_promotion(tmp_path: Path) -> None:
    report = make_shadow_report()
    source = CollectingShadowEvidenceSource(_source(report))
    log = CollectingRefusalLog()
    gate, store, promotion_log = make_stt_gate(
        tmp_path, _better(), shadow_source=source, refusal_log=log
    )

    result = _run(gate)

    assert result.decision.passed is True
    assert result.refusal is None
    assert result.promotion is not None
    assert result.promotion.before_version == "v3"
    assert result.promotion.after_version == "v4"
    assert result.shadow_evidence is not None
    assert result.shadow_evidence.turns_observed == 36
    assert store.get(CHAMPION_ALIAS) == "v4"
    assert promotion_log.verify_chain() == ()
    assert log.records == []
    assert len(source.requests) == 1
    assert source.requests[0].candidate_artifact_uri == CANDIDATE_URI
    assert source.requests[0].champion_artifact_uri == CHAMPION_URI
    assert source.requests[0].evaluated_at == TIMESTAMP


def test_shadow_source_is_read_only(tmp_path: Path) -> None:
    report = make_shadow_report()
    before = deepcopy(report)
    gate, _, _ = make_stt_gate(
        tmp_path, _better(), shadow_source=_source(report), refusal_log=CollectingRefusalLog()
    )

    result = _run(gate)

    assert result.decision.passed is True
    assert report == before


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda r: r.update({"promotion_decision": "promote"}), "shadow_evidence_promotion_claim"),
        (
            lambda r: r["provenance"].update({"synthetic": False}),
            "shadow_evidence_not_synthetic",
        ),
        (
            lambda r: r["sample_counts"].update({"turns_observed": 0}),
            "shadow_evidence_incomplete",
        ),
        (
            lambda r: r["provenance"]["models"][0].update(
                {"artifact_uri": "converted://other-stt"}
            ),
            "shadow_evidence_mismatch",
        ),
    ],
)
def test_bad_shadow_reports_refuse_without_side_effects(
    tmp_path: Path, mutate: Any, code: str
) -> None:
    report = make_shadow_report()
    mutate(report)
    log = CollectingRefusalLog()
    gate, store, promotion_log = make_stt_gate(
        tmp_path, _better(), shadow_source=_source(report), refusal_log=log
    )

    result = _run(gate)

    assert result.decision.passed is False
    assert result.refusal is not None
    assert result.refusal.code == code
    assert result.promotion is None
    assert store.get(CHAMPION_ALIAS) == "v3"
    assert promotion_log.entries() == ()
    assert len(log.records) == 1


def test_shadow_report_without_matching_version_pair_is_missing(tmp_path: Path) -> None:
    report = make_shadow_report()
    gate, store, promotion_log = make_stt_gate(
        tmp_path,
        _better(),
        shadow_source=StaticShadowEvidenceSource({("v9", "v3"): report}),
        refusal_log=CollectingRefusalLog(),
    )

    result = _run(gate)

    assert result.refusal is not None
    assert result.refusal.code == "shadow_evidence_missing"
    assert store.get(CHAMPION_ALIAS) == "v3"
    assert promotion_log.entries() == ()


def test_stale_shadow_report_refuses(tmp_path: Path) -> None:
    stale = make_shadow_report(
        generated_at=(NOW - timedelta(hours=25))
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    gate, store, promotion_log = make_stt_gate(
        tmp_path, _better(), shadow_source=_source(stale), refusal_log=CollectingRefusalLog()
    )

    result = _run(gate)

    assert result.refusal is not None
    assert result.refusal.code == "shadow_evidence_stale"
    assert result.promotion is None
    assert store.get(CHAMPION_ALIAS) == "v3"
    assert promotion_log.entries() == ()


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda r: r["sample_counts"].update({"calls_scored": 37}),
            "shadow_evidence_incomplete",
        ),
        (
            lambda r: r["sample_counts"].update({"results_candidate": 37}),
            "shadow_evidence_incomplete",
        ),
        (
            lambda r: r["sample_counts"].update({"results_champion": 37}),
            "shadow_evidence_incomplete",
        ),
        (
            lambda r: r["provenance"].update({"successful_results": 13}),
            "shadow_evidence_incomplete",
        ),
        (
            lambda r: r.update(
                {
                    "generated_at": (NOW + timedelta(seconds=61))
                    .isoformat(timespec="seconds")
                    .replace("+00:00", "Z")
                }
            ),
            "shadow_evidence_stale",
        ),
    ],
)
def test_inconsistent_counts_or_future_evidence_refuse_without_side_effects(
    tmp_path: Path, mutate: Any, code: str
) -> None:
    """New bounds never flip an alias, write a promotion audit or edit thresholds."""
    report = make_shadow_report()
    mutate(report)
    log = CollectingRefusalLog()
    gate, store, promotion_log = make_stt_gate(
        tmp_path, _better(), shadow_source=_source(report), refusal_log=log
    )
    aliases_before = store.snapshot()
    gate_settings_before = gate._settings.model_copy(deep=True)
    shadow_settings_before = gate._shadow_settings.model_copy(deep=True)

    result = _run(gate)

    assert result.decision.passed is False
    assert result.refusal is not None
    assert result.refusal.code == code
    assert result.promotion is None
    assert store.snapshot() == aliases_before == {CHAMPION_ALIAS: "v3", CHALLENGER_ALIAS: "v4"}
    assert promotion_log.entries() == ()
    assert gate._settings == gate_settings_before
    assert gate._shadow_settings == shadow_settings_before
    assert len(log.records) == 1
    assert log.records[0].code == code


def test_shadow_delta_does_not_override_failed_e2e_rule(tmp_path: Path) -> None:
    worse = make_report(artifact_uri=CANDIDATE_URI, e2e_accuracy=0.60, e2e_delta_lower=-0.05)
    worse["generated_at"] = TIMESTAMP
    log = CollectingRefusalLog()
    gate, store, promotion_log = make_stt_gate(
        tmp_path, worse, shadow_source=_source(make_shadow_report()), refusal_log=log
    )

    result = _run(gate)

    assert result.decision.passed is False
    assert "does not exceed required margin" in " ".join(result.decision.reasons)
    assert result.refusal is None
    assert result.promotion is None
    assert store.get(CHAMPION_ALIAS) == "v3"
    assert promotion_log.entries() == ()
    assert log.records == []


def test_shadow_delta_does_not_override_critical_slice_regression(tmp_path: Path) -> None:
    regressing = make_report(
        artifact_uri=CANDIDATE_URI,
        e2e_accuracy=0.90,
        e2e_delta_lower=0.20,
        slice_values={"names": (10.0, 0.55)},
    )
    regressing["generated_at"] = TIMESTAMP
    gate, store, promotion_log = make_stt_gate(
        tmp_path,
        regressing,
        shadow_source=_source(make_shadow_report()),
        refusal_log=CollectingRefusalLog(),
    )

    result = _run(gate)

    assert result.decision.passed is False
    assert "names: e2e regression" in " ".join(result.decision.reasons)
    assert result.promotion is None
    assert store.get(CHAMPION_ALIAS) == "v3"
    assert promotion_log.entries() == ()


def test_parser_path_keeps_p2_09_behaviour_without_shadow_source(tmp_path: Path) -> None:
    gate, store, _ = make_stt_gate(
        tmp_path, _better(), shadow_source=None, refusal_log=CollectingRefusalLog()
    )

    result = _run(gate, change="parser")

    assert result.decision.passed is True
    assert result.refusal is None
    assert result.promotion is not None
    assert store.get(CHAMPION_ALIAS) == "v4"


def test_invalid_change_kind_is_refused(tmp_path: Path) -> None:
    gate, store, promotion_log = make_stt_gate(
        tmp_path,
        _better(),
        shadow_source=_source(make_shadow_report()),
        refusal_log=CollectingRefusalLog(),
    )

    with pytest.raises(GateError, match="unknown change kind"):
        _run(gate, change="unknown")

    assert store.get(CHAMPION_ALIAS) == "v3"
    assert promotion_log.entries() == ()


def test_refusal_audit_is_hash_chained_and_pii_free(tmp_path: Path) -> None:
    log = PromotionRefusalLog(tmp_path / "refusals.jsonl")
    report = make_shadow_report()
    report["provenance"]["network_calls"] = True
    gate, _, _ = make_stt_gate(tmp_path, _better(), shadow_source=_source(report), refusal_log=log)

    result = _run(gate)

    assert result.refusal is not None
    assert log.verify_chain() == ()
    serialized = json.dumps([entry.to_dict() for entry in log.entries()])
    for forbidden in ("Ada Fixture", "ada.fixture@example.com", "+1 555", "4111", "audio_bytes"):
        assert forbidden not in serialized
    assert log.entries()[0].code == result.refusal.code


def test_refusal_audit_detects_tampering(tmp_path: Path) -> None:
    log = PromotionRefusalLog(tmp_path / "refusals.jsonl")
    gate, _, _ = make_stt_gate(tmp_path, _better(), shadow_source=None, refusal_log=log)

    _run(gate)
    _run(gate, reason="second attempt")
    assert log.verify_chain() == ()

    path = tmp_path / "refusals.jsonl"
    path.write_text(
        path.read_text(encoding="utf-8").replace("shadow_evidence_missing", "promoted"),
        encoding="utf-8",
    )

    assert any("hash mismatch" in error for error in log.verify_chain())
    assert REPORT_URI not in path.read_text(encoding="utf-8")


def test_configured_evidence_minimums_are_applied(tmp_path: Path) -> None:
    report = make_shadow_report(turns_observed=3, calls_scored=3, successful_results=1)
    settings = ShadowEvidenceSettings(min_turns_observed=4)
    log = CollectingRefusalLog()
    gate, store, promotion_log = make_stt_gate(
        tmp_path,
        _better(),
        shadow_source=_source(report),
        refusal_log=log,
        shadow_settings=settings,
    )

    result = _run(gate)

    assert result.refusal is not None
    assert result.refusal.code == "shadow_evidence_incomplete"
    assert result.promotion is None
    assert store.get(CHAMPION_ALIAS) == "v3"
    assert promotion_log.entries() == ()
