"""Minimal smoke test that keeps CI green on an empty skeleton."""

import voiceloop_common


def test_skeleton_smoke() -> None:
    """The shared package imports and exposes its core helpers."""
    assert callable(voiceloop_common.utcnow)
    assert callable(voiceloop_common.new_id)
    assert True
